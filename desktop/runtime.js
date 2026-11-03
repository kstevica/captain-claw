/**
 * Captain Claw – the desktop app's own Python, installed and updated from a manifest.
 *
 * The app ships without its backend. On first launch it reads the release manifest
 * (scripts/desktop_release.py writes it; by default the one on the latest GitHub
 * release) and installs, into the runtime folder:
 *
 *   bin/uv          uv, pinned by the manifest (version + sha256 per platform)
 *   python/         the Python uv installs (never the system's)
 *   cache/          uv's cache: an update re-downloads only what changed
 *   envs/<version>/ one venv per Captain Claw version: the locked dependencies
 *                   (every one hash-checked) and the Captain Claw wheel, which
 *                   carries the backend and the Flight Deck frontend
 *   state.json      which version runs (current), the one before it (previous),
 *                   and one installed and waiting for a restart (pending)
 *
 * An update installs the new version beside the running one, test-starts it on a
 * spare port with an empty data folder, and only then switches. The old venv stays
 * until the one after, so going back is a pointer flip, not a reinstall.
 */

const { net } = require("electron");
const { spawn } = require("child_process");
const crypto = require("crypto");
const fs = require("fs");
const fsp = fs.promises;
const http = require("http");
const netSock = require("net");
const os = require("os");
const path = require("path");
const log = require("electron-log/main");

// What this shell understands of the manifest. A manifest whose min_shell is higher
// needs a newer app (scripts/desktop_release.py MIN_SHELL).
const SHELL_PROTOCOL = 1;
const DEFAULT_MANIFEST_URL =
  "https://github.com/kstevica/captain-claw/releases/latest/download/desktop-manifest.json";
const READY_MARKER = ".captain-claw-desktop.json";
const IS_WIN = process.platform === "win32";

// ── Where things live ──────────────────────────────────────────

function manifestUrl() {
  return process.env.CAPTAIN_CLAW_MANIFEST_URL || DEFAULT_MANIFEST_URL;
}

// Not in Roaming on Windows (it syncs), not in ~/.config on Linux (it's not config).
function runtimeDir() {
  if (process.env.CAPTAIN_CLAW_RUNTIME_DIR) return process.env.CAPTAIN_CLAW_RUNTIME_DIR;
  if (IS_WIN) {
    return path.join(process.env.LOCALAPPDATA || path.join(os.homedir(), "AppData", "Local"), "Captain", "runtime");
  }
  if (process.platform === "linux") {
    const data = process.env.XDG_DATA_HOME || path.join(os.homedir(), ".local", "share");
    return path.join(data, "captain-claw-desktop", "runtime");
  }
  return path.join(os.homedir(), "Library", "Application Support", "Captain", "runtime");
}

const uvPath = () => path.join(runtimeDir(), "bin", IS_WIN ? "uv.exe" : "uv");
const envDir = (version) => path.join(runtimeDir(), "envs", version);
const envBin = (version) => path.join(envDir(version), IS_WIN ? "Scripts" : "bin");
const envPython = (version) => path.join(envBin(version), IS_WIN ? "python.exe" : "python");
const stateFile = () => path.join(runtimeDir(), "state.json");
const downloadsDir = () => path.join(runtimeDir(), "downloads");

function platformKey() {
  return `${process.platform}-${process.arch}`;
}

// ── state.json ─────────────────────────────────────────────────

function readState() {
  try {
    return JSON.parse(fs.readFileSync(stateFile(), "utf8")) || {};
  } catch {
    return {};
  }
}

function writeState(changes) {
  const next = { ...readState(), ...changes };
  fs.mkdirSync(runtimeDir(), { recursive: true });
  const tmp = `${stateFile()}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(next, null, 2));
  fs.renameSync(tmp, stateFile());
  return next;
}

function isReady(version) {
  return Boolean(version) && fs.existsSync(path.join(envDir(version), READY_MARKER)) && fs.existsSync(envPython(version));
}

// The version that starts now. A pending one (installed and test-started, waiting
// for a restart) takes over here; the caller rolls back if it then doesn't come up.
function takePending() {
  const state = readState();
  if (!state.pending || state.pending === state.current) {
    if (state.pending) writeState({ pending: null });
    return null;
  }
  if (!isReady(state.pending)) {
    writeState({ pending: null });
    return null;
  }
  const from = state.current || null;
  writeState({ current: state.pending, previous: from, pending: null });
  log.info(`Switched to Captain ${state.pending} (from ${from})`);
  return { from, to: state.pending };
}

function currentVersion() {
  const { current } = readState();
  return isReady(current) ? current : null;
}

function switchTo(version) {
  const state = readState();
  if (state.current === version) return;
  writeState({ current: version, previous: state.current || null, pending: null });
}

function rollBack(failed) {
  const state = readState();
  const back = state.previous && isReady(state.previous) ? state.previous : null;
  writeState({ current: back, previous: null, pending: null, failed });
  log.warn(`Went back from ${failed} to ${back}`);
  return back;
}

// ── What the shell starts ──────────────────────────────────────

// An environment without anything that could steer Python or uv elsewhere: a
// developer's activated venv or PYTHONPATH must not reach the app's own Python.
function cleanEnv() {
  const env = { ...process.env };
  for (const key of Object.keys(env)) {
    if (/^(UV_|PIP_|CONDA_)/.test(key) || ["PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONSTARTUP", "PYTHONUSERBASE"].includes(key)) {
      delete env[key];
    }
  }
  return env;
}

// How main.js starts a module of the current version.
function pythonFor(version) {
  const env = cleanEnv();
  env.PATH = [envBin(version), env.PATH || ""].filter(Boolean).join(path.delimiter);
  env.CAPTAIN_CLAW_DESKTOP_RUNTIME = version;
  env.PYTHONNOUSERSITE = "1";
  return { cmd: envPython(version), env };
}

// ── Manifest and downloads ─────────────────────────────────────

function compareVersions(a, b) {
  const parse = (v) => {
    const m = String(v).match(/^(\d+(?:\.\d+)*)(.*)$/);
    return m ? { nums: m[1].split(".").map(Number), pre: m[2] } : { nums: [0], pre: String(v) };
  };
  const x = parse(a);
  const y = parse(b);
  for (let i = 0; i < Math.max(x.nums.length, y.nums.length); i++) {
    const d = (x.nums[i] || 0) - (y.nums[i] || 0);
    if (d) return d;
  }
  // 0.8.6rc1 comes before 0.8.6.
  if (x.pre === y.pre) return 0;
  if (!x.pre) return 1;
  if (!y.pre) return -1;
  return x.pre < y.pre ? -1 : 1;
}

async function fetchManifest() {
  const url = manifestUrl();
  let text;
  if (url.startsWith("file:")) {
    text = await fsp.readFile(new URL(url), "utf8");
  } else {
    const res = await net.fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`The release list answered ${res.status} (${url})`);
    text = await res.text();
  }
  let manifest;
  try {
    manifest = JSON.parse(text);
  } catch {
    throw new Error(`The release list isn't readable (${url})`);
  }
  for (const key of ["version", "python", "wheel", "lock", "uv"]) {
    if (!manifest[key]) throw new Error(`The release list has no ${key} (${url})`);
  }
  if (!/^[0-9][0-9A-Za-z.+-]*$/.test(manifest.version)) throw new Error(`Odd version in the release list: ${manifest.version}`);
  manifest.shellTooOld = (manifest.min_shell || 1) > SHELL_PROTOCOL;
  return manifest;
}

// Saves *url* to *dest*, checking its sha256 on the way; *onProgress(done, total)*.
async function download(url, dest, { sha256, size, onProgress } = {}) {
  await fsp.mkdir(path.dirname(dest), { recursive: true });
  const part = `${dest}.part`;
  const hash = crypto.createHash("sha256");
  const out = fs.createWriteStream(part);
  let done = 0;
  const write = async (chunk) => {
    hash.update(chunk);
    done += chunk.length;
    if (!out.write(chunk)) await new Promise((resolve) => out.once("drain", resolve));
    onProgress?.(done, size || 0);
  };
  try {
    if (url.startsWith("file:")) {
      for await (const chunk of fs.createReadStream(new URL(url))) await write(chunk);
    } else {
      const res = await net.fetch(url, { cache: "no-store" });
      if (!res.ok) throw new Error(`Download failed: ${res.status} (${url})`);
      size = Number(res.headers.get("content-length")) || size;
      const reader = res.body.getReader();
      for (;;) {
        const { done: end, value } = await reader.read();
        if (end) break;
        await write(Buffer.from(value));
      }
    }
  } finally {
    await new Promise((resolve) => out.end(resolve));
  }
  const digest = hash.digest("hex");
  if (sha256 && digest !== sha256.toLowerCase()) {
    await fsp.rm(part, { force: true });
    throw new Error(`The download didn't match its checksum (${url})`);
  }
  await fsp.rename(part, dest);
  return dest;
}

// ── Running uv and Python ──────────────────────────────────────

function uvEnv() {
  const env = cleanEnv();
  Object.assign(env, {
    UV_CACHE_DIR: path.join(runtimeDir(), "cache"),
    UV_PYTHON_INSTALL_DIR: path.join(runtimeDir(), "python"),
    UV_PYTHON_PREFERENCE: "only-managed",
    UV_NO_CONFIG: "1",
    UV_COMPILE_BYTECODE: "1",
    UV_NO_PROGRESS: "1",
  });
  return env;
}

// Runs a command; *onLine* gets each line it prints. Rejects with its last lines.
function run(cmd, args, { env, cwd, onLine, timeoutMs = 30 * 60_000 } = {}) {
  return new Promise((resolve, reject) => {
    log.info(`$ ${cmd} ${args.join(" ")}`);
    const child = spawn(cmd, args, { env, cwd: cwd || runtimeDir(), stdio: ["ignore", "pipe", "pipe"], windowsHide: true });
    const tail = [];
    const take = (data) => {
      for (const line of data.toString().split(/\r?\n/)) {
        const text = line.trim();
        if (!text) continue;
        log.info(`  ${text}`);
        tail.push(text);
        if (tail.length > 12) tail.shift();
        onLine?.(text);
      }
    };
    child.stdout.on("data", take);
    child.stderr.on("data", take);
    const timer = setTimeout(() => child.kill("SIGKILL"), timeoutMs);
    child.on("error", (err) => {
      clearTimeout(timer);
      reject(err);
    });
    child.on("exit", (code, signal) => {
      clearTimeout(timer);
      if (code === 0) resolve(tail.join("\n"));
      else reject(new Error(`${path.basename(cmd)} ${args[0]} failed (${signal || code}):\n${tail.slice(-6).join("\n")}`));
    });
  });
}

const uv = (args, onLine) => run(uvPath(), args, { env: uvEnv(), onLine });

async function ensureUv(manifest, progress) {
  const want = manifest.uv.version;
  if (readState().uv === want && fs.existsSync(uvPath())) return;
  const asset = manifest.uv.platforms?.[platformKey()];
  if (!asset) throw new Error(`This release has nothing for ${platformKey()}`);
  progress({ step: "uv", text: `Downloading the installer (uv ${want})`, detail: `uv ${want}`, fraction: 0 });
  const archive = await download(asset.url, path.join(downloadsDir(), path.basename(new URL(asset.url).pathname)), {
    sha256: asset.sha256,
    onProgress: (done, total) => progress({ step: "uv", fraction: total ? done / total : null }),
  });
  const unpack = await fsp.mkdtemp(path.join(downloadsDir(), "uv-"));
  try {
    // tar reads both on every platform this app runs on (Windows 10+ has bsdtar).
    await run("tar", ["-xf", archive, "-C", unpack], { env: cleanEnv() });
    const name = IS_WIN ? "uv.exe" : "uv";
    const found = (await fsp.readdir(unpack, { recursive: true })).find((p) => path.basename(p) === name);
    if (!found) throw new Error(`No ${name} in ${path.basename(archive)}`);
    await fsp.mkdir(path.dirname(uvPath()), { recursive: true });
    await fsp.rm(uvPath(), { force: true });
    await fsp.rename(path.join(unpack, found), uvPath());
    if (!IS_WIN) await fsp.chmod(uvPath(), 0o755);
  } finally {
    await fsp.rm(unpack, { recursive: true, force: true });
    await fsp.rm(archive, { force: true });
  }
  writeState({ uv: want });
}

let installing = null;

function installInProgress() {
  return installing !== null;
}

// Installs *manifest*'s version beside whatever runs. Doesn't switch to it.
function install(manifest, progress = () => {}) {
  if (installing) return installing;
  installing = doInstall(manifest, progress).finally(() => {
    installing = null;
  });
  return installing;
}

async function doInstall(manifest, progress) {
  const version = manifest.version;
  if (isReady(version)) return version;
  await fsp.mkdir(runtimeDir(), { recursive: true });
  await ensureUv(manifest, progress);

  progress({ step: "python", text: `Installing Python ${manifest.python}`, detail: `Python ${manifest.python}`, fraction: null });
  await uv(["python", "install", manifest.python, "--no-bin", "--no-registry"], (line) => progress({ detail: line }));

  const dir = envDir(version);
  await fsp.rm(dir, { recursive: true, force: true });
  try {
    const lock = await download(manifest.lock.url, path.join(downloadsDir(), `desktop-lock-${version}.txt`), {
      sha256: manifest.lock.sha256,
    });
    const wheelName = path.basename(new URL(manifest.wheel.url).pathname);
    progress({ step: "wheel", text: `Downloading Captain ${version}`, fraction: 0 });
    const wheel = await download(manifest.wheel.url, path.join(downloadsDir(), wheelName), {
      sha256: manifest.wheel.sha256,
      size: manifest.wheel.size,
      onProgress: (done, total) => progress({ step: "wheel", fraction: total ? done / total : null }),
    });

    await uv(["venv", "--python", manifest.python, "--no-project", dir], (line) => progress({ detail: line }));
    progress({ step: "packages", text: "Installing the packages Captain uses", fraction: null });
    await uv(["pip", "sync", "--python", envPython(version), "--require-hashes", lock], (line) => progress({ detail: line }));
    progress({ step: "app", text: `Installing Captain ${version}`, fraction: null });
    await uv(["pip", "install", "--python", envPython(version), "--no-deps", wheel], (line) => progress({ detail: line }));

    await fsp.writeFile(
      path.join(dir, READY_MARKER),
      JSON.stringify({ version, wheel: manifest.wheel.sha256, lock: manifest.lock.sha256, installed: new Date().toISOString() }, null, 2),
    );
    await fsp.rm(wheel, { force: true });
    await fsp.rm(lock, { force: true });
  } catch (err) {
    await fsp.rm(dir, { recursive: true, force: true });
    throw err;
  }
  log.info(`Installed Captain ${version}`);
  return version;
}

// ── Test start ─────────────────────────────────────────────────

function freePort() {
  return new Promise((resolve, reject) => {
    const probe = netSock.createServer();
    probe.once("error", reject);
    probe.listen(0, "127.0.0.1", () => {
      const { port } = probe.address();
      probe.close(() => resolve(port));
    });
  });
}

function httpStatus(url) {
  return new Promise((resolve) => {
    const req = http.get(url, { timeout: 5000 }, (res) => {
      res.resume();
      resolve(res.statusCode);
    });
    req.on("error", () => resolve(0));
    req.on("timeout", () => {
      req.destroy();
      resolve(0);
    });
  });
}

function endGroup(child) {
  if (!child || child.exitCode !== null) return Promise.resolve();
  return new Promise((resolve) => {
    child.once("exit", () => resolve());
    if (IS_WIN) {
      spawn("taskkill", ["/pid", String(child.pid), "/f", "/t"], { windowsHide: true });
    } else {
      try { process.kill(-child.pid, "SIGTERM"); } catch { /* gone */ }
      setTimeout(() => {
        try { process.kill(-child.pid, "SIGKILL"); } catch { /* gone */ }
      }, 5000);
    }
  });
}

// What an installed version is checked with before anything switches to it: its
// agent code imports, its page has all its files, and its Flight Deck starts on a
// spare port with an empty data folder and none of the user's keys.
const CHECK_CODE = `
import pathlib, captain_claw, captain_claw.web_server
try:
    from captain_claw.flight_deck.deck_update import bundle_problems
except ImportError:
    bundle_problems = None
if bundle_problems:
    missing = bundle_problems(pathlib.Path(captain_claw.__file__).resolve().parent.parent)
    if missing:
        raise SystemExit("The page is missing %d of its files: %s" % (len(missing), ", ".join(missing[:3])))
`;

async function rehearse(version, progress = () => {}, timeoutMs = 180_000) {
  progress({ step: "check", text: `Test-starting Captain ${version}`, fraction: null });
  const top = await fsp.mkdtemp(path.join(os.tmpdir(), "captain-claw-test-start-"));
  const env = {};
  for (const key of ["PATH", "LANG", "LC_ALL", "TMPDIR", "TEMP", "TMP", "USER", "LOGNAME", "USERNAME", "SYSTEMROOT", "SSL_CERT_FILE", "SSL_CERT_DIR"]) {
    if (process.env[key]) env[key] = process.env[key];
  }
  Object.assign(env, {
    HOME: path.join(top, "home"),
    USERPROFILE: path.join(top, "home"),
    CAPTAIN_CLAW_HOME: path.join(top, "home", ".captain-claw"),
    FD_DATA_DIR: path.join(top, "data"),
    FD_JWT_SECRET: crypto.randomBytes(32).toString("hex"),
    FD_AUTH_ENABLED: "false",
    FD_DESKTOP: "1",
    PYTHONNOUSERSITE: "1",
    PYTHONDONTWRITEBYTECODE: "1",
  });
  let child = null;
  const said = [];
  try {
    for (const name of ["home", "data", "run"]) await fsp.mkdir(path.join(top, name), { recursive: true });
    await run(envPython(version), ["-c", CHECK_CODE], { env, cwd: path.join(top, "run"), timeoutMs });

    const port = await freePort();
    child = spawn(envPython(version), ["-m", "captain_claw.flight_deck.server", "--host", "127.0.0.1", "--port", String(port)], {
      env,
      cwd: path.join(top, "run"),
      stdio: ["ignore", "pipe", "pipe"],
      detached: !IS_WIN,
      windowsHide: true,
    });
    const keep = (data) => {
      said.push(data.toString());
      if (said.length > 40) said.shift();
    };
    child.stdout.on("data", keep);
    child.stderr.on("data", keep);
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      if (child.exitCode !== null) throw new Error(`Captain ${version} doesn't start:\n${said.join("").slice(-800)}`);
      const health = await httpStatus(`http://127.0.0.1:${port}/fd/health`);
      if (health) {
        const page = await httpStatus(`http://127.0.0.1:${port}/`);
        if (health === 200 && page === 200) return;
        throw new Error(`Captain ${version} started but doesn't answer properly: /fd/health ${health}, the page ${page}`);
      }
      await new Promise((r) => setTimeout(r, 1000));
    }
    throw new Error(`Captain ${version} didn't answer within ${Math.round(timeoutMs / 1000)} seconds:\n${said.join("").slice(-800)}`);
  } finally {
    await endGroup(child);
    await fsp.rm(top, { recursive: true, force: true }).catch(() => {});
  }
}

// ── Housekeeping ───────────────────────────────────────────────

// Keeps the versions that run or could be gone back to; the rest go, with what
// only they used in uv's cache.
async function prune() {
  const state = readState();
  const keep = new Set([state.current, state.previous, state.pending].filter(Boolean));
  let removed = 0;
  try {
    for (const name of await fsp.readdir(path.join(runtimeDir(), "envs"))) {
      if (keep.has(name) || (installing && !isReady(name))) continue;
      await fsp.rm(envDir(name), { recursive: true, force: true });
      removed++;
      log.info(`Removed Captain ${name}`);
    }
    if (removed && fs.existsSync(uvPath())) await uv(["cache", "prune"]).catch(() => {});
  } catch (err) {
    if (err.code !== "ENOENT") log.warn(`Cleaning up old versions: ${err.message}`);
  }
}

function removeVersion(version) {
  const state = readState();
  if ([state.current, state.previous].includes(version)) return Promise.resolve();
  return fsp.rm(envDir(version), { recursive: true, force: true });
}

// Chromium for the browser tools, in the background once per version: the agents
// would install it on first use anyway, but that first use then waits for it.
let chromium = null;
function ensureChromium() {
  const version = currentVersion();
  if (!version || chromium || readState().chromium === version) return chromium;
  const { cmd, env } = pythonFor(version);
  chromium = run(cmd, ["-m", "playwright", "install", "chromium"], { env, timeoutMs: 20 * 60_000 })
    .then(() => writeState({ chromium: version }))
    .catch((err) => log.warn(`Chromium install for the browser tools failed: ${err.message}`))
    .finally(() => {
      chromium = null;
    });
  return chromium;
}

module.exports = {
  SHELL_PROTOCOL,
  manifestUrl,
  runtimeDir,
  readState,
  currentVersion,
  takePending,
  switchTo,
  rollBack,
  writeState,
  isReady,
  pythonFor,
  compareVersions,
  fetchManifest,
  install,
  installInProgress,
  rehearse,
  prune,
  removeVersion,
  ensureChromium,
};
