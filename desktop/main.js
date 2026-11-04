/**
 * Captain Claw – Electron main process.
 *
 * Lifecycle:
 *   1. Find the backend: built in (resources/backend, the full build), or the
 *      app's own installed Python (runtime.js), installing it on first launch.
 *   2. Start the agent server and the Flight Deck as child processes.
 *   3. Wait for the Flight Deck to answer, then show it in the window.
 *   4. On quit, gracefully shut both down.
 *
 * The slim build ships without a backend: runtime.js installs it from the release
 * manifest and later installs each new version beside the running one, test-starts
 * it, and switches on a restart (going back by itself if it doesn't come up).
 */

const { app, BrowserWindow, Menu, shell, dialog, ipcMain } = require("electron");
const { spawn, execFileSync } = require("child_process");
const path = require("path");
const http = require("http");
const net = require("net");
const fs = require("fs");
const log = require("electron-log/main");

// A second app on the same machine (tests, or a build beside the installed one)
// keeps its own window storage and single-instance lock.
if (process.env.CAPTAIN_CLAW_USER_DATA) {
  app.setPath("userData", process.env.CAPTAIN_CLAW_USER_DATA);
  log.transports.file.resolvePathFn = () => path.join(process.env.CAPTAIN_CLAW_USER_DATA, "logs", "main.log");
}

const runtime = require("./runtime");

// ── Constants ──────────────────────────────────────────────────

const BACKEND_PORT = 23080;
const BACKEND_HOST = "127.0.0.1";
const BACKEND_URL = `http://${BACKEND_HOST}:${BACKEND_PORT}`;
// The app's own Flight Deck port — not the usual 25080: a deck already
// running there (a developer's) would answer the health check and the window
// would show that deck instead. The next free port up to FD_PORT_LAST if busy.
const FD_PORT = 25090;
const FD_PORT_LAST = 25099;
// Earlier builds ran it on 25080: an install upgraded from one keeps that
// port (and its window's origin, with the page storage on it) while it's free.
const LEGACY_FD_PORT = 25080;
// The port is kept from launch to launch (desktop-fd.json in the user data
// dir): the window's origin, and with it everything a deck without sign-in
// keeps in the page's storage, stays the same.

// One app at a time: a second launch focuses this window instead of starting
// a second Flight Deck on the same data dir.
const GOT_LOCK = app.requestSingleInstanceLock();
if (!GOT_LOCK) app.quit();
const FD_HOST = "127.0.0.1";
const HEALTH_POLL_MS = 300;
const HEALTH_TIMEOUT_MS = 30_000;
// The installed Python's first start after an install or update reads every
// module from disk for the first time.
const FD_TIMEOUT_MS = 90_000;
const DEV_MODE = process.argv.includes("--dev");
// `npm run dev -- --runtime`: the unpackaged shell with the installed Python.
const RUNTIME_FLAG = process.argv.includes("--runtime");
const UPDATE_CHECK_MS = 6 * 60 * 60 * 1000;

// ── State ──────────────────────────────────────────────────────

let mainWindow = null;
let backendProcess = null;
let fdProcess = null;
let backendReady = false;
let fdReady = false;
let actualPort = BACKEND_PORT;
let actualFdPort = FD_PORT;
// What the Flight Deck last said on stderr: the reason when it doesn't come up.
let fdTail = [];

// ── Paths ──────────────────────────────────────────────────────

// Where the backend comes from: "bundled" (the full build's resources/backend),
// "runtime" (the app's own installed Python), or "source" (--dev, this checkout).
let backendMode = null;

function getBackendMode() {
  if (backendMode) return backendMode;
  if (RUNTIME_FLAG) backendMode = "runtime";
  else if (getFdBinary()) backendMode = "bundled";
  else backendMode = DEV_MODE ? "source" : "runtime";
  return backendMode;
}

// How to start one Python module of the installed version, or null outside runtime mode.
function runtimeCommand(module, args) {
  if (getBackendMode() !== "runtime") return null;
  const version = runtime.currentVersion();
  if (!version) return null;
  const { cmd, env } = runtime.pythonFor(version);
  return { cmd, args: ["-m", module, ...args], env, version };
}

function getBackendBinary() {
  const exeName =
    process.platform === "win32" ? "captain-claw-web.exe" : "captain-claw-web";

  if (DEV_MODE) {
    // In dev mode, look in the project dist/ folder
    const devPath = path.join(__dirname, "..", "dist", "captain-claw", exeName);
    if (fs.existsSync(devPath)) return devPath;

    // Fallback: try running from source via python
    return null;
  }

  // In production: binary is in resources/backend/
  const prodPath = path.join(process.resourcesPath, "backend", exeName);
  if (fs.existsSync(prodPath)) return prodPath;

  return null;
}

function getFdBinary() {
  const exeName =
    process.platform === "win32" ? "captain-claw-fd.exe" : "captain-claw-fd";

  if (DEV_MODE) {
    const devPath = path.join(__dirname, "..", "dist", "captain-claw", exeName);
    if (fs.existsSync(devPath)) return devPath;
    return null;
  }

  const prodPath = path.join(process.resourcesPath, "backend", exeName);
  if (fs.existsSync(prodPath)) return prodPath;

  return null;
}

function getUserDataDir() {
  // Captain Claw stores data in ~/.captain-claw by default.
  // We can override via CAPTAIN_CLAW_HOME for desktop packaging.
  const home =
    process.env.CAPTAIN_CLAW_HOME ||
    path.join(app.getPath("home"), ".captain-claw");
  if (!fs.existsSync(home)) {
    fs.mkdirSync(home, { recursive: true });
  }
  return home;
}

// ── Backend management ─────────────────────────────────────────

function spawnBackend() {
  const installed = runtimeCommand("captain_claw.web_server", []);
  const binary = getBackendMode() === "bundled" ? getBackendBinary() : null;

  let cmd, args, env;

  env = {
    ...(installed ? installed.env : process.env),
    CAPTAIN_CLAW_HOME: getUserDataDir(),
  };

  if (installed) {
    ({ cmd, args } = installed);
    log.info(`Starting backend: Captain ${installed.version}`);
  } else if (binary) {
    cmd = binary;
    args = [];
    log.info(`Starting backend: ${binary}`);
  } else if (DEV_MODE) {
    // Dev fallback: run from source
    cmd = process.platform === "win32" ? "python" : "python3";
    args = ["-m", "captain_claw.web_server"];
    env.PYTHONPATH = path.join(__dirname, "..");
    log.info("Starting backend from source (dev mode)");
  } else {
    dialog.showErrorBox(
      "Captain",
      "Backend binary not found. The application may not have been packaged correctly."
    );
    app.quit();
    return;
  }

  backendProcess = spawn(cmd, args, {
    env,
    cwd: getUserDataDir(),
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  });

  backendProcess.stdout.on("data", (data) => {
    const text = data.toString().trim();
    if (text) log.info(`[backend] ${text}`);

    // Detect actual port from backend output
    const portMatch = text.match(/running at http:\/\/[^:]+:(\d+)/);
    if (portMatch) {
      actualPort = parseInt(portMatch[1], 10);
      log.info(`Backend port detected: ${actualPort}`);
    }
  });

  backendProcess.stderr.on("data", (data) => {
    const text = data.toString().trim();
    if (text) log.warn(`[backend:err] ${text}`);
  });

  backendProcess.on("exit", (code, signal) => {
    log.info(`Backend exited: code=${code} signal=${signal}`);
    backendProcess = null;

    if (backendReady && mainWindow) {
      // Backend crashed while running – show error
      dialog
        .showMessageBox(mainWindow, {
          type: "error",
          title: "Captain",
          message: "The backend process has stopped unexpectedly.",
          buttons: ["Restart", "Quit"],
        })
        .then(({ response }) => {
          if (response === 0) {
            backendReady = false;
            spawnBackend();
            waitForBackend().then(() => mainWindow?.loadURL(getAppUrl()));
          } else {
            app.quit();
          }
        });
    }
  });
}

function waitForBackend() {
  return new Promise((resolve, reject) => {
    const start = Date.now();

    function poll() {
      const url = `http://${BACKEND_HOST}:${actualPort}/api/commands`;
      http
        .get(url, (res) => {
          if (res.statusCode === 200) {
            backendReady = true;
            log.info("Backend is ready");
            resolve();
          } else {
            retry();
          }
        })
        .on("error", () => retry());
    }

    function retry() {
      if (Date.now() - start > HEALTH_TIMEOUT_MS) {
        reject(new Error("Backend did not start within timeout"));
        return;
      }
      setTimeout(poll, HEALTH_POLL_MS);
    }

    poll();
  });
}

// Stops a child process; resolves once it has exited.
function stopProcess(proc, name) {
  if (!proc || proc.exitCode !== null || proc.signalCode !== null) return Promise.resolve();
  return new Promise((resolve) => {
    proc.once("exit", () => resolve());
    log.info(`Stopping ${name}...`);
    if (process.platform === "win32") {
      // On Windows, spawn taskkill for the process tree
      spawn("taskkill", ["/pid", proc.pid.toString(), "/f", "/t"]);
      return;
    }
    proc.kill("SIGTERM");
    // Force kill after 5 seconds
    setTimeout(() => {
      if (proc.exitCode === null && proc.signalCode === null) {
        log.warn(`${name} did not exit gracefully, force killing`);
        proc.kill("SIGKILL");
      }
    }, 5000);
  });
}

function stopBackend() {
  // Not a crash: no "stopped unexpectedly" dialog for this exit.
  backendReady = false;
  return stopProcess(backendProcess, "backend");
}

// ── Flight Deck management ─────────────────────────────────────

function spawnFlightDeck() {
  const fdArgs = ["--host", FD_HOST, "--port", String(actualFdPort)];
  const installed = runtimeCommand("captain_claw.flight_deck.server", fdArgs);
  const binary = getBackendMode() === "bundled" ? getFdBinary() : null;

  let cmd, args, env;

  env = {
    ...(installed ? installed.env : process.env),
    FD_DATA_DIR: path.join(getUserDataDir(), "fd-data"),
    FD_AUTH_ENABLED: "false",
    // This app's own deck: one local user on loopback, so Google and Microsoft
    // connect there — their sign-in in the system browser (openSignIn).
    FD_DESKTOP: "1",
    // Text as UTF-8 on Windows too (runtime.js pythonFor), for a deck from source.
    PYTHONUTF8: "1",
  };
  fdTail = [];

  if (installed) {
    ({ cmd, args } = installed);
    log.info(`Starting Flight Deck: Captain ${installed.version}`);
  } else if (binary) {
    cmd = binary;
    args = ["--host", FD_HOST, "--port", String(actualFdPort)];
    log.info(`Starting Flight Deck: ${binary}`);
  } else if (DEV_MODE) {
    cmd = process.platform === "win32" ? "python" : "python3";
    args = ["-m", "captain_claw.flight_deck.server", "--host", FD_HOST, "--port", String(actualFdPort)];
    env.PYTHONPATH = path.join(__dirname, "..");
    log.info("Starting Flight Deck from source (dev mode)");
  } else {
    log.error("Flight Deck binary not found");
    return;
  }

  fdProcess = spawn(cmd, args, {
    env,
    cwd: getUserDataDir(),
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  });

  fdProcess.stdout.on("data", (data) => {
    const text = data.toString().trim();
    if (text) log.info(`[flight-deck] ${text}`);

    const portMatch = text.match(/running on http:\/\/[^:]+:(\d+)/i) ||
                      text.match(/Uvicorn running on http:\/\/[^:]+:(\d+)/i);
    if (portMatch) {
      actualFdPort = parseInt(portMatch[1], 10);
      log.info(`Flight Deck port detected: ${actualFdPort}`);
    }
  });

  fdProcess.stderr.on("data", (data) => {
    const text = data.toString().trim();
    if (text) log.warn(`[flight-deck:err] ${text}`);
    if (text) {
      fdTail.push(text);
      if (fdTail.length > 20) fdTail.shift();
    }

    // uvicorn logs to stderr
    const portMatch = text.match(/Uvicorn running on http:\/\/[^:]+:(\d+)/i);
    if (portMatch) {
      actualFdPort = parseInt(portMatch[1], 10);
      log.info(`Flight Deck port detected: ${actualFdPort}`);
    }
  });

  writeFdState({ port: actualFdPort, pid: fdProcess.pid });

  fdProcess.on("exit", (code, signal) => {
    log.info(`Flight Deck exited: code=${code} signal=${signal}`);
    fdProcess = null;
    writeFdState({ port: actualFdPort });
  });
}

// ── The app's own Flight Deck between launches ─────────────────

function fdStateFile() {
  return path.join(getUserDataDir(), "desktop-fd.json");
}

function readFdState() {
  try {
    return JSON.parse(fs.readFileSync(fdStateFile(), "utf8")) || {};
  } catch {
    return {};
  }
}

function writeFdState(state) {
  try {
    fs.mkdirSync(getUserDataDir(), { recursive: true });
    fs.writeFileSync(fdStateFile(), JSON.stringify(state));
  } catch (err) {
    log.warn(`Could not save ${fdStateFile()}: ${err.message}`);
  }
}

function alive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

// Whether a pid is still the Flight Deck an earlier launch started (and not
// some other process that got the number since). Windows can't be told
// apart this way, so there it is left alone.
function isOurFd(pid, port) {
  if (process.platform === "win32" || !port) return false;
  try {
    const cmd = execFileSync("ps", ["-p", String(pid), "-o", "command="], { encoding: "utf8" });
    return /captain-claw-fd|captain_claw\.flight_deck\.server/.test(cmd)
      && new RegExp(`--port\\s+${Number(port)}(\\s|$)`).test(cmd);
  } catch {
    return false;
  }
}

// A crash or force-quit skips before-quit, leaving the earlier Flight Deck
// running on its port: stop it, or this launch would start a second one on
// the same data dir.
async function stopLeftoverFd() {
  const { pid, port } = readFdState();
  if (!pid || !alive(pid) || !isOurFd(pid, port)) return;
  log.warn(`Stopping the Flight Deck left over from an earlier run (pid ${pid})`);
  try {
    process.kill(pid, "SIGTERM");
  } catch {
    return;
  }
  for (let i = 0; i < 50 && alive(pid); i++) await new Promise((r) => setTimeout(r, 100));
  if (alive(pid)) {
    try { process.kill(pid, "SIGKILL"); } catch { /* already gone */ }
  }
}

function portFree(port) {
  return new Promise((resolve) => {
    const probe = net.createServer();
    probe.once("error", () => resolve(false));
    probe.once("listening", () => probe.close(() => resolve(true)));
    probe.listen(port, FD_HOST);
  });
}

async function pickFdPort(preferred) {
  const ports = [];
  if (preferred === LEGACY_FD_PORT || (preferred >= FD_PORT && preferred <= FD_PORT_LAST)) ports.push(preferred);
  for (let port = FD_PORT; port <= FD_PORT_LAST; port++) if (port !== preferred) ports.push(port);
  for (const port of ports) {
    if (await portFree(port)) return port;
    log.warn(`Port ${port} is busy — trying the next one`);
  }
  throw new Error(`No free port for Flight Deck between ${FD_PORT} and ${FD_PORT_LAST}`);
}

function getFdUrl() {
  return `http://${FD_HOST}:${actualFdPort}`;
}

// What the window shows: the deck in desktop mode. `?app=desktop` makes the
// page act as the desktop app even without the preload bridge, and the page
// keeps it for the window's life, so reloads and /console/… addresses stay
// in desktop mode — never the full Flight Deck.
function getAppUrl() {
  return `${getFdUrl()}/?app=desktop`;
}

function waitForFlightDeck() {
  const timeoutMs = getBackendMode() === "runtime" ? FD_TIMEOUT_MS : HEALTH_TIMEOUT_MS;
  const proc = fdProcess;
  return new Promise((resolve, reject) => {
    const start = Date.now();

    function poll() {
      if (!proc || proc.exitCode !== null || proc.signalCode !== null) {
        reject(new Error(`It stopped before it answered:\n${fdTail.join("\n").split("\n").slice(-6).join("\n")}`));
        return;
      }
      const url = `http://${FD_HOST}:${actualFdPort}/api/health`;
      http
        .get(url, (res) => {
          if (res.statusCode === 200 || res.statusCode === 404) {
            // 404 is OK — means server is up, just no /api/health route
            fdReady = true;
            log.info("Flight Deck is ready");
            resolve();
          } else {
            retry();
          }
        })
        .on("error", () => retry());
    }

    function retry() {
      if (Date.now() - start > timeoutMs) {
        reject(new Error(`Flight Deck did not start within ${Math.round(timeoutMs / 1000)} seconds`));
        return;
      }
      setTimeout(poll, HEALTH_POLL_MS);
    }

    poll();
  });
}

function stopFlightDeck() {
  fdReady = false;
  return stopProcess(fdProcess, "Flight Deck");
}

// ── Window management ──────────────────────────────────────────

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 800,
    minHeight: 600,
    title: "Captain",
    // The page's own navy, so nothing flashes white before it paints.
    backgroundColor: "#07071a",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      // The preload is sandboxed and can't read package.json: hand it the version.
      additionalArguments: [`--captain-claw-version=${app.getVersion()}`],
      nodeIntegration: false,
      contextIsolation: true,
    },
    show: false,
  });

  // Show when ready to prevent visual flash
  mainWindow.once("ready-to-show", () => {
    mainWindow.show();
  });

  // Open external links in the default browser
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("http")) shell.openExternal(url);
    return { action: "deny" };
  });

  // The page's errors, and a crash of its renderer, reach the log. A crash
  // otherwise leaves a blank window, with nothing anywhere saying why.
  mainWindow.webContents.on("console-message", (_event, level, message, line, sourceId) => {
    if (level >= 3) log.warn(`[page] ${message} (${sourceId}:${line})`);
  });
  let pageCrashes = [];
  mainWindow.webContents.on("render-process-gone", (_event, details) => {
    log.error(`The window's page crashed: ${details.reason} (exit code ${details.exitCode})`);
    if (details.reason === "clean-exit" || !mainWindow || mainWindow.isDestroyed()) return;
    // Back to the start page, not a reload of the one that crashed; and not
    // forever if every load crashes (the log says why).
    const now = Date.now();
    pageCrashes = pageCrashes.filter((t) => now - t < 60_000).concat(now);
    if (pageCrashes.length <= 3) mainWindow.loadURL(getAppUrl());
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

function buildMenu() {
  const isMac = process.platform === "darwin";
  // Only the slim build installs its own updates; the full build is replaced whole.
  const updateItems =
    getBackendMode() === "runtime"
      ? [
          {
            label: updateBusy ? "Installing Update…" : "Check for Updates…",
            enabled: !updateBusy,
            click: () => checkForUpdates({ interactive: true }),
          },
        ]
      : [];

  const template = [
    ...(isMac
      ? [
          {
            label: "Captain",
            // The app's internal name is captain-claw (package.json "name", which
            // also names its data folder), so the roles get the brand by hand.
            submenu: [
              { role: "about", label: "About Captain" },
              ...updateItems,
              { type: "separator" },
              { role: "services" },
              { type: "separator" },
              { role: "hide", label: "Hide Captain" },
              { role: "hideOthers" },
              { role: "unhide" },
              { type: "separator" },
              { role: "quit", label: "Quit Captain" },
            ],
          },
        ]
      : []),
    {
      label: "Edit",
      submenu: [
        { role: "undo" },
        { role: "redo" },
        { type: "separator" },
        { role: "cut" },
        { role: "copy" },
        { role: "paste" },
        { role: "selectAll" },
      ],
    },
    {
      label: "View",
      submenu: [
        { role: "reload" },
        { role: "forceReload" },
        { role: "toggleDevTools" },
        { type: "separator" },
        { role: "resetZoom" },
        { role: "zoomIn" },
        { role: "zoomOut" },
        { type: "separator" },
        { role: "togglefullscreen" },
      ],
    },
    {
      label: "Window",
      submenu: [
        { role: "minimize" },
        { role: "zoom" },
        ...(isMac
          ? [{ type: "separator" }, { role: "front" }]
          : [{ role: "close" }]),
      ],
    },
    ...(!isMac && updateItems.length ? [{ label: "Help", submenu: updateItems }] : []),
  ];

  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

// ── Sign-in in the system browser ──────────────────────────────

// Google refuses sign-in inside an app's own window, so the page asks the
// shell to open its deck's /fd/google|microsoft/login?ticket=… in the system
// browser; the provider then comes back to the deck's loopback /callback.
// Only that: the page must be this app's deck, and so must the URL.
const SIGN_IN_PATHS = new Set(["/fd/google/login", "/fd/microsoft/login"]);

function isOwnDeck(url) {
  try {
    const u = new URL(url);
    return u.protocol === "http:" && u.hostname === FD_HOST && u.port === String(actualFdPort);
  } catch {
    return false;
  }
}

ipcMain.handle("open-sign-in", async (event, url) => {
  if (!isOwnDeck(event.senderFrame?.url || "") || !isOwnDeck(url) || !SIGN_IN_PATHS.has(new URL(url).pathname)) {
    log.warn("Refused to open a sign-in that isn't this app's own");
    return false;
  }
  await shell.openExternal(url);
  return true;
});

// ── The setup / loading page ───────────────────────────────────

function logPath() {
  return log.transports.file.getFile().path;
}

// The page every start shows first: mode "start" (the centred Bot), "update"
// (the same, while switching versions) or "install" (the first start's steps and
// product tour). *progress* is what setupProgress takes.
function showSetupPage(progress) {
  if (!mainWindow) return Promise.resolve();
  return mainWindow.loadFile(path.join(__dirname, "setup.html")).then(() => setupProgress(progress));
}

// { mode, version, title, text, detail, step, fraction } onto the setup page
// (fraction null = no end known).
function setupProgress(progress) {
  if (!mainWindow || mainWindow.isDestroyed() || !mainWindow.webContents.getURL().endsWith("setup.html")) return;
  mainWindow.webContents
    .executeJavaScript(`window.setupProgress && window.setupProgress(${JSON.stringify(progress)})`)
    .catch(() => {});
}

class QuitRequested extends Error {}

// Whether the window had page storage before this launch showed anything: the
// setup page creates it, so it's read before the window opens (see startServers).
let hadPageStorage = false;

// ── Starting the servers ───────────────────────────────────────

async function startServers() {
  // Backend is optional — start it but don't block on it
  spawnBackend();
  waitForBackend()
    .then(() => log.info("Backend ready (background)"))
    .catch((err) => log.warn("Backend failed to start (non-fatal):", err.message));

  // Flight Deck is the primary UI — must start
  await stopLeftoverFd();
  // No port saved yet but the window had page storage before this launch: an
  // upgrade from a build that ran on 25080 — keep its origin while 25080 is free.
  const saved = readFdState().port;
  const upgraded = saved === undefined && hadPageStorage;
  actualFdPort = await pickFdPort(saved ?? (upgraded ? LEGACY_FD_PORT : undefined));
  spawnFlightDeck();
  await waitForFlightDeck();
}

async function stopServers() {
  await Promise.all([stopFlightDeck(), stopBackend()]);
}

// Starts the current version; when that is a version just switched to and it
// doesn't come up, goes back to the one before it and starts that.
async function startServersOrGoBack(switched) {
  try {
    await startServers();
    return null;
  } catch (err) {
    if (!switched || !switched.from) throw err;
    log.error(`Captain ${switched.to} didn't start: ${err.message}`);
    setupProgress({ text: `Captain ${switched.to} didn't start. Going back to ${switched.from}…` });
    await stopServers();
    const back = runtime.rollBack(switched.to);
    if (!back) throw err;
    await startServers();
    return { failed: switched.to, back, error: err };
  }
}

function tellWentBack(wentBack) {
  if (!wentBack) return;
  dialog.showMessageBox(mainWindow, {
    type: "warning",
    title: "Captain",
    message: `Captain ${wentBack.failed} didn't start, so the app went back to ${wentBack.back}.`,
    detail: `${wentBack.error.message}\n\nThe log: ${logPath()}`,
  });
}

// First launch of the slim build: install the newest release. Needs the internet.
async function firstInstall() {
  await showSetupPage({ mode: "install" });
  for (;;) {
    try {
      setupProgress({ detail: "Looking up the newest version" });
      const manifest = await runtime.fetchManifest();
      setupProgress({ version: manifest.version });
      if (manifest.shellTooOld) {
        throw new Error(`Captain ${manifest.version} needs a newer version of this app. Download it from ${manifest.notes_url || "the Captain releases page"}.`);
      }
      await runtime.install(manifest, setupProgress);
      runtime.switchTo(manifest.version);
      setupProgress({ step: "start", detail: "" });
      return;
    } catch (err) {
      log.error("Setup failed:", err);
      const { response } = await dialog.showMessageBox(mainWindow, {
        type: "error",
        title: "Captain",
        message: "Captain couldn't finish setting up.",
        detail: `${err.message}\n\nThe first start downloads what Captain runs on (about 200 MB), so it needs the internet.\n\nThe log: ${logPath()}`,
        buttons: ["Try Again", "Quit"],
        defaultId: 0,
        cancelId: 1,
      });
      if (response !== 0) throw new QuitRequested();
    }
  }
}

// After the deck is up: what an installed version does in the background.
function afterStart() {
  if (getBackendMode() !== "runtime") return;
  const version = runtime.currentVersion();
  // It runs now, so it's no longer one that failed here.
  if (runtime.readState().failed === version) runtime.writeState({ failed: null });
  if (process.platform === "darwin") {
    app.setAboutPanelOptions({ applicationName: "Captain", applicationVersion: version, version: `app ${app.getVersion()}` });
  }
  runtime.prune();
  runtime.ensureChromium();
  if (!updateTimer) {
    setTimeout(() => checkForUpdates({ interactive: false }), 60_000);
    updateTimer = setInterval(() => checkForUpdates({ interactive: false }), UPDATE_CHECK_MS);
  }
}

// ── Updates (the slim build) ───────────────────────────────────

let updateBusy = false;
let updateTimer = null;
// The version the user said "Later" to: not offered again until the app restarts
// or they check themselves.
let declinedVersion = null;

function info(message, detail) {
  return dialog.showMessageBox(mainWindow, { type: "info", title: "Captain", message, detail });
}

async function checkForUpdates({ interactive }) {
  if (getBackendMode() !== "runtime" || !mainWindow) return;
  if (updateBusy) {
    if (interactive) info("An update is already being installed.", "You'll be asked to restart when it's ready.");
    return;
  }
  let manifest;
  try {
    manifest = await runtime.fetchManifest();
  } catch (err) {
    log.warn(`Update check failed: ${err.message}`);
    if (interactive) dialog.showMessageBox(mainWindow, { type: "error", title: "Captain", message: "Couldn't check for updates.", detail: err.message });
    return;
  }
  const current = runtime.currentVersion();
  if (runtime.compareVersions(manifest.version, current) <= 0) {
    if (interactive) info(`Captain ${current} is the newest version.`);
    return;
  }
  const { pending } = runtime.readState();
  if (pending === manifest.version && runtime.isReady(pending)) {
    if (interactive || declinedVersion !== pending) await offerRestart(pending);
    return;
  }
  // One that didn't start here and was gone back from: only when asked.
  const failedHere = runtime.readState().failed === manifest.version;
  if (!interactive && (declinedVersion === manifest.version || failedHere)) return;
  if (manifest.shellTooOld) {
    const { response } = await dialog.showMessageBox(mainWindow, {
      type: "info",
      title: "Captain",
      message: `Captain ${manifest.version} is out, and it needs a newer version of this app.`,
      detail: "Download the new app and install it over this one. Your agents and settings stay.",
      buttons: ["Download", "Later"],
      defaultId: 0,
      cancelId: 1,
    });
    if (response === 0 && manifest.notes_url) shell.openExternal(manifest.notes_url);
    else declinedVersion = manifest.version;
    return;
  }
  for (;;) {
    const { response } = await dialog.showMessageBox(mainWindow, {
      type: "info",
      title: "Captain",
      message: `Captain ${manifest.version} is available. You have ${current}.`,
      detail: `${manifest.summary ? `${manifest.summary}\n\n` : ""}${failedHere ? "It didn't start here last time, and the app went back. " : ""}It installs in the background while you keep working, and asks before it restarts.`,
      buttons: ["Install", "What's New", "Later"],
      defaultId: 0,
      cancelId: 2,
    });
    if (response === 1) {
      if (manifest.notes_url) shell.openExternal(manifest.notes_url);
      continue;
    }
    if (response === 2) {
      declinedVersion = manifest.version;
      return;
    }
    break;
  }
  await installUpdate(manifest);
}

async function installUpdate(manifest) {
  updateBusy = true;
  buildMenu();
  const progress = (p) => {
    if (!mainWindow || mainWindow.isDestroyed()) return;
    if (typeof p.fraction === "number") mainWindow.setProgressBar(p.fraction);
    else if (p.fraction === null) mainWindow.setProgressBar(2);
    if (p.text) log.info(`Update: ${p.text}`);
  };
  try {
    await runtime.install(manifest, progress);
    await runtime.rehearse(manifest.version, progress);
    runtime.writeState({ pending: manifest.version });
  } catch (err) {
    log.error(`Update to ${manifest.version} failed:`, err);
    await runtime.removeVersion(manifest.version).catch(() => {});
    dialog.showMessageBox(mainWindow, {
      type: "error",
      title: "Captain",
      message: `Captain ${manifest.version} couldn't be installed, so nothing changed.`,
      detail: `${err.message}\n\nThe log: ${logPath()}`,
    });
    return;
  } finally {
    updateBusy = false;
    if (mainWindow && !mainWindow.isDestroyed()) mainWindow.setProgressBar(-1);
    buildMenu();
  }
  await offerRestart(manifest.version);
}

async function offerRestart(version) {
  const { response } = await dialog.showMessageBox(mainWindow, {
    type: "info",
    title: "Captain",
    message: `Captain ${version} is ready.`,
    detail: "Restart to start using it. Your agents stop for a moment and start again. If you'd rather wait, it starts the next time you open the app.",
    buttons: ["Restart Now", "Later"],
    defaultId: 0,
    cancelId: 1,
  });
  if (response !== 0) {
    declinedVersion = version;
    return;
  }
  await restartIntoPending();
}

async function restartIntoPending() {
  const { pending } = runtime.readState();
  await showSetupPage({ mode: "update", text: "Stopping the agents for a moment", version: pending || "" });
  await stopServers();
  const switched = runtime.takePending();
  setupProgress({ text: `Updating to Captain ${runtime.currentVersion()}` });
  try {
    const wentBack = await startServersOrGoBack(switched);
    mainWindow?.loadURL(getAppUrl());
    tellWentBack(wentBack);
    afterStart();
  } catch (err) {
    log.error("Failed to start Flight Deck after the update:", err);
    dialog.showErrorBox("Captain", `Captain couldn't start:\n\n${err.message}\n\nThe log: ${logPath()}`);
    app.quit();
  }
}

// ── App lifecycle ──────────────────────────────────────────────

app.on("second-instance", () => {
  if (!mainWindow) return;
  if (mainWindow.isMinimized()) mainWindow.restore();
  mainWindow.focus();
});

app.whenReady().then(async () => {
  if (!GOT_LOCK) return;
  log.info("Captain Desktop starting...");
  log.info(`Platform: ${process.platform} ${process.arch}`);
  log.info(`Dev mode: ${DEV_MODE}`);
  log.info(`Backend: ${getBackendMode()}`);
  log.info(`User data: ${getUserDataDir()}`);
  if (getBackendMode() === "runtime") log.info(`Runtime: ${runtime.runtimeDir()} (releases: ${runtime.manifestUrl()})`);

  hadPageStorage = fs.existsSync(path.join(app.getPath("userData"), "Local Storage"));
  buildMenu();
  createWindow();

  try {
    let switched = null;
    if (getBackendMode() === "runtime") {
      // A version installed and test-started earlier, waiting for this start.
      switched = runtime.takePending();
      if (!runtime.currentVersion()) await firstInstall();
      else {
        await showSetupPage({
          mode: switched ? "update" : "start",
          text: switched ? `Updating to Captain ${switched.to}` : "Starting…",
          version: runtime.currentVersion(),
        });
      }
    } else {
      await showSetupPage({ mode: "start", text: "Starting…" });
    }
    const wentBack = await startServersOrGoBack(switched);
    mainWindow.loadURL(getAppUrl());
    tellWentBack(wentBack);
    afterStart();
  } catch (err) {
    if (err instanceof QuitRequested) {
      app.quit();
      return;
    }
    log.error("Failed to start Flight Deck:", err);
    dialog.showErrorBox("Captain", `Captain couldn't start:\n\n${err.message}\n\nThe log: ${logPath()}`);
    app.quit();
  }
});

app.on("window-all-closed", () => {
  stopFlightDeck();
  stopBackend();
  app.quit();
});

app.on("before-quit", () => {
  stopFlightDeck();
  stopBackend();
});

app.on("activate", () => {
  // macOS: re-create window when dock icon is clicked
  if (BrowserWindow.getAllWindows().length === 0 && fdReady) {
    createWindow();
    mainWindow.loadURL(getAppUrl());
  }
});
