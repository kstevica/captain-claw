/**
 * Captain Claw – Electron main process.
 *
 * Lifecycle:
 *   1. Spawn the PyInstaller-built backend (captain-claw-web) as a child process.
 *   2. Wait for the backend HTTP server to become ready.
 *   3. Open a BrowserWindow pointing at the backend's URL.
 *   4. On quit, gracefully shut down the backend.
 */

const { app, BrowserWindow, Menu, shell, dialog } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const http = require("http");
const fs = require("fs");
const log = require("electron-log/main");

// ── Constants ──────────────────────────────────────────────────

const BACKEND_PORT = 23080;
const BACKEND_HOST = "127.0.0.1";
const BACKEND_URL = `http://${BACKEND_HOST}:${BACKEND_PORT}`;
const HEALTH_POLL_MS = 300;
const HEALTH_TIMEOUT_MS = 30_000;
const DEV_MODE = process.argv.includes("--dev");

// ── State ──────────────────────────────────────────────────────

let mainWindow = null;
let backendProcess = null;
let backendReady = false;
let actualPort = BACKEND_PORT;

// ── Paths ──────────────────────────────────────────────────────

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
  const binary = getBackendBinary();

  let cmd, args, env;

  env = {
    ...process.env,
    CAPTAIN_CLAW_HOME: getUserDataDir(),
  };

  if (binary) {
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
      "Captain Claw",
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
          title: "Captain Claw",
          message: "The backend process has stopped unexpectedly.",
          buttons: ["Restart", "Quit"],
        })
        .then(({ response }) => {
          if (response === 0) {
            backendReady = false;
            spawnBackend();
            waitForBackend().then(() => mainWindow?.loadURL(getBackendUrl()));
          } else {
            app.quit();
          }
        });
    }
  });
}

function getBackendUrl() {
  return `http://${BACKEND_HOST}:${actualPort}`;
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

function stopBackend() {
  if (!backendProcess) return;

  log.info("Stopping backend...");

  if (process.platform === "win32") {
    // On Windows, spawn taskkill for the process tree
    spawn("taskkill", ["/pid", backendProcess.pid.toString(), "/f", "/t"]);
  } else {
    backendProcess.kill("SIGTERM");

    // Force kill after 5 seconds
    setTimeout(() => {
      if (backendProcess) {
        log.warn("Backend did not exit gracefully, force killing");
        backendProcess.kill("SIGKILL");
      }
    }, 5000);
  }
}

// ── Window management ──────────────────────────────────────────

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 800,
    minHeight: 600,
    title: "Captain Claw",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
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

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

function buildMenu() {
  const isMac = process.platform === "darwin";

  const template = [
    ...(isMac
      ? [
          {
            label: "Captain Claw",
            submenu: [
              { role: "about" },
              { type: "separator" },
              { role: "services" },
              { type: "separator" },
              { role: "hide" },
              { role: "hideOthers" },
              { role: "unhide" },
              { type: "separator" },
              { role: "quit" },
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
  ];

  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

// ── App lifecycle ──────────────────────────────────────────────

app.whenReady().then(async () => {
  log.info("Captain Claw Desktop starting...");
  log.info(`Platform: ${process.platform} ${process.arch}`);
  log.info(`Dev mode: ${DEV_MODE}`);
  log.info(`User data: ${getUserDataDir()}`);

  buildMenu();
  createWindow();

  // Show a loading state
  mainWindow.loadURL(
    `data:text/html;charset=utf-8,${encodeURIComponent(LOADING_HTML)}`
  );

  try {
    spawnBackend();
    await waitForBackend();
    mainWindow.loadURL(getBackendUrl());
  } catch (err) {
    log.error("Failed to start backend:", err);
    dialog.showErrorBox(
      "Captain Claw",
      `Failed to start the backend server:\n\n${err.message}\n\nCheck the logs at: ${log.transports.file.getFile().path}`
    );
    app.quit();
  }
});

app.on("window-all-closed", () => {
  stopBackend();
  app.quit();
});

app.on("before-quit", () => {
  stopBackend();
});

app.on("activate", () => {
  // macOS: re-create window when dock icon is clicked
  if (BrowserWindow.getAllWindows().length === 0 && backendReady) {
    createWindow();
    mainWindow.loadURL(getBackendUrl());
  }
});

// ── Loading screen HTML ────────────────────────────────────────

const LOADING_HTML = `<!DOCTYPE html>
<html>
<head>
<style>
  body {
    margin: 0;
    height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: #1a1a2e;
    color: #e0e0e0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
  }
  h1 { font-size: 2rem; margin-bottom: 0.5rem; color: #00d4ff; }
  p  { font-size: 1rem; opacity: 0.7; }
  .spinner {
    width: 40px; height: 40px;
    border: 3px solid rgba(0,212,255,0.2);
    border-top-color: #00d4ff;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-bottom: 1.5rem;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
  <div class="spinner"></div>
  <h1>Captain Claw</h1>
  <p>Starting backend server...</p>
</body>
</html>`;
