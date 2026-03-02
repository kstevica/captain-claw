/**
 * Captain Claw – Electron preload script.
 *
 * Exposes a minimal bridge to the renderer so the web UI can detect
 * that it is running inside the desktop shell.
 */

const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("captainClawDesktop", {
  platform: process.platform,
  arch: process.arch,
  isDesktop: true,
  version: require("./package.json").version,
});
