// The full build: the PyInstaller backend inside the app, so its first start needs
// no internet. About 1 GB, and every update is a new download of all of it.
// `./build-desktop.sh --full`; the default (package.json "build") is the slim app,
// which installs its backend on first launch (runtime.js).
const base = require("./package.json").build;

module.exports = {
  ...base,
  extraResources: [{ from: "../dist/captain-claw", to: "backend", filter: ["**/*"] }],
  mac: { ...base.mac, target: ["dmg", "zip"] },
  win: { ...base.win, target: ["nsis", "zip"] },
};
