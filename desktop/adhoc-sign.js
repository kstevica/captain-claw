// electron-builder afterPack hook (package.json "build.afterPack").
//
// With no Developer ID certificate on the build machine, electron-builder skips
// signing. The app then carries only the linker's signature on its main
// executable: no sealed resources, Info.plist not bound. A copy downloaded in a
// browser is quarantined, and Gatekeeper calls it "damaged and can't be opened"
// with no way past it.
//
// Signing the whole bundle ad hoc makes the signature valid, so a download is
// merely unverified: macOS offers Open Anyway (System Settings → Privacy &
// Security). A real identity (CSC_LINK / CSC_NAME, or one in the keychain)
// still signs over this, as before.
//
// No hardened runtime here: Electron needs JIT entitlements under it, and an
// ad-hoc signature can't be notarized anyway.

const { execFileSync } = require("child_process");
const path = require("path");

exports.default = async function adhocSign(context) {
  if (context.electronPlatformName !== "darwin") return;
  const app = path.join(context.appOutDir, `${context.packager.appInfo.productFilename}.app`);
  console.log(`  • ad-hoc signing ${path.basename(app)} (no Developer ID: unverified, not damaged)`);
  execFileSync("codesign", ["--force", "--deep", "--sign", "-", app], { stdio: "inherit" });
  execFileSync("codesign", ["--verify", "--deep", "--strict", app], { stdio: "inherit" });
};
