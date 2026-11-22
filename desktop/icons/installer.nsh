; Added to electron-builder's NSIS installer (package.json build.nsis.include;
; it lives in buildResources, which is icons/).
;
; electron-builder unpacks an ARM64 app only on ARM64 Windows. An installer that
; carries nothing else therefore unpacks nothing on an x64 PC and still exits 0,
; leaving shortcuts and an uninstaller but no Captain.exe (0.9.1 shipped one as
; "x64"). Stop at the start instead, with a message and a non-zero exit code;
; /SD keeps silent installs (/S) from waiting on the message.
!macro customInit
  !ifdef APP_ARM64
    !ifndef APP_64
      ${IfNot} ${IsNativeARM64}
        MessageBox MB_OK|MB_ICONSTOP "This installer is for Windows on ARM. On this PC, use Captain-Setup-${VERSION}-x64.exe." /SD IDOK
        SetErrorLevel 2
        Quit
      ${EndIf}
    !endif
  !endif
!macroend
