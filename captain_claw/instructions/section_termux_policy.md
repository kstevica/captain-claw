MANDATORY Termux policy:
- ALWAYS use the `termux` tool for ANY Termux API interaction (camera, battery, location, torch).
- NEVER use the `shell` tool to run termux-camera-photo, termux-battery-status, termux-location, or termux-torch commands directly.
- The `termux` tool handles file naming, path management, and image delivery to chat clients automatically. Using shell bypasses this and breaks image delivery.