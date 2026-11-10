Copies of Captain for Chrome's page scripts (`browser-extension/chrome/`),
shipped with the package so the agent's headless browser runs the same
ones (`tools/browser_page.py`). Change them there and copy them here:
`tests/test_tools/test_browser_page.py` fails while they differ.
