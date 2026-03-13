# Summary: browser_credentials.py

# browser_credentials.py Summary

**Summary:**
Stateless credential encryption/decryption service for automated browser login flows with dual-mode security: Fernet symmetric encryption (primary) or Base64 obfuscation (fallback). Integrates with SessionManager for persistent storage in the `browser_credentials` database table and provides high-level helpers for storing, retrieving, listing, and deleting credentials with automatic password masking.

**Purpose:**
Solves the problem of securely managing login credentials for automated browser automation tasks. Provides flexible encryption with graceful degradation when the cryptography library is unavailable, while maintaining a clear audit trail of encryption state and key resolution order. Enables session persistence through cookie storage and retrieval.

**Most Important Functions/Classes/Procedures:**

1. **`CredentialStore.__init__(key: str | None = None)`**
   - Initializes the credential store with key resolution from explicit parameter, config, or environment variable. Sets up Fernet cipher if cryptography is available and key is valid, with comprehensive logging of fallback scenarios. Validates key format and logs warnings for insecure configurations.

2. **`CredentialStore.encrypt(plaintext: str) -> str`**
   - Core encryption method that returns Fernet-encrypted tokens when available, or Base64-obfuscated strings (prefixed with `b64:`) as fallback. Handles UTF-8 encoding and ASCII-safe token representation for database storage.

3. **`CredentialStore.decrypt(ciphertext: str) -> str`**
   - Dual-mode decryption that automatically detects and handles both Fernet tokens and Base64 fallback encoding. Provides detailed error messages when decryption fails (e.g., key mismatch), distinguishing between cryptography unavailability and actual decryption failures.

4. **`CredentialStore.store_credential(app_name, url, username, password, **kwargs) -> dict`**
   - High-level async method that encrypts password, serializes login selectors to JSON, and persists credential to SessionManager's database. Returns masked summary dict for safe logging/display without exposing plaintext password.

5. **`CredentialStore.get_credential(app_name: str) -> dict | None`**
   - Retrieves encrypted credential from database, decrypts password, and reconstructs full credential dict with JSON-deserialized selectors and cookies. Gracefully handles decryption errors by returning error field instead of raising, enabling error reporting to caller.

**Architecture & Dependencies:**
- **Encryption Strategy:** Optional Fernet (cryptography library) with Base64 fallback; key resolution follows 3-tier hierarchy (explicit → config → environment)
- **Database Integration:** Lazy imports SessionManager to avoid circular dependencies; all persistence delegated to `sm.create/get/list/delete_browser_credential()` methods
- **Error Handling:** Comprehensive logging with warnings for missing cryptography or encryption keys; decryption failures include actionable error messages
- **JSON Serialization:** Login selector maps and cookies stored as JSON strings in database, deserialized on retrieval
- **Security Posture:** Passwords masked in list/store responses; Base64 fallback explicitly documented as non-secure obfuscation; supports key rotation (though old tokens become unrecoverable if key changes)