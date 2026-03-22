/**
 * version.js — Auto-inject Captain Claw version badge into any page.
 *
 * Include via <script src="/static/version.js" defer></script>
 * Targets element with id="versionBadge" if present, otherwise
 * appends a small badge to the first <header> element.
 */
(function() {
    fetch('/api/version')
        .then(function(r) { return r.json(); })
        .then(function(v) {
            var badge = document.getElementById('versionBadge');
            if (!badge) {
                // Auto-create badge in first header if none exists.
                var header = document.querySelector('header');
                if (!header) return;
                badge = document.createElement('span');
                badge.id = 'versionBadge';
                badge.style.cssText = 'font-size:10px;color:#666;margin-left:6px;';
                // Insert after first child (usually logo/title).
                if (header.firstElementChild) {
                    header.firstElementChild.appendChild(badge);
                } else {
                    header.appendChild(badge);
                }
            }
            badge.textContent = 'v' + v.version;
            badge.title = 'Build: ' + (v.build_date || '');
        })
        .catch(function() {});
})();
