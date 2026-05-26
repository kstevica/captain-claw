# MRBD freshness probe

Tiny web app to verify how Meta Ray-Ban Display Web Apps load: fresh from the
server every launch, or cached on device.

## Run locally

```
python3 server.py            # 0.0.0.0:8765
python3 server.py --port 9000
```

Visit `http://localhost:8765/` in a browser to sanity-check it.

## Expose for the glasses

Meta requires a **publicly accessible HTTPS URL**. Two easy options for local
testing:

- `cloudflared tunnel --url http://localhost:8765`
- `ngrok http 8765`

Take the public `https://…` URL and load it via the Meta AI mobile app
(`Wearables → Web Apps → Add by URL`, or scan a QR pointing at that URL).

## How to read the result

The page server-renders three values that are unique to each request:

- `SSR token` — six random uppercase chars/digits, baked into the HTML.
- `SSR random` — random integer.
- `Req #` — monotonically increasing request counter.

Plus server time, process/system uptime, load avg, and memory.

What different reload behaviors mean:

| Observation on reload                                  | Conclusion                                                |
|--------------------------------------------------------|-----------------------------------------------------------|
| `SSR token` changes every time                          | App is loaded fresh from the server each launch.          |
| `SSR token` stays the same, "Refresh live" updates JSON | HTML was cached on device, but live fetches still work.   |
| Nothing updates and "Refresh live" errors               | Fully offline / cached; no server connectivity at runtime.|

The local JS clock at the bottom keeps ticking from the SSR timestamp — if it
keeps moving past a reload while the SSR token stays frozen, that's a strong
signal the HTML was served from cache.

## Endpoints

- `GET /` — the HTML page (server-rendered, `Cache-Control: no-store`).
- `GET /api/status` — same snapshot as JSON, CORS-open.
- `GET /healthz` — `ok`.
