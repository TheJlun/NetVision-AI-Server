# NetVision AI Server — starter

This is only a deployment/connectivity starter. It does **not** contain a trained model and does **not** detect cheats.

## Endpoints
- `GET /` — info
- `GET /health` — health check
- `POST /analyze` — JSON placeholder response

## Deploy
1. Upload these files to a GitHub repository.
2. On Render, create a new Web Service from the repository.
3. Use the included Dockerfile. If Render offers the Blueprint option, it can read `render.yaml`.
4. After deployment, open `/health` on the service URL; expected response: `{"status":"ok"}`.

## Important
NetVision's actual request format and endpoint must be verified from the plugin before enabling AI. This starter accepts JSON only; it is not yet compatible with NetVision's binary request payload. Do not enable AI in the Minecraft plugin until compatibility has been confirmed.
