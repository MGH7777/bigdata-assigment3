# Registration API

FastAPI + SQLAlchemy + JWT + per-route Rate Limiting + Tracking/Prometheus + HATEOAS.  
Includes a tiny HTML client to exercise the API.

## Features (mapped to assignment)
- **Tracking**: request logging middleware; Prometheus metrics at `/metrics`.
- **Segmentation**: events are tagged `segment = role:…|geo:…|age:…`.
- **Pattern recognition**: `/analytics/frequent-pairs?min_support=K`.
- **Feature extraction**: `/analytics/features` (most common codes, totals).
- **Security**: JWT login; admin-only routes; rate limiting per route & per IP.
- **REST + HATEOAS**: responses include `_links` for navigation.
- **Client**: `client/client.html` to register/login, create events, run analytics.

---

## 1) Prerequisites
- Python 3.12+
- Windows, macOS or Linux
- Optional: Redis (for distributed rate limiting). If not present, in-process fallback is used.

## 2) Setup & Run (local, no Docker)

```bash
# from repo root
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt

# start API (from repo root)
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
