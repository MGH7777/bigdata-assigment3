# Registration API

FastAPI + SQLAlchemy + JWT + per-route rate limiting + tracking/Prometheus + HATEOAS.  
Includes a tiny HTML client to exercise the API.

## Features (mapped to assignment)

- Tracking  
  Request logging middleware stores metadata for each request and exposes Prometheus metrics at /metrics.

- Segmentation  
  Events are tagged with a segment string like  
  segment = role:...|geo:...|age:...  
  based on the user’s role and other attributes.

- Pattern recognition  
  Endpoint for discovering frequent symptom combinations, e.g.:  
  GET /analytics/frequent-pairs?min_support=K

- Feature extraction  
  Endpoint that aggregates incoming data and returns high-level features:  
  GET /analytics/features (e.g. most common symptom codes, total counts, etc.)

- Security  
  - JWT-based login  
  - Admin-only routes  
  - Per-route and per-IP rate limiting

- REST + HATEOAS  
  JSON responses include _links pointing to related resources.

- Client  
  A minimal HTML client (client/client.html) for registration, login, events and analytics.

--------------------------------------------------------------------

## 1) Prerequisites

- Python 3.12+  
- Windows, macOS or Linux  
- Optional Redis for distributed rate limiting

--------------------------------------------------------------------

## 2) Setup & Run (local)

# create venv
python -m venv .venv

# Windows activate
. .venv/Scripts/activate

# macOS/Linux
# source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# run FastAPI server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

API runs at:
http://127.0.0.1:8000

--------------------------------------------------------------------

## 3) Using the simple web client

This repo includes a tiny HTML client in client/client.html.

It demonstrates:
- Register
- Login (JWT)
- Create events with symptoms
- Call analytics endpoints

### Steps:

1. Start API  
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

2. Serve the client  
   cd client  
   python -m http.server 8001

3. Open browser:  
   http://127.0.0.1:8001/client.html

--------------------------------------------------------------------

## 4) HTTP Methods Implemented

GET:
- /users/me
- /events
- /analytics/frequent-pairs
- /analytics/features
- /metrics

POST:
- /register
- /login
- /events

PUT:
- /users/{user_id}

DELETE:
- /events/{event_id}

--------------------------------------------------------------------

## 5) How this matches the assignment

Tracking:  
- Middleware logs every request  
- Metrics exposed at /metrics  

Segmentation:  
- Users/events segmented as role:user|geo:no|age:20-30  

Pattern Recognition:  
- /analytics/frequent-pairs finds common symptom pairs  

Feature Extraction:  
- /analytics/features aggregates statistics  

Scalability:  
- Stateless FastAPI → horizontal scaling  
- Optional Redis backend  
- Prometheus metrics for distributed monitoring  

REST + HATEOAS:  
- Responses include _links to related API endpoints  

Client:  
- Minimal HTML client demonstrates real API usage  

--------------------------------------------------------------------

## 6) Project Structure

Rest_webservice/
├─ app/
│  ├─ main.py
│  ├─ db.py
│  ├─ models.py
│  ├─ schemas.py
│  ├─ security.py
│  ├─ routers.py
│  ├─ services.py
│  ├─ middleware.py
│  ├─ rate_limit.py
│  └─ tasks.py
├─ client/
│  └─ client.html
├─ requirements.txt
└─ README.md
