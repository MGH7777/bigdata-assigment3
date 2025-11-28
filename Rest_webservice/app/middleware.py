import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from app.models import RequestLog
from app.db import SessionLocal

class TrackingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response: Response = await call_next(request)
        latency_ms = int((time.perf_counter() - start) * 1000)

        ip = request.client.host if request.client else "?"
        user_agent = request.headers.get("user-agent", "")
        segment = request.headers.get("X-User-Segment", "role:anon|geo:UNK|age:na")

        with SessionLocal() as db:
            db.add(RequestLog(
                method=request.method,
                path=request.url.path,
                user_id=None,
                status_code=response.status_code,
                latency_ms=latency_ms,
                ip=ip,
                user_agent=user_agent,
                segment=segment,
            ))
            db.commit()
        return response 
    
    
