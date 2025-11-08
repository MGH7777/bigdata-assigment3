# app/rate_limit.py
from __future__ import annotations
import os, time
from typing import Dict, Tuple
from fastapi import HTTPException, Request, Depends

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

try:
    import redis  # type: ignore
    _r = redis.from_url(REDIS_URL, decode_responses=True)
    try:
        _r.ping()
        _redis_ok = True
    except Exception:
        _r = None
        _redis_ok = False
except Exception:
    _r = None
    _redis_ok = False

_MEM: Dict[str, Tuple[int, int]] = {}  # bucket_key -> (count, expires_ts)

class TokenBucket:

    def __init__(self, key: str, per: int = 1, burst: int = 20):
        self.key = key
        self.per = max(int(per), 1)
        self.burst = max(int(burst), 0)

    def allow(self) -> bool:
        now = int(time.time())
        window = now // self.per
        bucket_key = f"rl:{self.key}:{window}"

        # Redis path
        if _r and _redis_ok:
            current = _r.get(bucket_key)
            if current is None:
                _r.setex(bucket_key, self.per * 2, 1)  # first hit
                return True
            if int(current) >= self.burst:
                return False
            _r.incr(bucket_key)
            return True

        count, expires = _MEM.get(bucket_key, (0, now + self.per))
        if now > expires:
            count, expires = 0, now + self.per
        if count >= self.burst:
            return False
        _MEM[bucket_key] = (count + 1, expires)
        return True


def rate_limiter(resource_key: str, per: int = 1, burst: int = 20):
    def _dep(request: Request):
        ip = request.client.host if request.client else "?"
        bucket = TokenBucket(f"{resource_key}:{ip}", per=per, burst=burst)
        if not bucket.allow():
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return Depends(_dep)
