from __future__ import annotations

from functools import lru_cache
import json
from typing import Any

from redis import Redis


@lru_cache(maxsize=8)
def get_redis_client(redis_url: str) -> Redis:
    return Redis.from_url(
        redis_url,
        decode_responses=False,
        socket_connect_timeout=0.2,
        socket_timeout=0.2,
    )


def get_json(redis_url: str, key: str) -> Any | None:
    try:
        raw = get_redis_client(redis_url).get(key)
        if raw is None:
            return None
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def set_json(
    redis_url: str,
    key: str,
    value: Any,
    *,
    ttl_seconds: int,
) -> None:
    try:
        client = get_redis_client(redis_url)
        payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        if ttl_seconds > 0:
            client.setex(key, ttl_seconds, payload.encode("utf-8"))
        else:
            client.set(key, payload.encode("utf-8"))
    except Exception:
        return

