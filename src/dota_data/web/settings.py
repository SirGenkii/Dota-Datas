from __future__ import annotations

import os


def env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


DATABASE_URL = env("DATABASE_URL", "postgresql+psycopg://dota:dota@localhost:5432/dota_data")
REDIS_URL = env("REDIS_URL", "redis://localhost:6379/0")

JWT_SECRET_KEY = env("JWT_SECRET_KEY", "dev-change-me")  # nosec - dev default only
JWT_ALGORITHM = env("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(env("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "1440") or "1440")

