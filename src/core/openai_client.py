from __future__ import annotations

import os
from typing import Any, Type


def _langfuse_credentials_present() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(
        os.getenv("LANGFUSE_SECRET_KEY")
    )


def _resolve_openai_class() -> Type[Any]:
    if _langfuse_credentials_present():
        try:
            from langfuse.openai import OpenAI as LangfuseOpenAI  # type: ignore

            return LangfuseOpenAI
        except Exception:
            # Fall back to native client if Langfuse package is unavailable/misconfigured.
            pass

    from openai import OpenAI

    return OpenAI


def create_openai_client(*, api_key: str) -> Any:
    openai_cls = _resolve_openai_class()
    return openai_cls(api_key=api_key)


def langfuse_enabled() -> bool:
    return _langfuse_credentials_present()
