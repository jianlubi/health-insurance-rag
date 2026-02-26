from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter


_SETUP_COMPLETE = False
_LIBS_INSTRUMENTED = False
_TRUTHY = {"1", "true", "yes", "on"}
_OTEL_ENABLED_ENV = "OTEL_ENABLED"
_OTEL_SERVICE_NAME_ENV = "OTEL_SERVICE_NAME"
_OTEL_SERVICE_VERSION_ENV = "OTEL_SERVICE_VERSION"
_OTEL_ENVIRONMENT_ENV = "OTEL_ENVIRONMENT"
_OTEL_ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"
_OTEL_INSECURE_ENV = "OTEL_EXPORTER_OTLP_INSECURE"
_OTEL_CONSOLE_EXPORT_ENV = "OTEL_EXPORTER_CONSOLE_ENABLED"
_OTEL_FASTAPI_EXCLUDED_URLS_ENV = "OTEL_FASTAPI_EXCLUDED_URLS"

_DEFAULT_SERVICE_NAME = "health-insurance-rag-api"
_DEFAULT_SERVICE_VERSION = "0.1.0"
_DEFAULT_DEPLOYMENT_ENVIRONMENT = "local"
_DEFAULT_OTLP_ENDPOINT = "http://127.0.0.1:4317"
_DEFAULT_OTLP_INSECURE = "true"
_DEFAULT_FASTAPI_EXCLUDED_URLS = "/health"


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in _TRUTHY


def telemetry_enabled() -> bool:
    return _is_truthy(os.getenv(_OTEL_ENABLED_ENV, "false"))


def setup_telemetry(
    *,
    service_name: str | None = None,
    service_version: str | None = None,
) -> bool:
    global _SETUP_COMPLETE
    if _SETUP_COMPLETE:
        return True
    if not telemetry_enabled():
        return False

    resolved_service_name = (
        service_name
        if service_name is not None
        else os.getenv(_OTEL_SERVICE_NAME_ENV, _DEFAULT_SERVICE_NAME)
    )
    resolved_service_version = (
        service_version
        if service_version is not None
        else os.getenv(_OTEL_SERVICE_VERSION_ENV, _DEFAULT_SERVICE_VERSION)
    )

    current_provider = trace.get_tracer_provider()
    if isinstance(current_provider, TracerProvider):
        _instrument_libraries()
        _SETUP_COMPLETE = True
        return True

    resource = Resource.create(
        {
            "service.name": resolved_service_name,
            "service.version": resolved_service_version,
            "deployment.environment": os.getenv(
                _OTEL_ENVIRONMENT_ENV, _DEFAULT_DEPLOYMENT_ENVIRONMENT
            ),
        }
    )
    provider = TracerProvider(resource=resource)

    endpoint = os.getenv(_OTEL_ENDPOINT_ENV, _DEFAULT_OTLP_ENDPOINT)
    insecure = _is_truthy(os.getenv(_OTEL_INSECURE_ENV, _DEFAULT_OTLP_INSECURE))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=insecure))
    )

    if _is_truthy(os.getenv(_OTEL_CONSOLE_EXPORT_ENV, "false")):
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _instrument_libraries()
    _SETUP_COMPLETE = True
    return True


def _instrument_libraries() -> None:
    global _LIBS_INSTRUMENTED
    if _LIBS_INSTRUMENTED:
        return

    for instrumentor in (
        Psycopg2Instrumentor(),
        RedisInstrumentor(),
        HTTPXClientInstrumentor(),
    ):
        try:
            instrumentor.instrument()
        except Exception:
            # Avoid breaking startup if instrumentation is already active.
            pass
    _LIBS_INSTRUMENTED = True


def instrument_fastapi_app(app: FastAPI) -> None:
    if not telemetry_enabled():
        return
    excluded_urls = os.getenv(
        _OTEL_FASTAPI_EXCLUDED_URLS_ENV, _DEFAULT_FASTAPI_EXCLUDED_URLS
    )
    FastAPIInstrumentor.instrument_app(app, excluded_urls=excluded_urls)


def get_tracer(name: str, version: str | None = None) -> Any:
    return trace.get_tracer(name, version)
