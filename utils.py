"""utils.py

Shared utility functions for the MCP evaluation harness:
- Environment validation
- XML tag extraction from LLM responses
- JSON serialisation helper
- JSON-path traversal
- Backend error classification
- Basic statistics helpers (mean, stdev)
"""

from __future__ import annotations

import json
import os
import re
import statistics
from typing import Any, Optional

from models import ErrorType


# =========================
# Environment
# =========================


def validate_environment(required_vars: list[str]) -> None:
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


# =========================
# String / JSON helpers
# =========================


def _extract_xml_tag(text: str, tag: str) -> Optional[str]:
    pat = rf"<{tag}>(.*?)</{tag}>"
    m = re.findall(pat, text, re.DOTALL | re.IGNORECASE)
    return m[-1].strip() if m else None


def _safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _get_by_path(obj: Any, path: str) -> Any:
    """Very small JSON-path helper: dot separated, supports integer list indices."""
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
            continue
        if isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError:
                return None
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
            continue
        return None
    return cur


# =========================
# Error classification
# =========================


def _classify_backend_error(backend_type: Optional[str], message: str) -> ErrorType:
    t = (backend_type or "").lower()
    m = (message or "").lower()

    # ---- Missing dataset / resource not found ----
    if (
        any(x in t for x in ["resourcenotfound", "datasetnotfound", "notfound"])
        or "was not found in the workspace" in m
        or "dataset '" in m and "was not found" in m
    ):
        return ErrorType.MISSING_DATASET

    # ---- Invalid parameter / schema / validation ----
    # ToolError is frequently a wrapper around pydantic/value_error; the signal is in the message.
    if (
        any(x in t for x in ["validation", "valueerror", "invalid", "schema"])
        or any(x in m for x in [
            "validation error",
            "value error",
            "type=value_error",
            "pydantic",
            "not found in data",
            "possible columns",
            "field required",
            "extra inputs are not permitted",
            "input should be",
        ])
        or t in ["toolerror"]  # wrapper: classify via message patterns above
    ):
        # If it got here via ToolError but message contains validation/value_error patterns, treat as INVALID_PARAMETER
        return ErrorType.INVALID_PARAMETER

    # ---- Timeout ----
    if (
        any(x in t for x in ["timeout", "timedout"])
        or any(x in m for x in ["timeout", "timed out", "deadline exceeded"])
    ):
        return ErrorType.TIMEOUT

    # ---- Auth / quota / rate limit ----
    if any(x in m for x in [
        "unauthorized",
        "forbidden",
        "permission denied",
        "not permitted",
        "rate limit",
        "too many requests",
        "quota",
        "insufficient privileges",
    ]):
        return ErrorType.AUTH_OR_QUOTA

    # ---- Network / transport ----
    if (
        any(x in t for x in ["connectionerror", "connecterror", "ssLError".lower()])
        or any(x in m for x in [
            "connection reset",
            "failed to establish a new connection",
            "temporary failure in name resolution",
            "name or service not known",
            "tls",
            "ssl",
            "502",
            "503",
            "504",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
        ])
    ):
        return ErrorType.NETWORK_ERROR

    return ErrorType.UNKNOWN


# =========================
# Statistics helpers
# =========================


def _mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def _stdev(xs: list[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0