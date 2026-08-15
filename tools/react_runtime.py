"""Shared runtime guards for ReAct-mode nodes."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence

try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover - keeps lightweight tests importable.
    BaseCallbackHandler = object  # type: ignore[misc,assignment]


DEFAULT_REACT_RECURSION_LIMIT = 50


def _as_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except Exception:
            return 0
    return 0


def _has_token_usage(usage: Dict[str, int] | None) -> bool:
    if not usage:
        return False
    return any(_as_int(usage.get(key)) for key in ("input_tokens", "input_cache_tokens", "output_tokens"))


def _merge_token_usage(base: Dict[str, int] | None, update: Dict[str, int] | None) -> Dict[str, int]:
    merged = _empty_token_usage()
    for key in ("input_tokens", "input_cache_tokens", "output_tokens"):
        merged[key] = _as_int((base or {}).get(key)) + _as_int((update or {}).get(key))
    merged["total_tokens"] = merged["input_tokens"] + merged["output_tokens"]
    return merged


class ReactUsageCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
    """Persist ReAct LLM usage as each model call completes.

    ReAct nodes can terminate through a tool exception, for example when
    ``ask_user`` is unavailable in batch mode. In that case ``agent.invoke`` does
    not return a final message list, so token usage must be captured before the
    exception unwinds the call.
    """

    def __init__(self, state: Dict[str, Any], node_name: str):
        self._state = state
        self._node_name = node_name

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            from graph_workflow import _extract_token_usage, _record_node_token_usage

            usage = _extract_token_usage(response)
            metrics = ensure_react_node_metrics(self._state, self._node_name)
            metrics["n_llm_calls"] = int(metrics.get("n_llm_calls") or 0) + 1
            metrics["react_llm_callback_calls"] = int(metrics.get("react_llm_callback_calls") or 0) + 1
            if _has_token_usage(usage):
                metrics["token_usage"] = _merge_token_usage(metrics.get("token_usage"), usage)
                metrics["react_token_callback_recorded"] = True
                _record_node_token_usage(self._state, self._node_name, usage)
            set_react_node_metrics(self._state, self._node_name, metrics)
        except Exception:
            # Telemetry must not interfere with the ReAct node itself.
            pass


def react_recursion_limit(state: Dict[str, Any] | None) -> int:
    try:
        value = int((state or {}).get("react_recursion_limit") or DEFAULT_REACT_RECURSION_LIMIT)
    except Exception:
        return DEFAULT_REACT_RECURSION_LIMIT
    return max(2, value)


def react_invoke_config(state: Dict[str, Any] | None, node_name: str) -> Dict[str, Any]:
    run_id = str((state or {}).get("run_id") or "run")
    config = {
        "recursion_limit": react_recursion_limit(state),
        "configurable": {"thread_id": f"{run_id}:{node_name}:react"},
    }
    if isinstance(state, dict):
        config["callbacks"] = [ReactUsageCallbackHandler(state, node_name)]
    return config


def react_failure_reason(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "recursion" in text or "recursion limit" in text:
        return "react_recursion_limit_exceeded"
    return "react_agent_exception"


def _empty_token_usage() -> Dict[str, int]:
    return {
        "input_tokens": 0,
        "input_cache_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def _default_react_metrics() -> Dict[str, Any]:
    return {
        "mode": "react",
        "wall_clock_s": 0.0,
        "n_llm_calls": 0,
        "n_tool_calls": 0,
        "tool_call_log": [],
        "token_usage": _empty_token_usage(),
        "n_user_interactions": 0,
        "success": False,
        "trajectory_steps": 0,
        "configuration_set": [],
        "deviations_from_reference": [],
        "error": None,
    }


def ensure_react_node_metrics(state: Dict[str, Any], node_name: str) -> Dict[str, Any]:
    nm = dict(state.get("node_metrics") or {})
    metrics = dict(_default_react_metrics())
    metrics.update(dict(nm.get(node_name) or {}))
    metrics["mode"] = "react"
    metrics["react_recursion_limit"] = react_recursion_limit(state)
    nm[node_name] = metrics
    state["node_metrics"] = nm
    return metrics


def set_react_node_metrics(state: Dict[str, Any], node_name: str, metrics: Dict[str, Any]) -> None:
    nm = dict(state.get("node_metrics") or {})
    nm[node_name] = metrics
    state["node_metrics"] = nm


def bump_react_tool_call(
    state: Dict[str, Any],
    node_name: str,
    tool_name: str,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    metrics = ensure_react_node_metrics(state, node_name)
    metrics["n_tool_calls"] = int(metrics.get("n_tool_calls") or 0) + 1
    log = list(metrics.get("tool_call_log") or [])
    log.append({"tool": tool_name, "payload": payload, "t": time.time()})
    metrics["tool_call_log"] = log
    set_react_node_metrics(state, node_name, metrics)
    try:
        from graph_workflow import _trace_tool_call

        _trace_tool_call(state, node_name, tool_name, payload)
    except Exception:
        pass


def finalize_react_node_metrics(
    state: Dict[str, Any],
    node_name: str,
    messages: Optional[Sequence[Any]],
    *,
    success: bool,
    termination_reason: Optional[str],
) -> Dict[str, Any]:
    metrics = ensure_react_node_metrics(state, node_name)
    if not metrics.get("react_llm_callback_calls"):
        metrics["n_llm_calls"] = int(metrics.get("n_llm_calls") or 0) + sum(
            1 for message in (messages or []) if getattr(message, "type", "") == "ai"
        )
    metrics["trajectory_steps"] = len(messages or [])
    metrics["success"] = bool(success)
    metrics["termination_reason"] = termination_reason
    set_react_node_metrics(state, node_name, metrics)
    return metrics


def mark_react_exception(state: Dict[str, Any], node_name: str, exc: BaseException) -> None:
    metrics = ensure_react_node_metrics(state, node_name)
    metrics["error"] = f"{type(exc).__name__}: {exc}"
    metrics["success"] = False
    metrics["termination_reason"] = react_failure_reason(exc)
    set_react_node_metrics(state, node_name, metrics)
