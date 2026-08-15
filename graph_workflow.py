"""Multi-agent graph for WDN optimisation (control-transfer study).

Pipeline::

    planning_agent -> parameter_agent -> [experiment_agent] -> running_node
                   -> report_agent -> END

Each of parameter_agent / experiment_agent / running_node can run in either
workflow mode (structured LLM call with a fixed Pydantic schema) or ReAct mode
(ReAct agent with a bounded tool set). The selection is read from
``state["node_modes"]``.
"""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Dict, List, Literal, Optional, Sequence, Union
from langchain.chat_models import init_chat_model
from tools.tools import *
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from structured_output import *
import base64
import os
import random
import re
from pydantic import ValidationError

# === Global path config ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NETWORK_DIR = (
    os.path.join(BASE_DIR, "networks")
    if os.path.isdir(os.path.join(BASE_DIR, "networks"))
    else BASE_DIR
)

model = None


def _require_main_model():
    if model is None:
        raise RuntimeError(
            "graph_workflow.model is not configured. Set it in the entry runner, "
            "for example: graph_workflow.model = graph_workflow.init_chat_model(...)."
        )
    return model


multi_modal = init_chat_model(
            "Qwen/Qwen3-VL-235B-A22B-Instruct",
            model_provider="huggingface",
            backend="endpoint",
            temperature=0,
            timeout=300,
        )

MAX_EXPERIMENT_CONFIGS = 5
OBJECTIVE_PUMP_ENERGY = "pump_energy"
OBJECTIVE_MODIFIED_RESILIENCE_INDEX = "modified_resilience_index"
WORKFLOW_OBJECTIVES = (OBJECTIVE_PUMP_ENERGY, OBJECTIVE_MODIFIED_RESILIENCE_INDEX)

# Nodes that dispatch between workflow and ReAct mode. planning_agent and
# report_agent are workflow-only.
TOGGLEABLE_NODES = ("parameter_agent", "experiment_agent", "running_node")

units = {
            "pipe_diameter": "m",
            "pipe_roughness": "mm",
            "pump_speed": "ratio",
            OBJECTIVE_PUMP_ENERGY: "kwh",
        }


def _display_objective_name(name: str) -> str:
    mapping = {
        OBJECTIVE_PUMP_ENERGY: "pump energy",
        OBJECTIVE_MODIFIED_RESILIENCE_INDEX: "modified resilience index",
    }
    return mapping.get(str(name), str(name).replace("_", " "))


def _short_text(value: Any, max_len: int = 32) -> str:
    text = str(value)
    return text if len(text) <= max_len else text[: max_len - 3].rstrip() + "..."


def _pretty_label_number(value: str) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.3g}"


def _label_numbers(text: str) -> List[str]:
    return [_pretty_label_number(value) for value in re.findall(r"\d+(?:\.\d+)?", text)]


def _speed_label_bounds(numbers: List[str]) -> Optional[Tuple[str, str]]:
    if len(numbers) < 2:
        return None
    try:
        lo = float(numbers[0])
        hi = float(numbers[1])
    except Exception:
        return numbers[0], numbers[1]
    if lo > 2 and hi > 2:
        lo /= 10.0
        hi /= 10.0
    return _pretty_label_number(str(lo)), _pretty_label_number(str(hi))


def _as_label_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _label_pressure_value(objective_parameters: Optional[Mapping[str, Any]]) -> Optional[float]:
    obj = objective_parameters or {}
    for key in ("minimum_service_pressure", "required_pressure", "service_pressure_min"):
        value = _as_label_float(obj.get(key))
        if value is not None:
            return value
    value = _as_label_float(obj.get("pressure_min_constraint"))
    return value if value and value > 0 else None


def _label_speed_bounds(
    variable_parameters: Optional[Sequence[Mapping[str, Any]]],
) -> Optional[Tuple[float, float]]:
    for item in variable_parameters or []:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("name") or "").lower() != "pump_speed":
            continue
        bounds = item.get("bounds") or {}
        if not isinstance(bounds, Mapping):
            continue
        lb = _as_label_float(bounds.get("lb"))
        ub = _as_label_float(bounds.get("ub"))
        if lb is not None and ub is not None:
            return lb, ub
    return None


def _bounds_changed(
    bounds: Optional[Tuple[float, float]],
    baseline: Optional[Tuple[float, float]],
) -> bool:
    if bounds is None or baseline is None:
        return False
    return abs(bounds[0] - baseline[0]) > 1e-9 or abs(bounds[1] - baseline[1]) > 1e-9


def _format_label_range(bounds: Tuple[float, float]) -> str:
    return f"{_pretty_label_number(str(bounds[0]))}-{_pretty_label_number(str(bounds[1]))}"


def _pressure_level(value: Optional[float], baseline: Optional[float], normalized_name: str) -> Optional[str]:
    if "very_high" in normalized_name:
        return "Very high"
    if "very_low" in normalized_name:
        return "Very low"
    if "high" in normalized_name:
        return "High"
    if "low" in normalized_name:
        return "Low"
    if re.search(r"intermediate|medium|mid", normalized_name):
        return "Mid"
    if value is not None and baseline is not None:
        if value > baseline:
            return "High"
        if value < baseline:
            return "Low"
    return None


def _compact_run_base(name: str, max_len: int = 34) -> str:
    raw = str(name).split(" - ", 1)[0].strip() or "run"
    normalized = re.sub(r"[^a-z0-9.]+", "_", raw.lower()).strip("_")
    numbers = _label_numbers(normalized)

    if "baseline" in normalized or normalized == "base" or normalized.startswith("base_"):
        return "Baseline"

    if "speed" in normalized:
        bounds = _speed_label_bounds(numbers)
        prefix = "Tight speed" if re.search(r"tight|narrow|bounded|range", normalized) else "Speed"
        return f"{prefix} ({bounds[0]}-{bounds[1]})" if bounds else prefix

    level = None
    if "very_high" in normalized:
        level = "Very high"
    elif "very_low" in normalized:
        level = "Very low"
    elif "high" in normalized:
        level = "High"
    elif "low" in normalized:
        level = "Low"
    elif re.search(r"intermediate|medium|mid", normalized):
        level = "Mid"

    pressure_context = re.search(
        r"pressure|p_req|preq|required|service|min_pressure|minimum_service",
        normalized,
    )
    if pressure_context or (level and numbers):
        prefix = f"{level} pressure" if level else "Pressure"
        return f"{prefix} ({numbers[0]} m)" if numbers else prefix

    if "critical" in normalized:
        return "Critical"
    if level:
        return level

    label = re.sub(r"[\s_-]+", " ", raw).strip()
    label = re.sub(r"\s+", " ", label)
    if len(label) <= max_len:
        return label
    return label[: max_len - 3].rstrip() + "..."


def _run_label(
    name: str,
    suffix: Optional[str] = None,
    *,
    objective_parameters: Optional[Mapping[str, Any]] = None,
    variable_parameters: Optional[Sequence[Mapping[str, Any]]] = None,
    baseline_objective_parameters: Optional[Mapping[str, Any]] = None,
    baseline_variable_parameters: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    normalized = re.sub(r"[^a-z0-9.]+", "_", str(name or "").lower()).strip("_")
    if "baseline" in normalized or normalized == "base" or normalized.startswith("base_"):
        base = "Baseline"
    else:
        pressure = _label_pressure_value(objective_parameters)
        baseline_pressure = _label_pressure_value(baseline_objective_parameters)
        bounds = _label_speed_bounds(variable_parameters)
        baseline_bounds = _label_speed_bounds(baseline_variable_parameters)
        pressure_context = re.search(
            r"pressure|p_req|preq|required|service|min_pressure|minimum_service",
            normalized,
        )
        speed_context = "speed" in normalized or "bounds" in normalized or "range" in normalized
        objectives = list((objective_parameters or {}).get("objectives") or [])

        if speed_context and bounds is not None:
            prefix = "Tight speed" if (
                "tight" in normalized
                or "narrow" in normalized
                or _bounds_changed(bounds, baseline_bounds)
            ) else "Speed"
            base = f"{prefix} ({_format_label_range(bounds)})"
        elif (pressure_context or (pressure is not None and baseline_pressure is not None and pressure != baseline_pressure)) and pressure is not None:
            level = _pressure_level(pressure, baseline_pressure, normalized)
            prefix = f"{level} pressure" if level else "Pressure"
            base = f"{prefix} ({_pretty_label_number(str(pressure))} m)"
        elif len(objectives) > 1 and "modified_resilience_index" in objectives:
            base = "Energy + MRI"
        else:
            base = _compact_run_base(name, 34)
    return f"{base} ({suffix})" if suffix else base


def _empty_token_usage() -> Dict[str, int]:
    return {
        "input_tokens": 0,
        "input_cache_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def _as_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+", value)
        if match:
            return int(match.group(0))
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


def _extract_first_int(payload: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        if key in payload and payload.get(key) is not None:
            return _as_int(payload.get(key))
    return None


def _extract_direct_token_usage(payload: Mapping[str, Any]) -> Dict[str, int]:
    usage = _empty_token_usage()

    input_tokens = _extract_first_int(payload, ("input_tokens", "prompt_tokens"))
    output_tokens = _extract_first_int(payload, ("output_tokens", "completion_tokens"))
    cache_tokens = _extract_first_int(
        payload,
        ("input_cache_tokens", "cached_input_tokens", "prompt_cache_hit_tokens", "cache_read_input_tokens"),
    )

    for detail_key in (
        "input_token_details",
        "input_tokens_details",
        "prompt_token_details",
        "prompt_tokens_details",
    ):
        details = payload.get(detail_key)
        if isinstance(details, Mapping):
            detail_cache = _extract_first_int(
                details,
                ("cache_read", "cached_tokens", "prompt_cache_hit_tokens", "input_cache_tokens"),
            )
            if detail_cache is not None:
                cache_tokens = detail_cache
                break

    if input_tokens is not None:
        usage["input_tokens"] = input_tokens
    if cache_tokens is not None:
        usage["input_cache_tokens"] = cache_tokens
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


def _extract_token_usage(payload: Any) -> Dict[str, int]:
    if payload is None:
        return _empty_token_usage()

    if isinstance(payload, (list, tuple, set)):
        usage = _empty_token_usage()
        for item in payload:
            usage = _merge_token_usage(usage, _extract_token_usage(item))
        return usage

    if isinstance(payload, Mapping):
        direct_usage = _extract_direct_token_usage(payload)
        if _has_token_usage(direct_usage):
            return direct_usage

        for key in (
            "raw",
            "raw_response",
            "llm_output",
            "usage_metadata",
            "usage",
            "token_usage",
            "response_metadata",
            "models_usage",
            "generations",
            "message",
            "messages",
        ):
            nested = payload.get(key)
            nested_usage = _extract_token_usage(nested)
            if _has_token_usage(nested_usage):
                return nested_usage

        return _empty_token_usage()

    direct_fields_usage = _extract_direct_token_usage(
        {
            key: getattr(payload, key)
            for key in (
                "input_tokens",
                "prompt_tokens",
                "output_tokens",
                "completion_tokens",
                "input_cache_tokens",
                "cached_input_tokens",
                "prompt_cache_hit_tokens",
            )
            if hasattr(payload, key)
        }
    )
    if _has_token_usage(direct_fields_usage):
        return direct_fields_usage

    for attr in (
        "raw",
        "raw_response",
        "llm_output",
        "usage_metadata",
        "usage",
        "token_usage",
        "response_metadata",
        "models_usage",
        "generations",
        "message",
        "messages",
    ):
        if hasattr(payload, attr):
            nested_usage = _extract_token_usage(getattr(payload, attr))
            if _has_token_usage(nested_usage):
                return nested_usage

    return _empty_token_usage()


def _record_node_token_usage(state: WorkflowState, node_name: str, usage: Dict[str, int] | None) -> None:
    normalized_usage = _merge_token_usage(None, usage)
    workflow_usage = _merge_token_usage(state.get("workflow_token_usage"), normalized_usage)
    breakdown = dict(state.get("workflow_token_breakdown") or {})
    breakdown[node_name] = _merge_token_usage(breakdown.get(node_name), normalized_usage)
    state["workflow_token_usage"] = workflow_usage
    state["workflow_token_breakdown"] = breakdown


def _trace_enabled(state: Mapping[str, Any] | None) -> bool:
    return bool((state or {}).get("trace_enabled", False))


def _trace_max_chars(state: Mapping[str, Any] | None) -> int:
    try:
        return int((state or {}).get("trace_max_chars") or 5000)
    except Exception:
        return 5000


def _trace_shorten(text: Any, max_chars: int) -> str:
    value = str(text)
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 80].rstrip() + f"\n...[truncated {len(value) - max_chars + 80} chars]"


def _trace_format_content(content: Any, max_chars: int) -> str:
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, Mapping):
                item_type = item.get("type")
                if item_type == "image_url":
                    parts.append("[image_url omitted]")
                elif item_type == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(_trace_shorten(item, max_chars))
            else:
                parts.append(_trace_shorten(item, max_chars))
        return _trace_shorten("\n".join(parts), max_chars)
    return _trace_shorten(content, max_chars)


def _trace_format_messages(messages: Any, max_chars: int) -> str:
    lines: List[str] = []
    for idx, msg in enumerate(messages or [], start=1):
        tool_calls = None
        name = None
        if isinstance(msg, Mapping):
            role = msg.get("role") or msg.get("type") or "message"
            content = msg.get("content", "")
            name = msg.get("name")
            extra = msg.get("additional_kwargs") if isinstance(msg.get("additional_kwargs"), Mapping) else {}
            tool_calls = msg.get("tool_calls") or extra.get("tool_calls")
        else:
            role = getattr(msg, "type", None) or getattr(msg, "role", None) or msg.__class__.__name__
            content = getattr(msg, "content", msg)
            name = getattr(msg, "name", None)
            extra = getattr(msg, "additional_kwargs", {}) or {}
            if isinstance(extra, Mapping):
                tool_calls = getattr(msg, "tool_calls", None) or extra.get("tool_calls")
            else:
                tool_calls = getattr(msg, "tool_calls", None)
        label = f"{role}({name})" if name else str(role)
        rendered = f"[{idx}] {label}: {_trace_format_content(content, max_chars)}"
        if tool_calls:
            rendered += f"\n    tool_calls: {_trace_shorten(tool_calls, max_chars)}"
        lines.append(rendered)
    return _trace_shorten("\n".join(lines), max_chars)


def _trace_format_response(response: Any, max_chars: int) -> str:
    if isinstance(response, Mapping):
        if "structured_response" in response:
            value = response.get("structured_response")
        elif "messages" in response:
            return _trace_format_messages(response.get("messages"), max_chars)
        elif "parsed" in response:
            value = response.get("parsed")
        elif "output" in response:
            value = response.get("output")
        else:
            value = response
    elif hasattr(response, "content"):
        value = getattr(response, "content")
    elif hasattr(response, "model_dump"):
        value = response.model_dump(exclude_none=True)
    else:
        value = response
    return _trace_shorten(value, max_chars)


def _trace_print(state: Mapping[str, Any] | None, text: str) -> None:
    if _trace_enabled(state):
        print(text)


def _trace_llm_io(
    state: Mapping[str, Any] | None,
    node_name: str,
    label: str,
    messages: Any,
    response: Any = None,
) -> None:
    if not _trace_enabled(state) or not (state or {}).get("trace_llm_io", True):
        return
    max_chars = _trace_max_chars(state)
    if response is None:
        print(f"\n[trace][{node_name}][LLM input] {label}")
        print(_trace_format_messages(messages, max_chars))
        return
    print(f"[trace][{node_name}][LLM output] {label}")
    print(_trace_format_response(response, max_chars))


def _trace_tool_call(
    state: Mapping[str, Any] | None,
    node_name: str,
    tool_name: str,
    payload: Any = None,
    result: Any = None,
) -> None:
    if not _trace_enabled(state):
        return
    max_chars = _trace_max_chars(state)
    print(f"\n[trace][{node_name}][tool] {tool_name}")
    if payload is not None:
        print(f"payload: {_trace_shorten(payload, max_chars)}")
    if result is not None:
        print(f"result: {_trace_shorten(result, max_chars)}")


# ----- node-level telemetry for the control-transfer study ---------------
import time as _time
import functools as _functools


def _empty_node_metrics(mode: str = "workflow") -> Dict[str, Any]:
    return {
        "mode": mode,
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


def _get_node_metrics(state: WorkflowState, node_name: str) -> Dict[str, Any]:
    """Return (and lazily create) the per-node metrics dict in state."""
    all_metrics = state.get("node_metrics") or {}
    if node_name not in all_metrics:
        all_metrics[node_name] = _empty_node_metrics()
    state["node_metrics"] = all_metrics
    return all_metrics[node_name]


def _set_node_metrics(state: WorkflowState, node_name: str, metrics: Dict[str, Any]) -> None:
    all_metrics = dict(state.get("node_metrics") or {})
    all_metrics[node_name] = metrics
    state["node_metrics"] = all_metrics


def _add_node_llm_calls(state: WorkflowState, node_name: str, count: int = 1) -> None:
    metrics = _get_node_metrics(state, node_name)
    metrics["n_llm_calls"] = int(metrics.get("n_llm_calls") or 0) + int(count)
    _set_node_metrics(state, node_name, metrics)


def _resolve_node_mode(state: WorkflowState, node_name: str) -> str:
    modes = state.get("node_modes") or {}
    mode = modes.get(node_name, "workflow")
    if node_name not in TOGGLEABLE_NODES and mode != "workflow":
        # Non-toggleable nodes are workflow-only regardless of state["node_modes"].
        mode = "workflow"
    return mode


def record_node_metrics(node_name: str):
    """Decorator wrapping a node body to fill state["node_metrics"][node_name].

    The decorator records wall-clock, success/error, the active mode, and the
    token-usage delta produced by this node (read from
    ``workflow_token_breakdown`` before/after the call). It does NOT count
    LLM/tool calls itself; those are populated by the node body (workflow
    via ``_invoke_structured_model``; ReAct via a callback installed in
    ``parameter_react_tools`` etc.).
    """

    def _decorator(fn):
        @_functools.wraps(fn)
        def _wrapper(state: WorkflowState, *args, **kwargs):
            mode = _resolve_node_mode(state, node_name)
            metrics = _get_node_metrics(state, node_name)
            metrics["mode"] = mode
            pre_breakdown = dict(state.get("workflow_token_breakdown") or {})
            _trace_print(state, f"\n[trace][node:start] {node_name} mode={mode}")
            t0 = _time.perf_counter()
            try:
                result_state = fn(state, *args, **kwargs)
            except Exception as exc:
                # Persist failure into state but re-raise so LangGraph can
                # surface it; tests rely on this propagation.
                elapsed = _time.perf_counter() - t0
                metrics["wall_clock_s"] = float(elapsed)
                metrics["success"] = False
                metrics["error"] = f"{type(exc).__name__}: {exc}"
                _set_node_metrics(state, node_name, metrics)
                _trace_print(
                    state,
                    f"[trace][node:error] {node_name} wall={elapsed:.2f}s error={metrics['error']}",
                )
                raise

            elapsed = _time.perf_counter() - t0
            out_state = result_state if isinstance(result_state, Mapping) else state
            # The node body may have replaced node_metrics wholesale; merge.
            current = (out_state.get("node_metrics") or {}).get(node_name, metrics)
            merged = dict(metrics)
            merged.update(current)
            merged["wall_clock_s"] = float(elapsed)
            merged["mode"] = mode
            explicit_failure = (
                current.get("success") is False
                and bool(current.get("error") or current.get("termination_reason"))
            )
            merged["success"] = not explicit_failure

            # Token-usage delta for this node (post − pre).
            post_breakdown = dict(out_state.get("workflow_token_breakdown") or {})
            pre = pre_breakdown.get(node_name) or _empty_token_usage()
            post = post_breakdown.get(node_name) or _empty_token_usage()
            delta = {k: _as_int(post.get(k)) - _as_int(pre.get(k))
                     for k in ("input_tokens", "input_cache_tokens", "output_tokens", "total_tokens")}
            # If the existing token_usage is empty, fill it from the delta.
            if not _has_token_usage(merged.get("token_usage")):
                merged["token_usage"] = delta
            _trace_print(
                out_state,
                (
                    f"[trace][node:end] {node_name} mode={mode} "
                    f"wall={elapsed:.2f}s success={merged.get('success', True)} "
                    f"llm_calls={merged.get('n_llm_calls', 0)} "
                    f"tool_calls={merged.get('n_tool_calls', 0)} "
                    f"tokens={merged.get('token_usage') or delta}"
                ),
            )

            all_metrics = dict(out_state.get("node_metrics") or {})
            all_metrics[node_name] = merged
            # Persist back into the (possibly new) state dict.
            if isinstance(result_state, Mapping):
                new_state = dict(result_state)
                new_state["node_metrics"] = all_metrics
                return new_state
            state["node_metrics"] = all_metrics
            return state

        return _wrapper

    return _decorator


def _append_configuration_trace(
    state: WorkflowState, node_name: str, mode: str, field: str, value: Any
) -> None:
    """Append a (field, value, source, node, mode, t) record to configuration_trace."""
    trace = list(state.get("configuration_trace") or [])
    trace.append(
        {
            "node": node_name,
            "mode": mode,
            "field": str(field),
            "value": value,
            "t": _time.time(),
        }
    )
    state["configuration_trace"] = trace
    metrics = _get_node_metrics(state, node_name)
    cfg_set = list(metrics.get("configuration_set") or [])
    if field not in cfg_set:
        cfg_set.append(field)
    metrics["configuration_set"] = cfg_set
    _set_node_metrics(state, node_name, metrics)


def _invoke_structured_model(chat_model, schema, messages):
    def _parse_result(result: Any) -> tuple[Any, Dict[str, int]]:
        if isinstance(result, Mapping):
            parsed = result.get("parsed")
            if parsed is None:
                parsed = result.get("output")
            if parsed is None:
                parsed = result
            usage = _extract_token_usage(result.get("raw") or result)
            return parsed, usage
        return result, _extract_token_usage(result)

    # Try the richest signature first so we keep raw responses and token usage
    # whenever the installed LangChain/provider combination supports it.
    attempts = (
        {"include_raw": True, "method": "function_calling"},
        {"include_raw": True},
        {"method": "function_calling"},
        {},
    )
    last_type_error: TypeError | None = None

    for kwargs in attempts:
        try:
            runnable = chat_model.with_structured_output(schema, **kwargs)
            result = runnable.invoke(messages)
            return _parse_result(result)
        except TypeError as exc:
            last_type_error = exc
            continue

    if last_type_error is not None:
        raise last_type_error
    raise RuntimeError("with_structured_output failed without raising a TypeError.")


def _deep_merge_dict(base: dict | None, update: dict | None) -> dict:
    merged = dict(base or {})
    for key, value in (update or {}).items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_numeric_value(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if any(token in text for token in ["non negative", "non-negative", "nonnegative", "no negative"]):
            return 0.0
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if match:
            return float(match.group(0))
    return None


def _canonicalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _contains_explicit_inp_path(text: Any) -> bool:
    return bool(re.search(r"\.inp\b", str(text or ""), flags=re.IGNORECASE))


def _contains_explicit_service_pressure(text: Any) -> bool:
    raw = str(text or "")
    pressure_phrase = (
        r"(required|min(?:imum)?|service)\s+pressure|"
        r"pressure\s+(?:required|min(?:imum)?|service)|"
        r"\bPstar\b"
    )
    return bool(
        re.search(pressure_phrase, raw, flags=re.IGNORECASE)
        and re.search(r"\d+(?:\.\d+)?\s*m?\b", raw, flags=re.IGNORECASE)
    )


def _canonicalize_objective_name(value: str) -> str | None:
    text = _canonicalize_text(value)
    mapping = {
        "pump energy": OBJECTIVE_PUMP_ENERGY,
        "energy": OBJECTIVE_PUMP_ENERGY,
        "modified resilience index": OBJECTIVE_MODIFIED_RESILIENCE_INDEX,
        "modified resilience": OBJECTIVE_MODIFIED_RESILIENCE_INDEX,
        "resilience index": OBJECTIVE_MODIFIED_RESILIENCE_INDEX,
        "mri": OBJECTIVE_MODIFIED_RESILIENCE_INDEX,
        "resilience": OBJECTIVE_MODIFIED_RESILIENCE_INDEX,
    }
    return mapping.get(text)


def _canonicalize_demand_model(value: str) -> str | None:
    text = _canonicalize_text(value)
    mapping = {
        "pdd": "PDD",
        "pressure dependent demand": "PDD",
        "dda": "DDA",
        "dd": "DDA",
        "demand driven analysis": "DDA",
        "demand driven": "DDA",
    }
    return mapping.get(text)


def _canonicalize_algorithm_name(value: str) -> str | None:
    text = _canonicalize_text(value)
    mapping = {
        "ga": "ga",
        "genetic algorithm": "ga",
        "nsga2": "nsga2",
        "nsga ii": "nsga2",
        "nsga 2": "nsga2",
        "smsemoa": "smsemoa",
        "sms emoa": "smsemoa",
        "sms": "smsemoa",
        "moead": "moead",
        "moea d": "moead",
        "moea": "moead",
        "agemoea": "agemoea",
        "age moea": "agemoea",
        "age": "agemoea",
        "agemoea2": "agemoea",
        "rvea": "rvea",
        "r vea": "rvea",
        "reference vector": "rvea",
        "de": "de",
        "differential evolution": "de",
        "cmaes": "cmaes",
        "cma es": "cmaes",
        "pso": "pso",
        "particle swarm": "pso",
        "particle swarm optimisation": "pso",
        "particle swarm optimization": "pso",
    }
    return mapping.get(text)


def _canonicalize_termination_type(value: str) -> str | None:
    text = _canonicalize_text(value)
    mapping = {
        "n gen": "n_gen",
        "ngen": "n_gen",
        "n_gen": "n_gen",
        "generation": "n_gen",
        "generations": "n_gen",
        "time": "time",
    }
    return mapping.get(text)


def _canonicalize_items(value):
    if not isinstance(value, str):
        return value
    text = _canonicalize_text(value)
    mapping = {
        "all pipes": "ALL_PIPES",
        "all pumps": "ALL_PUMPS",
        "all valves": "ALL_VALVES",
        "all tanks": "ALL_TANKS",
        "all junctions": "ALL_JUNCTIONS",
        "all sources": "ALL_SOURCES",
        "all reservoirs": "ALL_RESERVOIRS",
    }
    return mapping.get(text, value)


def _canonicalize_var_name(value: str) -> str | None:
    text = _canonicalize_text(value)
    mapping = {
        "pipe diameter": "pipe_diameter",
        "pipe diameters": "pipe_diameter",
        "pipe roughness": "pipe_roughness",
        "pump speed": "pump_speed",
        "pump speeds": "pump_speed",
        "valve setting": "valve_setting",
        "valve settings": "valve_setting",
    }
    return mapping.get(text)


def _canonicalize_setter(value: str) -> str | None:
    text = _canonicalize_text(value)
    mapping = {
        "pipe diameter masked": "pipe_diameter_masked",
        "pipe_diameter_masked": "pipe_diameter_masked",
        "pipe roughness masked": "pipe_roughness_masked",
        "pipe_roughness_masked": "pipe_roughness_masked",
        "pump speed masked": "pump_speed_masked",
        "pump_speed_masked": "pump_speed_masked",
        "valve setting masked": "valve_setting_masked",
        "valve_setting_masked": "valve_setting_masked",
    }
    return mapping.get(text)


def _normalize_algorithm_kwargs(kwargs: dict | None) -> dict:
    if not isinstance(kwargs, dict):
        return {}
    normalized = {}
    for key, value in kwargs.items():
        text = _canonicalize_text(str(key))
        if text in {"population size", "pop size", "pop_size"}:
            parsed = _parse_numeric_value(value)
            normalized["pop_size"] = int(parsed) if parsed is not None else value
        else:
            normalized[key] = value
    return normalized


def _normalize_partial_objective(obj: dict | None) -> dict | None:
    if not obj:
        return None

    normalized = dict(obj)

    objectives = normalized.get("objectives")
    if isinstance(objectives, list):
        cleaned = []
        for item in objectives:
            if isinstance(item, str):
                mapped = _canonicalize_objective_name(item)
                cleaned.append(mapped or item)
            else:
                cleaned.append(item)
        normalized["objectives"] = cleaned

    demand_model = normalized.get("demand_model")
    if isinstance(demand_model, str):
        normalized["demand_model"] = _canonicalize_demand_model(demand_model) or demand_model

    minimum_service_pressure = normalized.get("minimum_service_pressure")
    parsed_service_pressure = _parse_numeric_value(minimum_service_pressure)
    if parsed_service_pressure is not None:
        normalized["minimum_service_pressure"] = parsed_service_pressure
    normalized["pressure_min_constraint"] = 0.0

    detection_limit = normalized.get("detection_limit")
    parsed_detection = _parse_numeric_value(detection_limit)
    if parsed_detection is not None:
        normalized["detection_limit"] = parsed_detection

    algorithm = normalized.get("algorithm")
    if isinstance(algorithm, dict):
        algo_name = algorithm.get("name")
        if isinstance(algo_name, str):
            algorithm["name"] = _canonicalize_algorithm_name(algo_name) or algo_name
        algorithm["kwargs"] = _normalize_algorithm_kwargs(algorithm.get("kwargs"))
        normalized["algorithm"] = algorithm

    termination = normalized.get("termination")
    if isinstance(termination, dict):
        term_type = termination.get("type")
        if isinstance(term_type, str):
            termination["type"] = _canonicalize_termination_type(term_type) or term_type
        parsed_value = _parse_numeric_value(termination.get("value"))
        if parsed_value is not None:
            termination["value"] = parsed_value
        normalized["termination"] = termination

    return normalized


def _normalize_partial_var(var_spec: dict | None) -> dict | None:
    if not var_spec:
        return None

    normalized = dict(var_spec)

    name = normalized.get("name")
    if isinstance(name, str):
        normalized["name"] = _canonicalize_var_name(name) or name

    items = normalized.get("items")
    normalized["items"] = _canonicalize_items(items)

    setter = normalized.get("setter")
    if isinstance(setter, str):
        normalized["setter"] = _canonicalize_setter(setter) or setter

    bounds = normalized.get("bounds")
    if isinstance(bounds, dict):
        parsed_lb = _parse_numeric_value(bounds.get("lb"))
        parsed_ub = _parse_numeric_value(bounds.get("ub"))
        if parsed_lb is not None:
            bounds["lb"] = parsed_lb
        if parsed_ub is not None:
            bounds["ub"] = parsed_ub
        normalized["bounds"] = bounds

    return normalized


def _initial_minimum_service_pressure_confirmed(state: Mapping[str, Any]) -> bool:
    obj = state.get("objective_parameters") or {}
    objectives = obj.get("objectives") or []
    if not isinstance(objectives, list):
        objectives = []
    canonical_objectives = [
        _canonicalize_objective_name(str(name)) or str(name)
        for name in objectives
    ]
    return (
        OBJECTIVE_MODIFIED_RESILIENCE_INDEX in canonical_objectives
        and "minimum_service_pressure" in obj
        and obj.get("minimum_service_pressure") is not None
    )


def _merge_var_specs(existing: list | None, incoming: list | None) -> list | None:
    if not incoming:
        return existing

    existing_specs = [dict(spec) for spec in (existing or [])]
    merged_specs = []

    for idx, partial in enumerate(incoming):
        if idx < len(existing_specs):
            merged_specs.append(_deep_merge_dict(existing_specs[idx], partial))
        else:
            merged_specs.append(partial)

    if len(existing_specs) > len(incoming):
        merged_specs.extend(existing_specs[len(incoming):])

    return merged_specs


def _reference_parameter_defaults(state: Mapping[str, Any]) -> tuple[dict, list]:
    ref_obj = state.get("reference_objective_parameters") or {}
    ref_vars = state.get("reference_variable_parameters") or []

    normalized_obj = _normalize_partial_objective(dict(ref_obj)) if isinstance(ref_obj, Mapping) else {}
    normalized_vars = [
        normalized
        for normalized in (
            _normalize_partial_var(dict(item))
            for item in ref_vars
            if isinstance(item, Mapping)
        )
        if normalized
    ]
    if normalized_obj and state.get("random_seed") is not None:
        normalized_obj["seed"] = int(state.get("random_seed") or 0)
    return normalized_obj or {}, normalized_vars


def _complete_parameters_from_reference(
    state: Mapping[str, Any],
    objective_data: dict | None,
    variable_data: list | None,
) -> tuple[dict | None, list | None, bool]:
    ref_obj, ref_vars = _reference_parameter_defaults(state)
    if not ref_obj and not ref_vars:
        return objective_data, variable_data, False

    merged_obj = _deep_merge_dict(ref_obj, objective_data or {}) if ref_obj else objective_data
    merged_vars = _merge_var_specs(ref_vars, variable_data or []) if ref_vars else variable_data
    return merged_obj, merged_vars, True


def _validate_complete_parameters(objective_data: dict | None, variable_data: list | None):
    validated_objective = None
    validated_variables = None

    if objective_data:
        try:
            validated_objective = ObjectiveParameters.model_validate(objective_data)
        except ValidationError:
            validated_objective = None

    if variable_data:
        try:
            validated_variables = [VarSpec.model_validate(v) for v in variable_data]
        except ValidationError:
            validated_variables = None

    return validated_objective, validated_variables


_PARAMETER_KEY_ORDER = (
    "objective.inp_path",
    "objective.objectives",
    "objective.demand_model",
    "objective.minimum_service_pressure",
    "variables.var_specs",
    "objective.algorithm.name",
    "objective.algorithm.kwargs",
    "objective.termination",
)


def _parameter_key_rank(key: str) -> int:
    text = str(key)
    for idx, prefix in enumerate(_PARAMETER_KEY_ORDER):
        if text == prefix or text.startswith(prefix + "."):
            return idx
    return len(_PARAMETER_KEY_ORDER)


def _ordered_missing_keys(keys: Sequence[str]) -> List[str]:
    unique: List[str] = []
    for key in keys or []:
        text = str(key)
        if text not in unique:
            unique.append(text)
    return sorted(unique, key=lambda key: (_parameter_key_rank(key), key))


def _runtime_required_missing_keys(
    objective_data: dict | None,
    variable_data: list | None,
    validated_objective: Any,
    validated_variables: Any,
    minimum_service_pressure_confirmed: bool = True,
) -> List[str]:
    missing: List[str] = []
    obj = objective_data or {}
    algo = obj.get("algorithm") if isinstance(obj.get("algorithm"), dict) else {}

    if not obj.get("inp_path"):
        missing.append("objective.inp_path")
    objectives = obj.get("objectives") or []
    if not isinstance(objectives, list) or not objectives:
        missing.append("objective.objectives")
    elif any(str(name) not in WORKFLOW_OBJECTIVES for name in objectives):
        missing.append("objective.objectives")
    if obj.get("demand_model") not in {"PDD", "DDA"}:
        missing.append("objective.demand_model")
    if OBJECTIVE_MODIFIED_RESILIENCE_INDEX in objectives:
        if obj.get("minimum_service_pressure") is None or not minimum_service_pressure_confirmed:
            missing.append("objective.minimum_service_pressure")
    if not _canonicalize_algorithm_name(str(algo.get("name") or "")):
        missing.append("objective.algorithm.name")
    if not isinstance(algo.get("kwargs"), dict) or "pop_size" not in algo.get("kwargs", {}):
        missing.append("objective.algorithm.kwargs")
    termination = obj.get("termination") if isinstance(obj.get("termination"), dict) else {}
    if (
        not termination
        or not _canonicalize_termination_type(str(termination.get("type") or ""))
        or termination.get("value") is None
    ):
        missing.append("objective.termination")
    if not variable_data:
        missing.append("variables.var_specs")

    if variable_data and validated_variables is None and "variables.var_specs" not in missing:
        missing.append("variables.var_specs")

    return missing


def _fallback_questions_for_missing(
    missing_keys: Sequence[str],
    objective_data: dict | None = None,
) -> List[str]:
    questions: List[str] = []
    ordered_keys = _ordered_missing_keys(missing_keys)
    missing_set = set(ordered_keys)
    consumed: set[str] = set()

    for key in ordered_keys:
        if key in consumed:
            continue
        if key == "objective.minimum_service_pressure":
            questions.append(
                "objective.minimum_service_pressure: What required service pressure "
                "in meters should PDD use for the modified resilience index Pstar? "
                "This value is written as the WNTR required_pressure and reused as Pstar. "
                "You can keep the default 20 m or provide another value."
            )
        elif key == "objective.inp_path":
            questions.append(
                "objective.inp_path: What is the path to your .inp network file? "
                "For the bundled C-Town model you can answer ctown.inp; absolute paths "
                "are also accepted."
            )
        elif key == "objective.objectives":
            questions.append(
                "objective.objectives: Which optimisation objectives do you want to include? "
                "Available options are pump_energy (lower total pump energy) and "
                "modified_resilience_index (higher hydraulic resilience, internally "
                "minimised as its negative value). You can choose one or both."
            )
        elif key == "objective.demand_model":
            questions.append(
                "objective.demand_model: Which demand model should be used? PDD means "
                "Pressure Dependent Demand, where delivered demand falls when pressure is "
                "insufficient; DDA means Demand Driven Analysis, where requested demand is "
                "imposed regardless of pressure."
            )
        elif key == "objective.algorithm.name":
            objectives = (objective_data or {}).get("objectives") or []
            if len(objectives) > 1:
                questions.append(
                    "objective.algorithm.name: Which multi-objective optimisation algorithm "
                    "do you prefer? Available options are NSGA2 (searches a diverse Pareto "
                    "front) and SMSEMOA (indicator-based Pareto search)."
                )
            elif len(objectives) == 1:
                questions.append(
                    "objective.algorithm.name: Which single-objective optimisation algorithm "
                    "do you prefer? Available options are GA (robust evolutionary search), "
                    "DE (continuous global search), CMAES (adaptive continuous search), and "
                    "PSO (particle-swarm search)."
                )
            else:
                questions.append(
                    "objective.algorithm.name: Which optimisation algorithm do you prefer? "
                    "For one objective use GA, DE, CMAES, or PSO; for multiple objectives use "
                    "NSGA2 or SMSEMOA."
                )
        elif key == "objective.algorithm.kwargs":
            if "objective.termination" in missing_set:
                questions.append(
                    "objective.algorithm.kwargs and objective.termination: For the chosen "
                    "algorithm, what population size and stopping rule should be used? "
                    "For example, pop_size=50 and termination n_gen=50."
                )
                consumed.add("objective.termination")
            else:
                questions.append(
                    "objective.algorithm.kwargs: What population size should the algorithm "
                    "use? For example, pop_size=50 gives 50 candidate schedules per generation."
                )
        elif key == "objective.termination":
            questions.append(
                "objective.termination: What stopping rule should be used? For example, "
                "n_gen=50 stops after 50 generations."
            )
        elif key == "variables.var_specs":
            questions.append(
                "variables.var_specs: Which decision variables do you want to optimise? "
                "Available options are pump_speed (timeseries speed multiplier for pumps) "
                "and valve_setting (timeseries valve setting if the network supports it). "
                "For pump_speed, specify controlled pumps such as ALL_PUMPS or a list of "
                "pump IDs, and give lower/upper bounds such as 0.8 to 1.2."
            )
        if len(questions) >= 3:
            break
    return questions


def _parameter_helper_message(model_message: str, missing_keys: Sequence[str]) -> str:
    ordered_keys = _ordered_missing_keys(missing_keys)
    if not ordered_keys:
        return model_message or "Parameter configuration is complete."
    preview = ", ".join(ordered_keys[:3])
    return f"Collected the parameters that could be inferred. Next required fields: {preview}."


def _question_keys(question: str) -> List[str]:
    prefix = str(question or "").split(":", 1)[0].strip()
    if not prefix:
        return []
    return [part.strip() for part in re.split(r"\s+and\s+", prefix) if part.strip()]


def _numbers_in_text(value: Any) -> List[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(value or ""))]


def _parse_inp_path_answer(answer: Any) -> Optional[str]:
    text = re.sub(r"^[>\s:]+", "", str(answer or "").strip()).strip("\"'`<> ")
    if not text:
        return None
    match = re.search(r"([^\s,;]+\.inp)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip("\"'`<> ")
    return text if text.lower().endswith(".inp") else None


def _parse_objective_answer(answer: Any) -> List[str]:
    text = _canonicalize_text(str(answer or ""))
    if not text:
        return []
    if any(token in text.split() for token in ("both", "all")):
        return [OBJECTIVE_PUMP_ENERGY, OBJECTIVE_MODIFIED_RESILIENCE_INDEX]

    objectives: List[str] = []
    resilience_requested = (
        "modified resilience index" in text
        or "modified resilience" in text
        or "resilience index" in text
        or "mri" in text
        or "resilience" in text
    )
    if ("pump" in text and "energy" in text) or text == "energy" or "pump energy" in text:
        objectives.append(OBJECTIVE_PUMP_ENERGY)
    if resilience_requested:
        objectives.append(OBJECTIVE_MODIFIED_RESILIENCE_INDEX)

    return list(dict.fromkeys(objectives))


def _parse_bounds_answer(answer: Any) -> Optional[Dict[str, float]]:
    nums = _numbers_in_text(answer)
    if len(nums) < 2:
        return None
    lb, ub = nums[0], nums[1]
    if lb > ub:
        lb, ub = ub, lb
    return {"lb": lb, "ub": ub}


def _parse_var_spec_answer(answer: Any) -> Optional[dict]:
    text = _canonicalize_text(str(answer or ""))
    if not text:
        return None

    if "pump" in text:
        name = "pump_speed"
        setter = "pump_speed_masked"
        items = "ALL_PUMPS" if "all" in text else None
    elif "valve" in text:
        name = "valve_setting"
        setter = "valve_setting_masked"
        items = "ALL_VALVES" if "all" in text else None
    else:
        return None

    quoted_items = re.findall(r"['\"]([^'\"]+)['\"]", str(answer or ""))
    if items is None and quoted_items:
        items = quoted_items
    if items is None:
        item_match = re.search(r"\b(?:pumps?|valves?)\s+([A-Za-z0-9_,\-\s]+)", str(answer or ""), flags=re.IGNORECASE)
        if item_match:
            parsed_items = [
                item.strip()
                for item in re.split(r"[, ]+", item_match.group(1))
                if item.strip() and item.strip().lower() not in {"range", "from", "to", "with", "bounds"}
            ]
            items = parsed_items or None
    if items is None:
        return None

    bounds = _parse_bounds_answer(answer)
    if bounds is None:
        return None

    return {
        "name": name,
        "items": items,
        "setter": setter,
        "setter_kwargs": {"group_name": "all"} if isinstance(items, str) and items.startswith("ALL_") else {},
        "timeseries": True,
        "bounds": bounds,
    }


def _parse_population_size(answer: Any) -> Optional[int]:
    text = str(answer or "")
    match = re.search(r"(\d+).*?(?:pop(?:ulation)?(?:\s|_)*size|population)", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:pop(?:ulation)?(?:\s|_)*size|population).*?(\d+)", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    nums = _numbers_in_text(text)
    return int(nums[0]) if nums else None


def _parse_termination_value(answer: Any) -> Optional[int]:
    text = str(answer or "")
    patterns = (
        r"(?:n_gen|ngen|generation|generations|iteration|iterations|stop after|after).*?(\d+)",
        r"(\d+).*?(?:generation|generations|iteration|iterations)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    nums = _numbers_in_text(text)
    if len(nums) >= 2:
        return int(nums[1])
    return int(nums[0]) if nums else None


def _parse_minimum_service_pressure_answer(answer: Any) -> Optional[float]:
    parsed_pressure = _parse_numeric_value(answer)
    if parsed_pressure is not None:
        return parsed_pressure
    text = _canonicalize_text(str(answer or ""))
    if any(token in text.split() for token in ("default", "keep")):
        return 20.0
    if "twenty" in text:
        return 20.0
    return None


def _answers_confirm_minimum_service_pressure(
    answers: Sequence[tuple[str, str]],
) -> bool:
    for question, answer in answers or []:
        if not str(answer or "").strip():
            continue
        if "objective.minimum_service_pressure" in set(_question_keys(question)):
            return _parse_minimum_service_pressure_answer(answer) is not None
    return False


def _apply_parameter_answers(
    objective_data: dict | None,
    variable_data: list | None,
    answers: Sequence[tuple[str, str]],
) -> tuple[dict | None, list | None]:
    obj = dict(objective_data or {})
    variables = [dict(spec) for spec in (variable_data or [])]

    for question, answer in answers or []:
        if not str(answer or "").strip():
            continue
        keys = set(_question_keys(question))

        if "objective.inp_path" in keys:
            inp_path = _parse_inp_path_answer(answer)
            if inp_path:
                obj["inp_path"] = inp_path

        if "objective.objectives" in keys:
            objectives = _parse_objective_answer(answer)
            if objectives:
                obj["objectives"] = objectives

        if "objective.demand_model" in keys:
            demand_model = _canonicalize_demand_model(str(answer))
            if demand_model:
                obj["demand_model"] = demand_model

        if "objective.minimum_service_pressure" in keys:
            parsed_pressure = _parse_minimum_service_pressure_answer(answer)
            if parsed_pressure is not None:
                obj["minimum_service_pressure"] = parsed_pressure

        if "variables.var_specs" in keys:
            var_spec = _parse_var_spec_answer(answer)
            if var_spec:
                variables = _merge_var_specs(variables, [_normalize_partial_var(var_spec)]) or variables

        if "objective.algorithm.name" in keys:
            algo_name = _canonicalize_algorithm_name(str(answer))
            if algo_name:
                algo = dict(obj.get("algorithm") or {})
                algo["name"] = algo_name
                obj["algorithm"] = algo

        if "objective.algorithm.kwargs" in keys:
            pop_size = _parse_population_size(answer)
            if pop_size is not None:
                algo = dict(obj.get("algorithm") or {})
                kwargs = dict(algo.get("kwargs") or {})
                kwargs["pop_size"] = pop_size
                if algo.get("name") in {"ga", "nsga2"}:
                    kwargs.setdefault("eliminate_duplicates", True)
                algo["kwargs"] = kwargs
                obj["algorithm"] = algo

        if "objective.termination" in keys:
            term_value = _parse_termination_value(answer)
            if term_value is not None:
                obj["termination"] = {"type": "n_gen", "value": term_value}

    normalized_obj = _normalize_partial_objective(obj) if obj else None
    normalized_vars = [
        normalized
        for normalized in (_normalize_partial_var(spec) for spec in variables)
        if normalized
    ]
    return normalized_obj, normalized_vars or None


def _algorithm_compatibility_question(objective_data: dict | None) -> Optional[str]:
    if not objective_data:
        return None
    objectives = objective_data.get("objectives") or []
    algorithm = objective_data.get("algorithm") or {}
    algo_name = algorithm.get("name") if isinstance(algorithm, dict) else None
    if not objectives or not algo_name:
        return None

    canon = _canonicalize_algorithm_name(str(algo_name))
    if canon is None:
        return None
    allowed = ("nsga2", "smsemoa") if len(objectives) > 1 else ("ga", "de", "cmaes", "pso")
    if canon in allowed:
        return None

    options = ", ".join(a.upper() for a in allowed)
    run_type = "multi-objective" if len(objectives) > 1 else "single-objective"
    return (
        f"objective.algorithm.name: The selected algorithm '{canon}' is not supported for this {run_type} "
        f"configuration. Available options are {options}; which one do you prefer?"
    )


def _parse_yes_no_answer(answer: Any) -> bool | None:
    if isinstance(answer, bool):
        return answer
    text = str(answer or "").strip().lower()
    text = re.sub(r"^[>\?\s:]+", "", text).strip()
    text = text.strip("\"'` ")
    if not text:
        return None
    yes_exact = {"y", "yes", "true", "1", "on"}
    no_exact = {"n", "no", "false", "0", "off"}
    yes_tokens = ("yes", "need", "needed", "enable")
    no_tokens = ("not need", "do not need", "don't need", "skip", "disable", "no")
    if text in no_exact or any(re.search(rf"\b{re.escape(token)}\b", text) for token in no_tokens):
        return False
    if text in yes_exact or any(re.search(rf"\b{re.escape(token)}\b", text) for token in yes_tokens):
        return True
    return None


def _ask_yes_no_choice(prompt: str, *, default: bool, allow_human_input: bool) -> bool:
    if not allow_human_input:
        return default

    while True:
        answer = input(prompt)
        parsed = _parse_yes_no_answer(answer)
        if parsed is not None:
            return parsed
        print("Please answer yes or no.")

@record_node_metrics("planning_agent")
def planning_agent(state: WorkflowState) -> WorkflowState:
    user_query = state.get("user_query", "")
    node_usage = _empty_token_usage()
    chat_model = _require_main_model()

    planning = create_agent(
        model=chat_model,
        system_prompt=(
            "You are the Planning Agent for a water distribution network (WDN) "
            "optimisation pipeline.\n"
            "Your job:\n"
            "1. Understand the user's high-level task.\n"
            "2. Propose a 3-7 step high-level plan covering:\n"
            "   - loading/validating the WDN .inp file and key modelling assumptions\n"
            "   - defining optimisation objectives\n"
            "   - defining decision variables\n"
            "   - choosing algorithm & termination\n"
            "   - running optimisation and plotting results\n"
            "   - reporting.\n"
            "Do NOT pick specific numeric values. Return a concise markdown plan."
        ),
    )

    planning_messages = [{"role": "user", "content": user_query}]
    _trace_llm_io(state, "planning_agent", "planning", planning_messages)
    resp = planning.invoke({"messages": planning_messages})
    _trace_llm_io(state, "planning_agent", "planning", planning_messages, resp)
    _add_node_llm_calls(state, "planning_agent")
    node_usage = _merge_token_usage(node_usage, _extract_token_usage(resp))

    messages = resp.get("messages", resp)
    if isinstance(messages, list) and messages:
        base_plan = messages[-1].content
    else:
        base_plan = str(messages)

    decision_prompt = (
        "You are assisting the Planning Agent of a WDN optimisation pipeline.\n\n"
        "The user has asked the following (high-level task):\n"
        f"{user_query}\n\n"
        "The current high-level plan is:\n"
        f"{base_plan}\n\n"
        "Please classify this task into ONE of the following categories:\n"
        "1) OPTIMIZATION:\n"
        "   - Single- or multi-objective optimisation aimed at finding a good solution,\n"
        "   - No explicit requirement to compare multiple algorithms or multiple runs.\n"
        "2) COMPARISON:\n"
        "   - The user wants to compare modelling choices, hyperparameter settings,\n"
        "     or different sets of decision variables (e.g. 'which algorithm is better',\n"
        "     'compare PDD vs DDA', 'test different bounds', 'multi-run experiment').\n"
        "3) HYPOTHESIS:\n"
        "   - The user mainly wants to generate or test hypotheses, e.g. the effect of\n"
        "     certain variables or constraints, which *may* or may not require explicit\n"
        "     experimental design.\n\n"
        "Rules for experiment_required:\n"
        "- If category=COMPARISON, experiment_required MUST be true.\n"
        "- If category=OPTIMIZATION, experiment_required MUST be false.\n"
        "- If category=HYPOTHESIS, you must decide:\n"
        "    * experiment_required = true, if answering the question realistically\n"
        "      requires running and comparing multiple experiment settings;\n"
        "    * experiment_required = false, if a single optimisation run plus analysis\n"
        "      is sufficient.\n\n"
        "Return a PlannerDecision object with:\n"
        "- category: 'OPTIMIZATION' | 'COMPARISON' | 'HYPOTHESIS'\n"
        "- experiment_required: true/false\n"
        "- reason: a short explanation of your choice."
    )

    decision_messages = [
        SystemMessage(content="You are a classifier for WDN optimisation tasks."),
        HumanMessage(content=decision_prompt),
    ]
    _trace_llm_io(state, "planning_agent", "planner decision", decision_messages)
    decision, decision_usage = _invoke_structured_model(
        chat_model,
        PlannerDecision,
        decision_messages,
    )
    _trace_llm_io(state, "planning_agent", "planner decision", decision_messages, decision)
    _add_node_llm_calls(state, "planning_agent")
    node_usage = _merge_token_usage(node_usage, decision_usage)

    final_plan = base_plan
    hypothesis_state: dict[str, str] = {}

    if decision.category == "HYPOTHESIS":
        hypothesis_prompt = (
            "You are helping a water distribution network (WDN) planning agent.\n\n"
            "The user has asked a scientific or mechanism-oriented question.\n"
            "Based on the user request and the current high-level plan, return one explicit, "
            "testable hypothesis.\n\n"
            "Requirements:\n"
            "- hypothesis: one concise falsifiable statement\n"
            "- mechanism_rationale: one concise explanation of why this mechanism may hold\n"
            "- testable_prediction: one concise prediction that should be observable in experiments\n\n"
            f"User request:\n{user_query}\n\n"
            f"Current high-level plan:\n{base_plan}\n"
        )

        hypothesis_messages = [
            SystemMessage(content="You generate explicit scientific hypotheses for WDN tasks."),
            HumanMessage(content=hypothesis_prompt),
        ]
        _trace_llm_io(state, "planning_agent", "hypothesis", hypothesis_messages)
        hypothesis_bundle, hypothesis_usage = _invoke_structured_model(
            chat_model,
            HypothesisBundle,
            hypothesis_messages,
        )
        _trace_llm_io(state, "planning_agent", "hypothesis", hypothesis_messages, hypothesis_bundle)
        _add_node_llm_calls(state, "planning_agent")
        node_usage = _merge_token_usage(node_usage, hypothesis_usage)

        final_plan = (
            "## Hypothesis\n"
            f"- Hypothesis: {hypothesis_bundle.hypothesis}\n"
            f"- Mechanism: {hypothesis_bundle.mechanism_rationale}\n"
            f"- Testable prediction: {hypothesis_bundle.testable_prediction}\n\n"
            "## Plan\n"
            f"{base_plan}"
        )
        hypothesis_state = {
            "hypothesis_text": hypothesis_bundle.hypothesis,
            "mechanism_rationale": hypothesis_bundle.mechanism_rationale,
            "testable_prediction": hypothesis_bundle.testable_prediction,
        }

    print(final_plan)

    print(f"[PlannerDecision] category={decision.category}, "
          f"experiment_required={decision.experiment_required}")
    print(f"[PlannerDecision reason] {decision.reason}")

    next_state = {
        **state,
        "plan": final_plan,
        "task_category": decision.category,          # e.g. OPTIMIZATION / COMPARISON / HYPOTHESIS
        "needs_experiments": decision.experiment_required,
        "planner_reason": decision.reason,
        **hypothesis_state,
    }
    _record_node_token_usage(next_state, "planning_agent", node_usage)
    return next_state



def parameter_agent_workflow(state: WorkflowState) -> WorkflowState:

    current_state = dict(state)
    if "minimum_service_pressure_confirmed" not in current_state:
        current_state["minimum_service_pressure_confirmed"] = (
            _initial_minimum_service_pressure_confirmed(current_state)
        )
    allow_human_input = current_state.get("allow_human_input", True)
    node_usage = _empty_token_usage()
    example_objective = {
        "inp_path": "ctown.inp",
        "objectives": [OBJECTIVE_PUMP_ENERGY, OBJECTIVE_MODIFIED_RESILIENCE_INDEX],
        "use_epanet_toolkit": True,
        "demand_model": "PDD",
        # PDD required_pressure defaults to 20 m and is reused as MRI Pstar.
        "minimum_service_pressure": 20.0,
        "detection_limit": 1.0,
        "algorithm": {
            "name": "nsga2",
            "kwargs": {"pop_size": 20, "eliminate_duplicates": True},
        },
        "termination": {"type": "n_gen", "value": 20},
        "seed": int(current_state.get("random_seed") or 1),
        "verbose": True,
    }

    example_var = [
        {
            "name": "pump_speed",
            "items": "ALL_PUMPS",
            "setter": "pump_speed_masked",
            "setter_kwargs": {"group_name": "all"},
            "timeseries": True,
            "bounds": {"lb": 0.8, "ub": 1.2},
        }
    ]


    chat_model = _require_main_model()
    parameter = create_agent(
        model=chat_model,
        checkpointer=InMemorySaver(),
        response_format=ToolStrategy(schema=ParametersState),
        system_prompt=
            '''You are the Parameter Agent for a WDN optimisation pipeline. You output MUST conform to the ParametersState schema. For any enum / Literal field, you MUST choose one of the allowed values, and MUST NOT invent new strings.
            Conversation-first behaviour is required.
            - Do not open a turn by directly prescribing a configuration.
            - First state what the current selectable options are.
            - For each option, give a brief explanation clause saying what it means.
            - Then ask the user to choose.
            - Iterate in this dialogue style until the template is complete.
            Every time you ask the user to choose an objective, demand model, algorithm, variable setting, or numeric setting, include a brief explanation clause saying what it means or why it matters. Keep each explanation short: one short phrase or one short sentence.
            Demand model explanations must be explicit:
            - PDD means Pressure Dependent Demand: delivered demand falls when pressure is insufficient, so it is more realistic when low pressure matters.
            - DDA means Demand Driven Analysis: requested demand is imposed regardless of pressure, so it is a simpler idealised hydraulic assumption.
            - questions_to_user and suggestions must never be bare labels such as only "PDD" or only "pop_size=50"; write them as option plus short explanation.
            - Keep suggestions empty by default. Only use suggestions when the user explicitly asks for a recommendation or says they are unsure.
            - Do not claim that a parameter has been fixed unless the user explicitly chose it in the dialogue.
            - When only part of the template is known, you must still return the partial fields you have already inferred in objective and variables. Do not wait for the whole object to be complete before filling those fields.
            - Normalise common natural-language phrases into schema values when they are unambiguous, for example:
              * "pump energy" -> "pump_energy"
              * "modified resilience index" / "MRI" / "resilience" -> "modified_resilience_index"
              * "DDA" / "demand driven analysis" -> "DDA"
              * "PDD" / "pressure dependent demand" -> "PDD"
            - minimum_service_pressure defaults to 20 m when the user does not override it. If modified_resilience_index is selected, ask the user whether to keep 20 m or provide another value. Explain it as the PDD required_pressure written to the WNTR network and reused as Pstar in the modified resilience index.
            - When the user is selecting decision variables, do not stop at the variable family. You must also confirm the controlled items and the optimisation range.
            - When bounds are missing, ask for lb and ub explicitly and include one short reference example if helpful, but do not silently assign that example.
            - If the algorithm is already known but pop_size and the stopping rule are still missing, ask about them together in the same round.
            - Do not ask for use_epanet_toolkit, detection_limit, seed, or verbose unless the user explicitly wants to override defaults. They default to True, 1.0, the run seed supplied by the user or runner, and True.
            AVAILABLE ALGORITHM (choose from):
            single objective: GA, DE, CMAES, PSO
            multi objective: NSGA2, SMSEMOA
            The algorithm must match the objective count; do not use a single-objective algorithm for multi-objective runs or a multi-objective algorithm for single-objective runs.
            If you recommend NSGA2, say briefly that it is appropriate for searching a diverse Pareto front.
            If you recommend population size or number of generations, give a short reason such as search diversity, convergence, or compute budget.
            AVAILABLE OBJECTIVES (choose from):
            - pump_energy: total pump energy over the simulation horizon; lower means lower operating energy demand.
            - modified_resilience_index: WNTR modified resilience index computed at junctions using Pstar equal to required_pressure; the workflow minimises its negative value, so better solutions have higher displayed MRI.
            
            Notes:
            - The user may select one or multiple objectives.
            - You must ensure the list is non-empty.
            - Do NOT create new objective names.
            
            AVAILABLE DECISION VARIABLES (choose from):
            Operational variables:
            - pump_speed (timeseries variable)
            - valve_setting (if consistent with the .inp model, timeseries variable)
            
            Notes:
            - For pump_speed or valve_setting, you must also ask which pumps/valves are controlled and what numeric range should be explored.
            - When the variable family is known but the bounds are not, explicitly ask for the optimisation range next.
            - If the user is unsure, a short reference example is allowed, for example:
              * pump_speed: 0.8 to 1.2
            
            - bounds.lb and bounds.ub MUST be numeric and reasonable.
            - setter must be one of the existing tool functions:
              * pump_speed_masked
              * valve_setting_masked
            
            WHAT YOU MUST DO:
            You maintain two objects:
            1) ObjectiveParameters (experiment configuration)
            2) VariableParameters (list of VarSpecs).
            Required information includes at least:
             - objective.inp_path
             - objective.objectives (non-empty list)
             - objective.demand_model
             - objective.minimum_service_pressure defaults to 20 m, and should be confirmed when modified_resilience_index is selected
             - objective.algorithm.name
             - objective.algorithm.kwargs
             - objective.termination
             - at least one variable in variables.var_specs with valid bounds.
            Optional fields have defaults and should not block completion:
             - objective.use_epanet_toolkit=True
             - objective.detection_limit=1.0
             - objective.seed defaults to the run seed supplied by the user or runner
             - objective.verbose=True
            Preferred questioning order:
             - First confirm objective.inp_path and objective.objectives.
             - Then confirm objective.demand_model and objective.minimum_service_pressure if modified_resilience_index is selected; otherwise use the 20 m default.
             - Then confirm variables.var_specs, including variable family, controlled items, and bounds.
             - Then confirm objective.algorithm.name.
             - Then confirm objective.algorithm.kwargs and objective.termination together.
            Additional rules:
             - objective.observed is not required unless the user explicitly provides external observed data.
             - If minimum_service_pressure is missing for modified_resilience_index, ask whether to keep the 20 m default or provide another value.
             - If a variable type is known but its scope or bounds are still unclear, the next question block should ask for those missing variable details.
             - If population size and stopping rule are still missing or unclear, ask them together in the same question block.
             - A good variable question mentions the available variable options, what each means, and then asks both the controlled assets and the range.
            You must:
            1. Merge the current values in the state with any new suggestions.
            2. Decide which keys are still missing/unclear, and list them in missing_keys.
            3. If ANY required information is missing, set status='COLLECTING' and ask concrete questions in questions_to_user (<=3).
               Ask in dialogue form: "Now the available options are ... ; ... means ... ; which one do you prefer?"
            4. Only when ALL required information is present and reasonable, set status='COMPLETE' and leave missing_keys empty.
            5. helper_message should summarise what changed this turn and briefly explain the reasoning behind any recommendation.'''
    )

    while True:

        user_query = current_state.get("user_query", "")
        plan = current_state.get("plan", "")

        current_obj = current_state.get("objective_parameters")
        current_var = current_state.get("variable_parameters")
        human_prompt = (
            f"User high-level request:\n{user_query}\n\n"
            f"Global plan:\n{plan}\n\n"
            f"Current objective_parameters (may be partial):\n{current_obj}\n\n"
            f"Current variable_parameters (may be partial):\n{current_var}\n\n"
            "Example of a typical objective_parameters:\n"
            f"{example_objective}\n\n"
            "Example of typical variable_parameters:\n"
            f"{example_var}\n"
        )

        parameter_messages = [{"role": "user", "content": human_prompt}]
        _trace_llm_io(current_state, "parameter_agent", "parameter extraction", parameter_messages)
        out = parameter.invoke(
            {"messages": parameter_messages},
            config={"configurable": {"thread_id": "parameter-memory"}},
        )
        _trace_llm_io(current_state, "parameter_agent", "parameter extraction", parameter_messages, out)
        _add_node_llm_calls(current_state, "parameter_agent")
        node_usage = _merge_token_usage(node_usage, _extract_token_usage(out))

        structured: ParametersState = out["structured_response"]

        incoming_obj = None
        if structured.objective is not None:
            incoming_obj = _normalize_partial_objective(
                structured.objective.model_dump(exclude_none=True)
            )
            if (
                incoming_obj
                and incoming_obj.get("inp_path")
                and not (current_obj or {}).get("inp_path")
                and not _contains_explicit_inp_path(user_query)
            ):
                incoming_obj.pop("inp_path", None)
        new_obj = _deep_merge_dict(current_obj, incoming_obj) if incoming_obj else current_obj
        if (
            new_obj
            and new_obj.get("minimum_service_pressure") is not None
            and not current_state.get("minimum_service_pressure_confirmed", False)
            and _contains_explicit_service_pressure(user_query)
        ):
            current_state["minimum_service_pressure_confirmed"] = True

        incoming_var = None
        if structured.variables:
            incoming_var = [
                normalized
                for normalized in (
                    _normalize_partial_var(v.model_dump(exclude_none=True))
                    for v in structured.variables
                )
                if normalized
            ]
        new_var = _merge_var_specs(current_var, incoming_var) if incoming_var else current_var

        validated_obj, validated_vars = _validate_complete_parameters(new_obj, new_var)
        compatibility_question = _algorithm_compatibility_question(new_obj)
        runtime_missing = _runtime_required_missing_keys(
            new_obj,
            new_var,
            validated_obj,
            validated_vars,
            minimum_service_pressure_confirmed=bool(
                current_state.get("minimum_service_pressure_confirmed", False)
            ),
        )
        missing_keys = _ordered_missing_keys(runtime_missing)
        questions_to_user = _fallback_questions_for_missing(missing_keys, new_obj)
        if compatibility_question:
            validated_obj = None
            if "objective.algorithm.name" not in missing_keys:
                missing_keys = _ordered_missing_keys([*missing_keys, "objective.algorithm.name"])
            questions_to_user = [
                compatibility_question,
                *[q for q in questions_to_user if q != compatibility_question],
            ][:3]

        status = (
            ParameterStatus.COMPLETE
            if (not missing_keys and validated_obj is not None and validated_vars is not None)
            else ParameterStatus.COLLECTING
        )
        reference_fallback_used = False
        if status != ParameterStatus.COMPLETE and not allow_human_input:
            fallback_obj, fallback_var, reference_fallback_used = _complete_parameters_from_reference(
                current_state,
                new_obj,
                new_var,
            )
            if reference_fallback_used:
                new_obj, new_var = fallback_obj, fallback_var
                if _initial_minimum_service_pressure_confirmed({"objective_parameters": new_obj or {}}):
                    current_state["minimum_service_pressure_confirmed"] = True
                validated_obj, validated_vars = _validate_complete_parameters(new_obj, new_var)
                compatibility_question = _algorithm_compatibility_question(new_obj)
                runtime_missing = _runtime_required_missing_keys(
                    new_obj,
                    new_var,
                    validated_obj,
                    validated_vars,
                    minimum_service_pressure_confirmed=bool(
                        current_state.get("minimum_service_pressure_confirmed", False)
                    ),
                )
                missing_keys = _ordered_missing_keys(runtime_missing)
                questions_to_user = _fallback_questions_for_missing(missing_keys, new_obj)
                if compatibility_question:
                    validated_obj = None
                    if "objective.algorithm.name" not in missing_keys:
                        missing_keys = _ordered_missing_keys([*missing_keys, "objective.algorithm.name"])
                    questions_to_user = [
                        compatibility_question,
                        *[q for q in questions_to_user if q != compatibility_question],
                    ][:3]
                status = (
                    ParameterStatus.COMPLETE
                    if (not missing_keys and validated_obj is not None and validated_vars is not None)
                    else ParameterStatus.COLLECTING
                )
        helper_message = _parameter_helper_message(structured.helper_message, missing_keys)
        if reference_fallback_used:
            helper_message = (
                f"{helper_message}\nNon-interactive run: missing fields were filled from "
                "the reference specification."
            ).strip()

        if validated_obj is not None:
            new_obj = validated_obj.model_dump(exclude_none=True)
        if validated_vars is not None:
            new_var = [v.model_dump(exclude_none=True) for v in validated_vars]


        current_state.update(
            {
                "objective_parameters": new_obj,
                "variable_parameters": new_var,
                "parameter_status": status.value,
                "parameter_questions": questions_to_user,
                "helper_message": helper_message,
                "missing_keys": missing_keys,
                "reference_fallback_used": reference_fallback_used,
                "minimum_service_pressure_confirmed": bool(
                    current_state.get("minimum_service_pressure_confirmed", False)
                ),
            }
        )

        if status == ParameterStatus.COMPLETE:
            print("Parameter configuration COMPLETE.")
            _trace_tool_call(
                current_state,
                "parameter_agent",
                "finalize_configuration",
                result={
                    "objective_parameters": current_state.get("objective_parameters"),
                    "variable_parameters": current_state.get("variable_parameters"),
                    "reference_fallback_used": current_state.get("reference_fallback_used", False),
                },
            )
            if current_state.get("needs_experiments"):
                current_state["use_experiment_agent"] = True
            else:
                current_state["use_experiment_agent"] = False

            print(f"[Execution options] experiment_agent={current_state['use_experiment_agent']}")
            break

        if not allow_human_input:
            _record_node_token_usage(current_state, "parameter_agent", node_usage)
            missing_desc = ", ".join(missing_keys) if missing_keys else "unknown fields"
            raise RuntimeError(
                "parameter_agent requires additional user input, but allow_human_input is False. "
                f"Missing keys: {missing_desc}"
            )

        print("\nHelper message from agent:")
        print(helper_message)


        if missing_keys:
            print("\nThe following keys are still missing or unclear:")
            for k in missing_keys:
                print(f"  - {k}")

        print("\nThe agent needs more information:")
        answers = []
        questions = questions_to_user
        if questions:
            for i, q in enumerate(questions, start=1):
                ans = input(f"{i}. {q}\n> ")
                answers.append((q, ans))
        elif structured.suggestions:
            ans = input("Please provide the missing parameter choice.\n> ")
            answers.append(("fallback_prompt", ans))

        if structured.suggestions:
            print("\nOptional reference from agent:")
            for s in structured.suggestions:
                print(f"  - {s}")

        answered_obj, answered_var = _apply_parameter_answers(
            current_state.get("objective_parameters"),
            current_state.get("variable_parameters"),
            answers,
        )
        if _answers_confirm_minimum_service_pressure(answers):
            if answered_obj is None:
                answered_obj = dict(current_state.get("objective_parameters") or {})
            answered_obj.setdefault("minimum_service_pressure", 20.0)
            current_state["minimum_service_pressure_confirmed"] = True
        if answered_obj is not None:
            current_state["objective_parameters"] = answered_obj
        if answered_var is not None:
            current_state["variable_parameters"] = answered_var

        extra_info_lines = ["\n\n[User responses]"]
        for q, a in answers:
            extra_info_lines.append(f"\nQ: {q}\nA: {a}")
        extra_block = "".join(extra_info_lines)

        current_state["user_query"] = user_query + extra_block

    _record_node_token_usage(current_state, "parameter_agent", node_usage)
    return current_state

def experiment_agent_workflow(state: WorkflowState) -> WorkflowState:
    node_usage = _empty_token_usage()

    user_query = state.get("user_query", "")
    plan = state.get("plan", "")
    hypothesis_text = state.get("hypothesis_text", "")
    mechanism_rationale = state.get("mechanism_rationale", "")
    testable_prediction = state.get("testable_prediction", "")

    baseline_obj = state.get("objective_parameters") or {}
    baseline_var = state.get("variable_parameters") or []


    human_prompt = (
        f"User high-level request / questions:\n{user_query}\n\n"
        f"Global plan from planning_agent (if any):\n{plan}\n\n"
        f"Explicit hypothesis (if available):\n{hypothesis_text or 'N/A'}\n\n"
        f"Mechanism rationale (if available):\n{mechanism_rationale or 'N/A'}\n\n"
        f"Testable prediction (if available):\n{testable_prediction or 'N/A'}\n\n"
        "Baseline objective_parameters (already collected):\n"
        f"{baseline_obj}\n\n"
        "Baseline variable_parameters (already collected):\n"
        f"{baseline_var}\n\n"
        "Please propose the smallest set of experiment candidates needed to test the scientific question above.\n"
        "Return a structured ExperimentsDesign object. For comparison tasks, the experiments field should contain "
        "the non-baseline configuration(s) explicitly requested by the user request or global plan. The workflow "
        "adds the baseline separately, so do not duplicate the baseline as a candidate. Build each candidate by "
        "copying the baseline and changing only the fields that the user request or global plan explicitly varies.\n"
    )

    system_prompt = (
        "You are the Experiment Design Agent for a water distribution network (WDN) "
        "optimisation pipeline.\n\n"
        "You receive a single *baseline* configuration defined by:\n"
        "  - objective_parameters (including inp_path, objectives, algorithm, termination, etc.),\n"
        "  - variable_parameters (list of VarSpecs defining decision variables).\n\n"
        "Your task is to propose a set of experiment candidates from the global plan for "
        "scientific comparison or validation (not to run them).\n"
        "Each experiment must be a fully specified configuration:\n"
        "  - Start from the given baseline configuration.\n"
        "  - First infer the core scientific question or hypothesis from the user request and plan.\n"
        "  - Identify the primary factor that must vary to test that question.\n"
        "  - Generate the minimum number of experiments needed to test that factor.\n"
        "  - Keep all other settings matched to the baseline unless the plan explicitly requires changing them.\n"
        "  - Do not introduce extra algorithm, variable, bound, or objective changes unless they are necessary "
        "to test the stated question.\n"
        "  - If the plan implies a matched comparison, vary only the primary factor across its necessary levels.\n"
        "  - If the plan implies sensitivity or ablation, generate only the minimal set needed for that design.\n"
        "  - If the user asks for baseline plus two compact what-if cases, return exactly two variation "
        "candidates unless doing so would make the design invalid.\n"
        "  - If the user asks for a combined or matched comparison, keep the collected baseline implicit and "
        "return only the non-baseline comparison configuration(s) explicitly described in the user request or plan.\n"
        "  - Prefer fewer experiments over broader exploration when both would answer the question.\n"
        "  - Do NOT invent completely unrelated fields.\n"
        "  - Ensure that each candidate is self-consistent and runnable.\n\n"
        "You MUST:\n"
        "  - Keep inp_path consistent with the baseline.\n"
        "  - Keep observed data structure compatible with the baseline.\n"
        "  - Do not return a duplicate baseline candidate; the node adds it deterministically.\n"
        "  - For a comparison task with an explicitly requested non-baseline run, do not return an empty experiments list.\n"
        "  - Use valid schema field names and canonical enum strings corresponding to concepts in the baseline, user request, or plan.\n"
        "  - For each candidate, provide a short description of what is being varied and why.\n"
        "  - Return only the smallest set of experiments that can falsify or support the hypothesis.\n"
        f"  - Return no more than {MAX_EXPERIMENT_CONFIGS - 1} variation candidates under any circumstance.\n"
        "  - Avoid expanding the design just for diversity or completeness.\n\n"
        "Return a helper_message summarising the experimental design and a list "
        f"of at most {MAX_EXPERIMENT_CONFIGS} ParameterCandidate objects in the 'experiments' field."
    )

    chat_model = _require_main_model()
    experiment_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt},
    ]
    _trace_llm_io(state, "experiment_agent", "experiment design", experiment_messages)
    out, experiment_usage = _invoke_structured_model(
        chat_model,
        ExperimentsDesign,
        experiment_messages,
    )
    _trace_llm_io(state, "experiment_agent", "experiment design", experiment_messages, out)
    _add_node_llm_calls(state, "experiment_agent")
    node_usage = _merge_token_usage(node_usage, experiment_usage)


    experiments: List[Dict[str, Any]] = []

    def _dump_object(value: Any) -> Dict[str, Any]:
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True)
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    def _dump_sequence(values: Any) -> List[Dict[str, Any]]:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            return []
        dumped: List[Dict[str, Any]] = []
        for item in values:
            if hasattr(item, "model_dump"):
                dumped.append(item.model_dump(exclude_none=True))
            elif isinstance(item, Mapping):
                dumped.append(dict(item))
        return dumped

    helper_message = (
        out.get("helper_message")
        if isinstance(out, Mapping)
        else getattr(out, "helper_message", None)
    )
    raw_experiments = (
        list(out.get("experiments") or [])
        if isinstance(out, Mapping)
        else list(getattr(out, "experiments", []) or [])
    )

    # Baseline experiment (if available). ReAct mode enforces the same contract
    # in commit_experiments(), so both modes pass an identical first candidate
    # downstream before any designed variations.
    if baseline_obj and baseline_var:
        experiments.append(
            {
                "name": "baseline",
                "description": "Baseline configuration with no overrides.",
                "objective_parameters": baseline_obj,
                "variable_parameters": baseline_var,
            }
        )

    # Candidate experiments from structured output. Enforce the cap in code even
    # if the model proposes a broader design.
    remaining_slots = max(0, MAX_EXPERIMENT_CONFIGS - len(experiments))
    if len(raw_experiments) > remaining_slots:
        print(
            f"experiment_agent: truncating {len(raw_experiments)} proposed "
            f"experiments to {remaining_slots}."
        )
    for cand in raw_experiments[:remaining_slots]:
        if isinstance(cand, Mapping):
            cand_obj = _dump_object(cand.get("objective") or cand.get("objective_parameters"))
            cand_var = _dump_sequence(cand.get("variables") or cand.get("variable_parameters"))
            name = str(cand.get("name") or f"experiment_{len(experiments) + 1}")
            description = cand.get("description")
        else:
            cand_obj = _dump_object(getattr(cand, "objective", None))
            cand_var = _dump_sequence(getattr(cand, "variables", None))
            name = str(getattr(cand, "name", "") or f"experiment_{len(experiments) + 1}")
            description = getattr(cand, "description", None)

        if str(name).strip().lower() == "baseline":
            continue

        cand_obj = _deep_merge_dict(baseline_obj, cand_obj) if baseline_obj else cand_obj
        cand_var = cand_var or baseline_var

        experiments.append(
            {
                "name": name,
                "description": description,
                "objective_parameters": cand_obj,
                "variable_parameters": cand_var,
            }
        )

    state["experiments"] = experiments
    state["experiment_helper_message"] = str(helper_message or "")

    print(str(helper_message or ""))
    print(f"experiment_agent: created {len(experiments)} experiment configs.")
    _trace_tool_call(
        state,
        "experiment_agent",
        "commit_experiments",
        result={"count": len(experiments), "names": [e.get("name") for e in experiments]},
    )

    _record_node_token_usage(state, "experiment_agent", node_usage)
    return state


def _objective_matrix(res: Any) -> np.ndarray:
    values = getattr(res, "F", None)
    if values is None:
        pop = getattr(res, "pop", None)
        if pop is not None:
            values = pop.get("F")
    if values is None:
        return np.empty((0, 0), dtype=float)
    return np.atleast_2d(np.asarray(values, dtype=float))


def _timeseries_horizon(wn) -> int:
    try:
        pstep = int(wn.options.time.pattern_timestep)
        duration = int(wn.options.time.duration)
        return int(duration // pstep) + 1
    except Exception:
        return 1


def _timeseries_seconds_axis(wn, n_points: int) -> np.ndarray:
    try:
        pstep = int(wn.options.time.pattern_timestep)
        if pstep <= 0:
            raise ValueError("pattern_timestep must be positive")
    except Exception:
        pstep = 3600
    return np.arange(max(0, int(n_points)), dtype=float) * float(pstep)


def _hourly_series_view(values: np.ndarray, wn) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.asarray([], dtype=float), arr

    seconds = _timeseries_seconds_axis(wn, arr.shape[0])
    if seconds.size == 0:
        return np.asarray([], dtype=float), arr

    hourly_idx = np.where(np.isclose(np.mod(seconds, 3600.0), 0.0))[0]
    if hourly_idx.size == 0:
        duration = float(seconds[-1]) if seconds.size else 0.0
        hour_marks = np.arange(0.0, duration + 3600.0, 3600.0)
        picks: List[int] = []
        for mark in hour_marks:
            idx = int(np.argmin(np.abs(seconds - mark)))
            if not picks or idx != picks[-1]:
                picks.append(idx)
        hourly_idx = np.asarray(picks, dtype=int)

    if hourly_idx.size == 0:
        hourly_idx = np.arange(arr.shape[0], dtype=int)

    sampled_hours = seconds[hourly_idx] / 3600.0
    return sampled_hours, arr[hourly_idx, :]


def _slugify_filename(text: Any) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return slug or "run"


def _representative_solution_vector(res: Any) -> Optional[np.ndarray]:
    X = getattr(res, "X", None)
    if X is None:
        pop = getattr(res, "pop", None)
        if pop is not None:
            try:
                X = pop.get("X")
            except Exception:
                X = None
    if X is None:
        return None

    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        return X_arr.ravel()
    if X_arr.shape[0] == 1:
        return X_arr[0].ravel()

    F_arr = _objective_matrix(res)
    if F_arr.size == 0 or F_arr.shape[0] != X_arr.shape[0]:
        return X_arr[0].ravel()

    finite_mask = np.all(np.isfinite(F_arr), axis=1)
    valid_idx = np.where(finite_mask)[0]
    if valid_idx.size == 0:
        return X_arr[0].ravel()

    F_valid = F_arr[valid_idx]
    f_min = np.min(F_valid, axis=0, keepdims=True)
    f_max = np.max(F_valid, axis=0, keepdims=True)
    denom = np.where((f_max - f_min) > 1e-12, (f_max - f_min), 1.0)
    score = np.linalg.norm((F_valid - f_min) / denom, axis=1)
    pick = int(valid_idx[int(np.argmin(score))])
    return X_arr[pick].ravel()


def _decode_timeseries_variables(
    res: Any,
    varspecs: Sequence[Any],
    wn,
) -> Dict[str, Dict[str, Any]]:
    x = _representative_solution_vector(res)
    if x is None:
        return {}

    T = _timeseries_horizon(wn)
    decoded: Dict[str, Dict[str, Any]] = {}
    ofs = 0
    for spec in varspecs or []:
        items = list(getattr(spec, "items", []) or [])
        n_items = len(items)
        is_ts = bool(getattr(spec, "timeseries", False))
        need = n_items * T if is_ts else n_items
        sl = np.asarray(x[ofs : ofs + need], dtype=float)
        ofs += need
        if sl.size != need:
            break
        if not is_ts:
            continue
        arr = sl.reshape(T, n_items)
        decoded[str(getattr(spec, "name", f"var_{len(decoded)}"))] = {
            "items": items,
            "values": arr,
        }
    return decoded


def _sample_item_indices(
    items: Sequence[str],
    limit: int,
    seed_key: str,
) -> List[int]:
    count = min(max(0, int(limit)), len(items))
    if count <= 0:
        return []
    rng = random.Random(seed_key)
    order = list(range(len(items)))
    rng.shuffle(order)
    return sorted(order[:count])


def _plot_single_timeseries_run(
    var_name: str,
    run_name: str,
    items: Sequence[str],
    values: np.ndarray,
    wn,
    output_dir: str,
    prefix: str,
    seed_key: str,
) -> Optional[str]:
    picked = _sample_item_indices(items, 4, seed_key)
    if not picked:
        return None

    xi, hourly_values = _hourly_series_view(values, wn)
    if hourly_values.ndim != 2 or hourly_values.shape[0] == 0:
        return None
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfcfd")

    for idx in picked:
        ax.plot(
            xi,
            hourly_values[:, idx],
            linewidth=2.0,
            label=str(items[idx]),
        )

    title = f"{_display_objective_name(var_name)}: {_run_label(run_name)}"
    ax.set_title(title)
    ax.set_xlabel("Time (hours)")
    unit = units.get(str(var_name), "")
    ax.set_ylabel(f"{_display_objective_name(var_name)} [{unit}]" if unit else _display_objective_name(var_name))
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.28)
    ax.legend(loc="best", frameon=True, framealpha=0.86, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    path = os.path.join(
        output_dir,
        f"{prefix}__{_slugify_filename(run_name)}__{_slugify_filename(var_name)}__timeseries.png",
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_selected_timeseries_variables(
    run_entries: Sequence[Dict[str, Any]],
    wn,
    output_dir: str = ".",
    prefix: str = "vars",
    seed: int = 0,
) -> List[str]:
    """Plot capped time-series variable figures for each run."""
    if not run_entries:
        return []

    os.makedirs(output_dir, exist_ok=True)
    saved_paths: List[str] = []
    for entry in run_entries:
        base_name = str(entry.get("base_name") or "run")
        decoded = _decode_timeseries_variables(entry.get("res"), entry.get("varspecs") or [], wn)
        for var_name, data in decoded.items():
            path = _plot_single_timeseries_run(
                var_name,
                str(entry.get("run_name") or base_name),
                data["items"],
                np.asarray(data["values"], dtype=float),
                wn,
                output_dir,
                prefix,
                seed_key=f"{seed}:{entry.get('run_name') or base_name}:{var_name}:single",
            )
            if path:
                saved_paths.append(path)

    return saved_paths


def _apply_state_random_seed(obj: Dict[str, Any], state: Mapping[str, Any]) -> Dict[str, Any]:
    if state.get("random_seed") is None:
        return obj
    seeded = dict(obj)
    seeded["seed"] = int(state.get("random_seed") or 0)
    return seeded


def _should_plot_variable_timeseries(state: Mapping[str, Any]) -> bool:
    requirements = state.get("output_requirements") or {}
    if not isinstance(requirements, Mapping):
        return True
    return bool(requirements.get("plot_variable_timeseries", True))


def running_node_workflow(state: WorkflowState) -> WorkflowState:
    """Run the committed optimisation experiments and generate figures."""

    cfg_experiments = state.get("experiments") or []
    all_results: List[Dict[str, Any]] = []
    res_list: List[Any] = []
    labels: List[str] = []
    objective_lists: List[List[str]] = []
    timeseries_run_entries: List[Dict[str, Any]] = []
    node_usage = _empty_token_usage()

    raw_configs = (
        cfg_experiments
        if cfg_experiments
        else [
            {
                "name": "optimization",
                "objective_parameters": state.get("objective_parameters", {}) or {},
                "variable_parameters": state.get("variable_parameters", []) or [],
            }
        ]
    )

    first_obj: Optional[Dict[str, Any]] = None
    baseline_label_obj: Optional[Dict[str, Any]] = None
    baseline_label_var: Optional[List[Dict[str, Any]]] = None

    for exp in raw_configs:
        name = exp.get("name", "optimization")
        obj = normalize_objective_for_runtime(exp.get("objective_parameters") or {}, NETWORK_DIR)
        obj = _apply_state_random_seed(obj, state)
        var = exp.get("variable_parameters") or []
        objectives = obj.get("objectives", []) or []
        if first_obj is None:
            first_obj = obj
            baseline_label_obj = obj
            baseline_label_var = var

        _trace_tool_call(
            state,
            "running_node",
            "run_optimization_from_json",
            {
                "name": name,
                "inp_path": obj.get("inp_path"),
                "objectives": objectives,
                "algorithm": obj.get("algorithm"),
                "termination": obj.get("termination"),
                "seed": obj.get("seed"),
                "n_variables": len(var),
            },
        )
        res, vars_out = run_optimization_from_json(obj, var)
        _trace_tool_call(
            state,
            "running_node",
            "run_optimization_from_json",
            result={
                "name": name,
                "F_shape": list(np.atleast_2d(np.asarray(getattr(res, "F", []))).shape),
            },
        )
        all_results.append(
            {
                "name": name,
                "objective_parameters": obj,
                "variable_parameters": var,
                "results": res.F,
            }
        )
        res_list.append(res)
        display_label = _run_label(
            name,
            objective_parameters=obj,
            variable_parameters=var,
            baseline_objective_parameters=baseline_label_obj,
            baseline_variable_parameters=baseline_label_var,
        )
        labels.append(display_label)
        objective_lists.append(list(objectives))
        timeseries_run_entries.append(
            {
                "base_name": name,
                "run_name": display_label,
                "suffix": None,
                "res": res,
                "varspecs": vars_out,
            }
        )

    if first_obj is None:
        raise ValueError("running_node received no optimisation configuration.")

    wn = wntr.network.WaterNetworkModel(first_obj["inp_path"])
    output_dir = state.get("run_output_dir") or "."
    optimization = OptimizationWDNPlotter(wn=wn, units=units, save_dir=str(output_dir))

    objective_meta: Union[List[str], List[List[str]]] = (
        objective_lists[0] if len(objective_lists) == 1 else objective_lists
    )

    objective_paths = optimization.plot_optimization_results(
        res=res_list,
        objectives=objective_meta,
        labels=labels,
    )
    var_paths = (
        _plot_selected_timeseries_variables(
            timeseries_run_entries,
            wn,
            output_dir=str(output_dir),
            prefix="vars",
            seed=int(state.get("random_seed", 0) or 0),
        )
        if _should_plot_variable_timeseries(state)
        else []
    )

    def _normalize_paths(x: Union[str, Sequence[str], None]) -> List[str]:
        if x is None:
            return []
        if isinstance(x, str):
            return [x]
        return list(x)

    figure_info: List[Dict[str, Any]] = []
    for p in _normalize_paths(objective_paths):
        figure_info.append(
            {
                "path": os.path.abspath(p),
                "category": "objective",
                "description": "Objective values / convergence curves or Pareto fronts.",
            }
        )
    for p in _normalize_paths(var_paths):
        figure_info.append(
            {
                "path": os.path.abspath(p),
                "category": "variables",
                "description": "Representative time-series trajectories for optimised decision variables.",
            }
        )

    _trace_tool_call(
        state,
        "running_node",
        "plot_results",
        result={"figures": [item.get("path") for item in figure_info]},
    )

    state["experiment_results"] = all_results
    state["figure_info"] = figure_info

    _record_node_token_usage(state, "running_node", node_usage)
    return state


def _normalise_report_terms(text: str) -> str:
    text = re.sub(
        r"\bpressure[- ]temporal[- ]stability\b",
        "modified resilience index",
        str(text),
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bpressure[- ]stability\b",
        "modified resilience index",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bpressure[- ]service[- ]deviation\b",
        "modified resilience index",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bexpected[- ]demand[- ]served[- ]ratio\b",
        "modified resilience index",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"\b(water[- ]service[- ]availability|WSA)\b",
        "modified resilience index",
        text,
        flags=re.IGNORECASE,
    )


def _response_finish_reason(resp: Any) -> str:
    metadata = getattr(resp, "response_metadata", None) or {}
    if isinstance(metadata, Mapping):
        for key in ("finish_reason", "stop_reason", "finish_details"):
            value = metadata.get(key)
            if value:
                return str(value).lower()
    additional = getattr(resp, "additional_kwargs", None) or {}
    if isinstance(additional, Mapping):
        for key in ("finish_reason", "stop_reason", "finish_details"):
            value = additional.get(key)
            if value:
                return str(value).lower()
    return ""


def _report_looks_truncated(text: str, resp: Any = None) -> bool:
    reason = _response_finish_reason(resp)
    if any(token in reason for token in ("length", "token", "max")):
        return True

    stripped = str(text or "").rstrip()
    if not stripped:
        return True

    last_line = stripped.splitlines()[-1].strip()
    if last_line.startswith(("#", "-", "*")) and len(last_line) < 24:
        return True

    complete_endings = (".", "!", "?", ")", "]", "`")
    if stripped.endswith(complete_endings):
        return False
    if stripped.endswith((",", ":", ";", "and", "or", "the", "a", "an")):
        return True
    return len(last_line.split()) > 6


@record_node_metrics("report_agent")
def report_agent(state: WorkflowState) -> WorkflowState:
    """Multimodal reporting agent.

    Inputs from state:
    - ``user_query``, ``plan``
    - ``experiment_results``: list of experiments with configs + res.F
    - ``figure_info``: list of dicts {path, category, description}
    - ``node_modes`` / ``node_metrics``: per-node mode and telemetry
    """
    node_usage = _empty_token_usage()

    user_query = state.get("user_query", "")
    plan = state.get("plan", "")

    experiments: List[Dict[str, Any]] = state.get("experiment_results", []) or []
    figure_info: List[Dict[str, Any]] = state.get("figure_info", []) or []

    exp_text_blocks: List[str] = []
    for r in experiments:
        exp_text_blocks.append(
            f"Experiment: {r.get('name')}\n"
            f"objective_parameters:\n{r.get('objective_parameters')}\n"
            f"variable_parameters:\n{r.get('variable_parameters')}\n"
            f"results summary (objective values / res.F):\n{r.get('results')}\n"
        )
    exp_text = "\n\n".join(exp_text_blocks) if exp_text_blocks else "No experiments or optimisation results available."

    fig_lines: List[str] = []
    for f in figure_info:
        fig_lines.append(
            f"- path: {f.get('path')}, "
            f"category: {f.get('category')}, "
            f"description: {f.get('description')}"
        )
    fig_text = "\n".join(fig_lines) if fig_lines else "No figures generated."

    system_prompt = (
        "You are the Reporting Agent for a water distribution network (WDN) "
        "optimisation framework.\n\n"
        "You are given one or more optimisation experiments (configuration + "
        "objective values) and diagnostic plots provided as images.\n\n"
        "Use the term 'modified resilience index' for the resilience objective. "
        "It is internally minimised as a negative objective value, so higher "
        "displayed MRI indicates better hydraulic resilience.\n\n"
        "The report must be analytical, not a caption list. Directly answer the "
        "user's task in the first section, then explain why the result happened "
        "using WDN hydraulics and optimisation behaviour. For each important plot, "
        "state the claim it supports, the visual evidence, the likely mechanism, "
        "and the engineering implication. Do not merely say that a curve rises, "
        "falls, clusters, or converges without explaining what that means for pump "
        "operation, pressure-dependent demand, service delivery, energy use, or "
        "model limitations. If the evidence is weak or inconclusive, say so "
        "explicitly and explain what additional run or diagnostic would be needed.\n\n"
        "Your tasks:\n"
        "1. Start with a concise answer to the user's question or optimisation goal.\n"
        "2. Summarise each experiment (objectives, key algorithm settings, main outcomes).\n"
        "3. Interpret the provided images using claim-evidence-mechanism-implication\n"
        "   reasoning, including convergence, Pareto structure, and decision-variable patterns.\n"
        "4. Compare experiments and identify which configuration best matches the user's\n"
        "   goal, including trade-offs and failure/boundary conditions where relevant.\n"
        "5. Provide a mechanistic interpretation: how pump speed changes head, "
        "flow redistribution, PDD delivered demand, and energy consumption.\n"
        "6. Provide concrete recommendations for future runs (algorithm choice,\n"
        "   constraints, bounds, additional plots).\n\n"
        "Write a complete structured markdown report in fewer than 900 words. "
        "Use clear section headings such as 'Answer', 'Experiment Summary', "
        "'Mechanism', 'Figure Evidence', 'Recommendations'."
    )

    node_modes = state.get("node_modes") or {}
    node_metrics_all = state.get("node_metrics") or {}
    mode_rows: List[str] = []
    for n in ("planning_agent", "parameter_agent", "experiment_agent", "running_node", "report_agent"):
        mode = node_modes.get(n, "workflow")
        if n not in TOGGLEABLE_NODES:
            mode = "workflow (fixed)"
        m = node_metrics_all.get(n, {}) or {}
        tu = m.get("token_usage") or {}
        mode_rows.append(
            f"  - {n}: mode={mode}, wall={m.get('wall_clock_s', 0):.2f}s, "
            f"llm_calls={m.get('n_llm_calls', 0)}, tool_calls={m.get('n_tool_calls', 0)}, "
            f"hitl_interrupts={m.get('n_user_interactions', 0)}, "
            f"tokens(in/out/total)={tu.get('input_tokens', 0)}/{tu.get('output_tokens', 0)}/{tu.get('total_tokens', 0)}, "
            f"success={m.get('success', False)}, traj_steps={m.get('trajectory_steps', 0)}"
        )
    control_transfer_block = (
        "Control-transfer configuration and per-node telemetry:\n" + "\n".join(mode_rows)
    )

    text_part = (
        f"User query:\n{user_query}\n\n"
        f"High-level plan:\n{plan}\n\n"
        f"{control_transfer_block}\n\n"
        f"Experiments and results:\n{exp_text}\n\n"
        f"Figure descriptions (for reference):\n{fig_text}\n"
    )

    # ---------- 6. Build the *image* parts (local file -> base64 data URL) ---------- #
    # This assumes your backend model understands OpenAI-style `image_url` objects.
    # If all images are PNGs you can keep 'image/png'; otherwise you can detect MIME by extension.
    image_contents: List[Dict[str, Any]] = []

    for fig in figure_info:
        img_path = fig.get("path")
        if not img_path:
            continue
        try:
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
        except FileNotFoundError:
            # Skip missing files but still keep their textual description in text_part
            continue

        # data URL format understood by OpenAI-compatible vision models
        data_url = f"data:image/png;base64,{b64}"

        image_contents.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": data_url
                },
            }
        )

    # ---------- 7. Assemble multimodal HumanMessage ---------- #
    # If your LangChain backend uses a slightly different schema, you may need
    # to adapt this, but the idea is: one message = [text, image, image, ...].
    human_content: List[Dict[str, Any]] = [{"type": "text", "text": text_part}]
    human_content.extend(image_contents)

    report_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]
    _trace_llm_io(state, "report_agent", "multimodal report", report_messages)
    resp = multi_modal.invoke(report_messages)
    _trace_llm_io(state, "report_agent", "multimodal report", report_messages, resp)
    _add_node_llm_calls(state, "report_agent")
    node_usage = _merge_token_usage(node_usage, _extract_token_usage(resp))

    report_text = _normalise_report_terms(str(resp.content))

    for _ in range(2):
        if not _report_looks_truncated(report_text, resp):
            break
        continuation_prompt = (
            "The markdown report below was truncated. Continue from the exact point "
            "where it stopped and finish the report. Return only the missing "
            "continuation text; do not repeat completed sections. End with a complete "
            "sentence. Use the term 'modified resilience index'.\n\n"
            "Original report context:\n"
            f"{text_part}\n\n"
            "Truncated report:\n"
            f"{report_text}"
        )
        continuation_messages = [
            SystemMessage(
                content=(
                    "You complete truncated WDN optimisation markdown reports. "
                    "Return only the missing continuation."
                )
            ),
            HumanMessage(content=continuation_prompt),
        ]
        _trace_llm_io(state, "report_agent", "report continuation", continuation_messages)
        resp = multi_modal.invoke(continuation_messages)
        _trace_llm_io(state, "report_agent", "report continuation", continuation_messages, resp)
        _add_node_llm_calls(state, "report_agent")
        node_usage = _merge_token_usage(node_usage, _extract_token_usage(resp))
        continuation = _normalise_report_terms(str(resp.content)).strip()
        if not continuation:
            break
        current = report_text.rstrip()
        if current.endswith((".", "!", "?", ")", "]", "`")):
            separator = "\n"
        elif continuation and continuation[0].isalnum() and current and current[-1].isalnum():
            separator = " "
        else:
            separator = ""
        report_text = current + separator + continuation

    print(report_text)


    next_state = {**state, "report": report_text}
    _record_node_token_usage(next_state, "report_agent", node_usage)
    return next_state



# =============================================================================
# Control-transfer study: ReAct-mode siblings and per-node dispatchers
# =============================================================================
# Each toggleable node has TWO implementations:
#   * <node>_workflow(state) - current structured-LLM pipeline (renamed above)
#   * <node>_react(state)    - ReAct-agent body (in tools/*_react_tools.py)
# A dispatcher with the original public name (parameter_agent, experiment_agent,
# running_node) reads state["node_modes"][<name>] and forwards to the right
# implementation. Missing entries default to "workflow".

def _dispatch_node(state: WorkflowState, node_name: str, workflow_fn, react_fn):
    mode = _resolve_node_mode(state, node_name)
    metrics = _get_node_metrics(state, node_name)
    metrics["mode"] = mode
    _set_node_metrics(state, node_name, metrics)
    if mode == "react":
        return react_fn(state)
    return workflow_fn(state)


# Late import: ReAct node bodies live in separate modules to keep
# graph_workflow.py small and to let users edit prompts/tools without touching
# the main file. They import this module's _append_configuration_trace,
# _get_node_metrics, etc.
def _import_react_nodes():
    """Lazy-import the three ReAct node implementations.

    Returns a dict {parameter_agent_react, experiment_agent_react,
    running_node_react}. If any module is missing (e.g. during refactor in
    progress), the corresponding key is a callable that raises a clear
    NotImplementedError when invoked, so workflow-mode runs are unaffected.
    """
    impls: Dict[str, Any] = {}
    try:
        from tools.parameter_react_tools import parameter_agent_react
        impls["parameter_agent_react"] = parameter_agent_react
    except Exception as exc:
        impls["parameter_agent_react"] = _missing_react_stub("parameter_agent_react", exc)
    try:
        from tools.experiment_react_tools import experiment_agent_react
        impls["experiment_agent_react"] = experiment_agent_react
    except Exception as exc:
        impls["experiment_agent_react"] = _missing_react_stub("experiment_agent_react", exc)
    try:
        from tools.running_react_tools import running_node_react
        impls["running_node_react"] = running_node_react
    except Exception as exc:
        impls["running_node_react"] = _missing_react_stub("running_node_react", exc)
    return impls


def _missing_react_stub(name: str, exc: BaseException):
    def _stub(state):
        raise NotImplementedError(
            f"{name} is requested via node_modes but its module is not importable: {exc!r}. "
            f"Ensure tools/<node>_react_tools.py is present and importable."
        )
    return _stub


_REACT_IMPLS = _import_react_nodes()


@record_node_metrics("parameter_agent")
def parameter_agent(state: WorkflowState) -> WorkflowState:
    return _dispatch_node(
        state,
        "parameter_agent",
        parameter_agent_workflow,
        _REACT_IMPLS["parameter_agent_react"],
    )


@record_node_metrics("experiment_agent")
def experiment_agent(state: WorkflowState) -> WorkflowState:
    return _dispatch_node(
        state,
        "experiment_agent",
        experiment_agent_workflow,
        _REACT_IMPLS["experiment_agent_react"],
    )


@record_node_metrics("running_node")
def running_node(state: WorkflowState) -> WorkflowState:
    return _dispatch_node(
        state,
        "running_node",
        running_node_workflow,
        _REACT_IMPLS["running_node_react"],
    )


def build_workflow_graph(include_report_agent: bool = True):
    graph = StateGraph(WorkflowState)

    graph.add_node("planning_agent", planning_agent)
    graph.add_node("parameter_agent", parameter_agent)
    graph.add_node("experiment_agent", experiment_agent)
    graph.add_node("running_node", running_node)
    if include_report_agent:
        graph.add_node("report_agent", report_agent)

    graph.set_entry_point("planning_agent")

    graph.add_edge("planning_agent", "parameter_agent")

    def route_after_parameter(state: WorkflowState) -> str:
        """
        After parameter_agent:
        - If experiment design is confirmed -> experiment_agent
        - Otherwise                         -> running_node directly
        """
        if state.get("needs_experiments") and state.get("use_experiment_agent"):
            return "experiment_agent"
        return "running_node"

    graph.add_conditional_edges(
        "parameter_agent",
        route_after_parameter,
        {
            "experiment_agent": "experiment_agent",
            "running_node": "running_node",
        },
    )

    # experiment_agent always flows into running_node

    graph.add_edge("experiment_agent", "running_node")
    if include_report_agent:
        graph.add_edge("running_node", "report_agent")
        graph.add_edge("report_agent", END)
    else:
        graph.add_edge("running_node", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def build_viz_graph():
    graph = StateGraph(WorkflowState)

    # Single node: visualization_agent
    graph.add_node("running_node", running_node)
    graph.set_entry_point("running_node")
    graph.add_edge("running_node", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
