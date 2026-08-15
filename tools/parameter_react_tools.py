"""ReAct-mode implementation of the parameter_agent node.

Used by the control-transfer study (workflow ↔ ReAct autonomy axis). A
ReAct agent is given a bounded tool set and asked to elicit / configure a
complete WDN-optimisation specification iteratively, asking the user when
information is missing. The HITL loop (``ask_user``) is preserved exactly
as in the workflow-mode parameter_agent so that the only thing varying
between modes is the *control structure*, not the *capabilities*.

Tool set (kept deliberately bounded — no web/literature search):
    ask_user                — pause the graph and request input via interrupt()
    set_parameter           — write a (field, value) pair into the running spec
    validate_configuration  — Pydantic-validate against ParametersState
    finalize_configuration  — close the loop, emit objective+variable params

The agent is built with ``langgraph.prebuilt.create_react_agent``.
"""
from __future__ import annotations

import time
import threading
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from tools.react_runtime import (
    bump_react_tool_call,
    ensure_react_node_metrics,
    finalize_react_node_metrics,
    mark_react_exception,
    react_invoke_config,
    set_react_node_metrics,
)

# Module-local handle on the active state so tool callables can mutate it.
# The agent is single-threaded inside one node invocation, so this is safe.
_ACTIVE_STATE: Dict[str, Any] = {}
_ASK_USER_LOCK = threading.Lock()


def _state() -> Dict[str, Any]:
    return _ACTIVE_STATE


def _canonical_parameter_field(field: str) -> str:
    """Normalize common objective-field aliases emitted by the ReAct model."""
    text = str(field or "").strip()
    changed = True
    while changed:
        changed = False
        for prefix in ("objective_parameters.", "objective."):
            if text.startswith(prefix):
                text = text[len(prefix):]
                changed = True

    parts = [part for part in text.split(".") if part]
    if (
        parts
        and parts[-1] == "minimum_service_pressure"
        and not text.startswith("variable.")
    ):
        return "minimum_service_pressure"
    return text


def _service_pressure_was_set_explicitly(state: Dict[str, Any]) -> bool:
    for entry in state.get("configuration_trace") or []:
        if not isinstance(entry, dict):
            continue
        if _canonical_parameter_field(str(entry.get("field") or "")) == "minimum_service_pressure":
            return True
    return False


def _set_experiment_routing_flag(state: Dict[str, Any]) -> None:
    if state.get("needs_experiments"):
        state.setdefault("use_experiment_agent", True)
    else:
        state["use_experiment_agent"] = False


def _trace(field: str, value: Any) -> None:
    """Append to configuration_trace and record in node_metrics."""
    field = _canonical_parameter_field(field)
    state = _state()
    trace = list(state.get("configuration_trace") or [])
    trace.append(
        {
            "node": "parameter_agent",
            "mode": "react",
            "field": str(field),
            "value": value,
            "t": time.time(),
        }
    )
    state["configuration_trace"] = trace
    metrics = ensure_react_node_metrics(state, "parameter_agent")
    cfg_set = list(metrics.get("configuration_set") or [])
    if field not in cfg_set:
        cfg_set.append(field)
    metrics["configuration_set"] = cfg_set
    set_react_node_metrics(state, "parameter_agent", metrics)
    bump_react_tool_call(state, "parameter_agent", "set_parameter", {"field": field, "value": value})


def _bump_tool_call(tool_name: str, payload: Optional[Dict[str, Any]] = None) -> None:
    bump_react_tool_call(_state(), "parameter_agent", tool_name, payload)


@tool
def ask_user(question: str, options: Optional[List[str]] = None) -> str:
    """Pause the graph and ask the user a question; return their answer.

    Use this whenever a required spec field cannot be inferred from the network
    or from prior conversation. Preserve as many user wordings as possible so
    that ``set_parameter`` writes the user's actual choice. ``options`` is an
    optional hint; the user may answer freely.
    """
    with _ASK_USER_LOCK:
        _bump_tool_call("ask_user", {"question": question, "options": options})
        state = _state()
        if not state.get("allow_human_input", True):
            raise RuntimeError(
                "parameter_agent requires additional user input, but allow_human_input is False."
            )
        metrics = ensure_react_node_metrics(state, "parameter_agent")
        metrics["n_user_interactions"] = int(metrics.get("n_user_interactions") or 0) + 1
        set_react_node_metrics(state, "parameter_agent", metrics)

        # Use LangGraph interrupt() if available (lazy import keeps this module
        # importable in standalone tests).
        try:
            from langgraph.types import interrupt  # type: ignore

            answer = interrupt({"question": question, "options": options or []})
            if isinstance(answer, dict):
                return str(answer.get("value", answer.get("answer", "")))
            return str(answer)
        except Exception:
            # Fallback (non-LangGraph context): read from a queued answers buffer.
            queue = list(state.get("_pending_user_answers") or [])
            if queue:
                answer = queue.pop(0)
                state["_pending_user_answers"] = queue
                return str(answer)
            option_text = f" Options: {options}" if options else ""
            return input(f"{question}{option_text}\n> ")


@tool
def set_parameter(field: str, value: Any) -> Dict[str, Any]:
    """Set a field on the running optimisation spec.

    ``field`` uses dot-notation under ``objective_parameters`` or
    ``variable_parameters[i]``. Examples:
        - "inp_path" → objective_parameters.inp_path
        - "objectives" → objective_parameters.objectives (list[str])
        - "demand_model" → objective_parameters.demand_model
        - "minimum_service_pressure" → objective_parameters.minimum_service_pressure
        - "pressure_min_constraint" → objective_parameters.pressure_min_constraint
        - "algorithm.name" → objective_parameters.algorithm.name
        - "algorithm.kwargs.pop_size" → algorithm.kwargs.pop_size
        - "termination.type" / "termination.value"
        - "seed" → objective_parameters.seed
        - "variable.<name>.<sub>" — e.g. "variable.pump_speed.bounds.lb"
    """
    field = _canonical_parameter_field(field)
    state = _state()
    obj = dict(state.get("objective_parameters") or {})
    variables: List[Dict[str, Any]] = list(state.get("variable_parameters") or [])

    if field.startswith("variable."):
        # variable.<name>.<sub>
        _, vname, *rest = field.split(".")
        target = next((v for v in variables if v.get("name") == vname), None)
        if target is None:
            target = {"name": vname}
            variables.append(target)
        cursor: Dict[str, Any] = target
        for key in rest[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[rest[-1]] = value
    else:
        parts = field.split(".")
        cursor = obj
        for key in parts[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[parts[-1]] = value

    state["objective_parameters"] = obj
    state["variable_parameters"] = variables
    if field == "minimum_service_pressure" and value is not None:
        state["minimum_service_pressure_confirmed"] = True
    _trace(field, value)
    return {"ok": True, "field": field, "value": value}


def _validate_configuration_contract() -> Dict[str, Any]:
    """Run schema validation on the current spec.

    Returns ``{"valid": bool, "missing_keys": [...], "errors": [...]}``.
    Required fields: inp_path, objectives, demand_model,
    pressure_min_constraint, minimum_service_pressure, algorithm.name,
    algorithm.kwargs, termination.type, termination.value, seed,
    and at least one variable_parameters entry with name + bounds.
    """
    state = _state()
    obj = state.get("objective_parameters") or {}
    variables = state.get("variable_parameters") or []

    from graph_workflow import (
        _algorithm_compatibility_question,
        _complete_parameters_from_reference,
        _contains_explicit_service_pressure,
        _fallback_questions_for_missing,
        _normalize_partial_objective,
        _normalize_partial_var,
        _ordered_missing_keys,
        _runtime_required_missing_keys,
        _validate_complete_parameters,
    )

    normalized_obj = _normalize_partial_objective(dict(obj)) if obj else {}
    normalized_vars = [
        v
        for v in (
            _normalize_partial_var(dict(item))
            for item in variables
            if isinstance(item, dict)
        )
        if v
    ]
    if not state.get("allow_human_input", True):
        normalized_obj, normalized_vars, fallback_used = _complete_parameters_from_reference(
            state,
            normalized_obj,
            normalized_vars,
        )
        if fallback_used:
            state["reference_fallback_used"] = True
            if normalized_obj.get("minimum_service_pressure") is not None:
                state["minimum_service_pressure_confirmed"] = True
    state["objective_parameters"] = normalized_obj
    state["variable_parameters"] = normalized_vars
    if (
        normalized_obj.get("minimum_service_pressure") is not None
        and not state.get("minimum_service_pressure_confirmed", False)
        and (
            _contains_explicit_service_pressure(state.get("user_query", ""))
            or _service_pressure_was_set_explicitly(state)
        )
    ):
        state["minimum_service_pressure_confirmed"] = True

    validated_obj, validated_vars = _validate_complete_parameters(
        normalized_obj, normalized_vars
    )
    missing = _ordered_missing_keys(
        _runtime_required_missing_keys(
            normalized_obj,
            normalized_vars,
            validated_obj,
            validated_vars,
            minimum_service_pressure_confirmed=bool(
                state.get("minimum_service_pressure_confirmed", False)
            ),
        )
    )
    errors: List[str] = []
    compatibility_question = _algorithm_compatibility_question(normalized_obj)
    if compatibility_question:
        errors.append(compatibility_question)
        if "objective.algorithm.name" not in missing:
            missing = _ordered_missing_keys([*missing, "objective.algorithm.name"])
    if normalized_obj and validated_obj is None and "objective.objectives" not in missing:
        errors.append("objective_parameters does not satisfy the shared ObjectiveParameters schema.")
    if normalized_vars and validated_vars is None and "variables.var_specs" not in missing:
        errors.append("variable_parameters does not satisfy the shared VarSpec schema.")

    questions = _fallback_questions_for_missing(missing, normalized_obj)
    state["missing_keys"] = missing
    state["parameter_questions"] = questions
    valid = (
        not missing
        and not errors
        and validated_obj is not None
        and validated_vars is not None
    )
    state["parameter_status"] = "COMPLETE" if valid else "COLLECTING"

    if validated_obj is not None:
        state["objective_parameters"] = validated_obj.model_dump(exclude_none=True)
    if validated_vars is not None:
        state["variable_parameters"] = [
            v.model_dump(exclude_none=True) for v in validated_vars
        ]

    return {
        "valid": valid,
        "missing_keys": missing,
        "errors": errors,
        "questions_to_user": questions,
    }


@tool
def validate_configuration() -> Dict[str, Any]:
    """Run schema validation on the current spec."""
    _bump_tool_call("validate_configuration", None)
    return _validate_configuration_contract()


def _finalize_configuration_contract() -> Dict[str, Any]:
    """Close the parameter-collection loop.

    Validates first; if valid, sets ``parameter_status = COMPLETE`` and
    returns the final spec for downstream agents. If invalid, returns the
    list of missing fields and the spec stays in ``COLLECTING`` state.
    """
    state = _state()
    report = _validate_configuration_contract()
    if not report["valid"]:
        state["parameter_status"] = "COLLECTING"
        return {"status": "COLLECTING", **report}
    state["parameter_status"] = "COMPLETE"
    _set_experiment_routing_flag(state)
    return {
        "status": "COMPLETE",
        "objective_parameters": state.get("objective_parameters"),
        "variable_parameters": state.get("variable_parameters"),
    }


@tool
def finalize_configuration() -> Dict[str, Any]:
    """Close the parameter-collection loop."""
    _bump_tool_call("finalize_configuration", None)
    return _finalize_configuration_contract()


PARAMETER_TOOLS = [
    ask_user,
    set_parameter,
    validate_configuration,
    finalize_configuration,
]


SYSTEM_PROMPT = """You are the ReAct configurator for a WDN-optimisation study on the C-Town benchmark.

Your job: build a complete, valid configuration for the optimisation. Use the tools below in a reason-and-act loop. Ask the user only for things you cannot infer from the user request, plan, or current state. Validate before finalising. The required spec is:

  objective_parameters:
    inp_path                       # network .inp path
    objectives                     # list[str], e.g. ["pump_energy"] or ["pump_energy","modified_resilience_index"]
    demand_model                   # "PDD" or "DDA"
    pressure_min_constraint        # float, m  (hard minimum pressure)
    minimum_service_pressure       # float, m  (PDD required_pressure and MRI Pstar)
    algorithm.name                 # "GA" / "DE" / "NSGA2" / ...
    algorithm.kwargs               # e.g. {"pop_size": 20}
    termination.type               # "n_gen"
    termination.value              # int
    seed                           # int
  variable_parameters[*]:
    name                           # e.g. "pump_speed"
    items                          # e.g. "ALL_PUMPS" when the user requests all pumps
    setter                         # e.g. "pump_speed_masked"
    setter_kwargs                  # e.g. {"group_name":"all"}
    timeseries                     # true / false
    bounds.lb, bounds.ub           # floats

Important rules:
  * For multi-objective (≥2 objectives) you MUST pick a multi-objective algorithm (NSGA2, SMSEMOA, ...).
  * For single-objective you MUST pick a single-objective algorithm (GA, DE, CMAES, PSO).
  * Objective names are restricted to pump_energy and modified_resilience_index.
  * Ask the user when the answer is not derivable; do not invent values for objectives, bounds, demand_model, or service pressure.
  * Ask exactly one user-facing question at a time. Do not emit multiple ask_user tool calls in the same response.
  * If several fields are missing, ask only the highest-priority missing field, wait for the answer, then continue with the next field.
  * Call validate_configuration once you think you are done. If invalid, fix and re-validate.
  * Call finalize_configuration ONLY after validate_configuration returns valid=True.
"""


def _drain_collected_state(out_state: Dict[str, Any]) -> Dict[str, Any]:
    """Merge module-local _ACTIVE_STATE back into the returned state dict."""
    src = dict(_ACTIVE_STATE)
    out = dict(out_state)
    for k in (
        "objective_parameters",
        "variable_parameters",
        "configuration_trace",
        "node_metrics",
        "parameter_status",
        "parameter_questions",
        "helper_message",
        "missing_keys",
        "minimum_service_pressure_confirmed",
        "reference_fallback_used",
        "use_experiment_agent",
    ):
        if k in src:
            out[k] = src[k]
    return out


def parameter_agent_react(state):
    """ReAct-mode parameter_agent.

    Drives a ``create_react_agent`` over the tools in PARAMETER_TOOLS until the
    agent calls ``finalize_configuration`` and the resulting spec validates.
    Mirrors the contract of ``parameter_agent_workflow``: emits
    ``objective_parameters``, ``variable_parameters``, ``parameter_status``,
    and configuration_trace / node_metrics for the post-hoc evaluator.
    """
    global _ACTIVE_STATE
    _ACTIVE_STATE = dict(state)
    ensure_react_node_metrics(_ACTIVE_STATE, "parameter_agent")

    # Build the agent on first use (avoid module-level model init).
    from graph_workflow import _extract_token_usage, _record_node_token_usage, _require_main_model, _trace_llm_io
    from langgraph.prebuilt import create_react_agent

    agent = create_react_agent(_require_main_model(), tools=PARAMETER_TOOLS, prompt=SYSTEM_PROMPT)

    user_query = _ACTIVE_STATE.get("user_query") or ""
    plan = _ACTIVE_STATE.get("plan") or ""
    hypothesis = _ACTIVE_STATE.get("hypothesis_text") or ""
    messages = [
        {"role": "user", "content": (
            f"User request:\n{user_query}\n\nPlan:\n{plan}\n\nHypothesis (if any):\n{hypothesis}\n\n"
            "Configure the optimisation, ask the user for missing fields, and finalise."
        )},
    ]
    try:
        _trace_llm_io(_ACTIVE_STATE, "parameter_agent", "react agent", messages)
        result = agent.invoke(
            {"messages": messages},
            config=react_invoke_config(_ACTIVE_STATE, "parameter_agent"),
        )
        _trace_llm_io(_ACTIVE_STATE, "parameter_agent", "react agent", messages, result)
    except Exception as exc:
        mark_react_exception(_ACTIVE_STATE, "parameter_agent", exc)
        if not _ACTIVE_STATE.get("allow_human_input", True):
            validation = _validate_configuration_contract()
            if validation.get("valid"):
                _finalize_configuration_contract()
                metrics = ensure_react_node_metrics(_ACTIVE_STATE, "parameter_agent")
                metrics["error"] = None
                metrics["success"] = True
                metrics["termination_reason"] = None
                set_react_node_metrics(_ACTIVE_STATE, "parameter_agent", metrics)
                return _drain_collected_state(_ACTIVE_STATE)
            raise
        return _drain_collected_state(_ACTIVE_STATE)

    msgs = result.get("messages") if isinstance(result, dict) else None
    metrics = ensure_react_node_metrics(_ACTIVE_STATE, "parameter_agent")
    if not metrics.get("react_token_callback_recorded"):
        react_usage = _extract_token_usage(result)
        _record_node_token_usage(_ACTIVE_STATE, "parameter_agent", react_usage)
    validation = _validate_configuration_contract()
    if validation.get("valid"):
        _finalize_configuration_contract()
    elif not _ACTIVE_STATE.get("allow_human_input", True):
        missing_desc = ", ".join(validation.get("missing_keys") or ["unknown fields"])
        raise RuntimeError(
            "parameter_agent requires additional user input, but allow_human_input is False. "
            f"Missing keys: {missing_desc}"
        )

    success = _ACTIVE_STATE.get("parameter_status") == "COMPLETE"
    finalize_react_node_metrics(
        _ACTIVE_STATE,
        "parameter_agent",
        msgs,
        success=success,
        termination_reason=None if success else "react_validation_incomplete",
    )

    return _drain_collected_state(_ACTIVE_STATE)
