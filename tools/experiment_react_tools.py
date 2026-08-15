"""ReAct-mode implementation of the experiment_agent node.

Used by the control-transfer study. The node emits <=5 optimisation
experiments including baseline, with variations along scientifically
meaningful axes (service-pressure level, demand model, objective set). It
deliberately has no sensitivity-sampler or dry-run simulator tool: the
workflow-mode experiment_agent has neither, and we want the comparison to
isolate control structure.

Tool set:
    read_baseline         — return the current objective/variable params
    propose_experiment    — add an experiment (overrides on baseline)
    list_experiments      — read-back of committed experiments
    remove_experiment     — drop by name
    commit_experiments    — finalise; the node returns experiments=[...]
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from tools.react_runtime import (
    bump_react_tool_call,
    ensure_react_node_metrics,
    finalize_react_node_metrics,
    mark_react_exception,
    react_invoke_config,
)

_ACTIVE_STATE: Dict[str, Any] = {}


def _state() -> Dict[str, Any]:
    return _ACTIVE_STATE


def _bump_tool_call(tool_name: str, payload: Optional[Dict[str, Any]] = None) -> None:
    bump_react_tool_call(_state(), "experiment_agent", tool_name, payload)


@tool
def read_baseline() -> Dict[str, Any]:
    """Return the baseline ``objective_parameters`` and ``variable_parameters``.

    These are the values left in state by parameter_agent. Every proposed
    experiment will start from this baseline and apply overrides.
    """
    _bump_tool_call("read_baseline", None)
    state = _state()
    return {
        "objective_parameters": state.get("objective_parameters") or {},
        "variable_parameters": state.get("variable_parameters") or [],
    }


def _apply_overrides(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge ``overrides`` into a deep copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _apply_overrides(out[k], v)
        else:
            out[k] = v
    return out


@tool
def propose_experiment(name: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Add an experiment that varies the baseline.

    Args:
        name: short label, unique within this run.
        overrides: dict with optional keys ``objective_parameters`` and/or
            ``variable_parameters``. Each is merged into the baseline.
            Use overrides to vary one factor at a time
            (e.g. {"objective_parameters":{"minimum_service_pressure": 25.0}}).

    Returns the committed experiment record, or an error if the name clashes.
    """
    _bump_tool_call("propose_experiment", {"name": name})
    state = _state()
    experiments: List[Dict[str, Any]] = list(state.get("experiments") or [])
    if any(e.get("name") == name for e in experiments):
        return {"ok": False, "error": f"experiment '{name}' already exists"}
    base_obj = state.get("objective_parameters") or {}
    base_var = state.get("variable_parameters") or []
    obj = _apply_overrides(base_obj, (overrides or {}).get("objective_parameters") or {})
    # variable overrides: if a list is provided it replaces baseline; if a dict
    # keyed by variable name is provided, merge per-name.
    var_overrides = (overrides or {}).get("variable_parameters")
    if isinstance(var_overrides, list):
        var = copy.deepcopy(var_overrides)
    elif isinstance(var_overrides, dict):
        var = copy.deepcopy(base_var)
        for v in var:
            mer = var_overrides.get(v.get("name"))
            if isinstance(mer, dict):
                v.update(mer)
    else:
        var = copy.deepcopy(base_var)
    record = {"name": name, "objective_parameters": obj, "variable_parameters": var}
    experiments.append(record)
    state["experiments"] = experiments
    return {"ok": True, "experiment": record, "count": len(experiments)}


@tool
def list_experiments() -> List[Dict[str, Any]]:
    """Return all experiments committed so far (in order)."""
    _bump_tool_call("list_experiments", None)
    return list(_state().get("experiments") or [])


@tool
def remove_experiment(name: str) -> Dict[str, Any]:
    """Drop the experiment with the given name."""
    _bump_tool_call("remove_experiment", {"name": name})
    state = _state()
    experiments = [e for e in (state.get("experiments") or []) if e.get("name") != name]
    state["experiments"] = experiments
    return {"ok": True, "remaining": [e.get("name") for e in experiments]}


def _commit_experiment_contract() -> Dict[str, Any]:
    state = _state()
    from graph_workflow import MAX_EXPERIMENT_CONFIGS

    base_obj = state.get("objective_parameters") or {}
    base_var = state.get("variable_parameters") or []
    baseline = {
        "name": "baseline",
        "description": "Baseline configuration with no overrides.",
        "objective_parameters": copy.deepcopy(base_obj),
        "variable_parameters": copy.deepcopy(base_var),
    }
    proposed = [
        copy.deepcopy(e)
        for e in (state.get("experiments") or [])
        if str(e.get("name", "")).strip().lower() != "baseline"
    ]
    exps = [baseline, *proposed[: max(0, MAX_EXPERIMENT_CONFIGS - 1)]]
    state["experiments"] = exps
    state["experiment_helper_message"] = f"Committed {len(exps)} experiments via ReAct experiment_agent."
    return {"ok": True, "experiments": exps, "count": len(exps)}


@tool
def commit_experiments() -> Dict[str, Any]:
    """Close the design loop. Returns the final committed list."""
    _bump_tool_call("commit_experiments", None)
    return _commit_experiment_contract()


EXPERIMENT_TOOLS = [
    read_baseline,
    propose_experiment,
    list_experiments,
    remove_experiment,
    commit_experiments,
]


SYSTEM_PROMPT = """You are the ReAct experiment designer for a WDN-optimisation study.

You will design a set of up to 5 total experiments including the baseline. Variations should change one scientifically meaningful factor at a time. Examples of meaningful factors:
  * minimum_service_pressure ∈ {15, 20, 25, 30} m
  * demand_model ∈ {"PDD", "DDA"}
  * objectives = single vs the same single + a second objective (only if requested)
  * pump_speed bounds tightened (e.g. lb=0.7, ub=1.3)

Process:
  1. Call read_baseline to see the configured spec.
  2. Use propose_experiment one variation at a time, with overrides shaped as
     {"objective_parameters": {...}} and/or
     {"variable_parameters": {"<varname>": {"bounds": {...}}}}.
  3. Use list_experiments to review.
  4. Call commit_experiments when satisfied. Stop after that — do not
     keep adding experiments after commit.

Rules:
  * Do not invent objectives or algorithms the user did not request.
  * Objective names are restricted to pump_energy and modified_resilience_index.
  * Multi-objective experiments must keep a multi-objective algorithm.
  * If the request asks for baseline plus two compact what-if cases, propose two variations before commit.
  * If the request asks for a combined single-/multi-objective comparison, keep the baseline single-objective and add one matched multi-objective variation using pump_energy plus modified_resilience_index with a multi-objective algorithm.
  * Vary one factor per experiment so the comparison is interpretable.
  * Do not propose a duplicate baseline; commit_experiments adds it deterministically.
"""


def _drain(out_state: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(out_state)
    for k in ("experiments", "experiment_helper_message", "node_metrics"):
        if k in _ACTIVE_STATE:
            out[k] = _ACTIVE_STATE[k]
    return out


def experiment_agent_react(state):
    global _ACTIVE_STATE
    _ACTIVE_STATE = dict(state)
    ensure_react_node_metrics(_ACTIVE_STATE, "experiment_agent")

    from graph_workflow import _extract_token_usage, _record_node_token_usage, _require_main_model, _trace_llm_io
    from langgraph.prebuilt import create_react_agent

    agent = create_react_agent(_require_main_model(), tools=EXPERIMENT_TOOLS, prompt=SYSTEM_PROMPT)

    user_query = _ACTIVE_STATE.get("user_query") or ""
    plan = _ACTIVE_STATE.get("plan") or ""
    hyp = _ACTIVE_STATE.get("hypothesis_text") or ""
    messages = [
        {"role": "user", "content": (
            f"User request:\n{user_query}\n\nPlan:\n{plan}\n\nHypothesis:\n{hyp}\n\n"
            f"Design <=5 total experiments. Do not propose a duplicate baseline; commit_experiments adds it."
        )},
    ]
    try:
        _trace_llm_io(_ACTIVE_STATE, "experiment_agent", "react agent", messages)
        result = agent.invoke(
            {"messages": messages},
            config=react_invoke_config(_ACTIVE_STATE, "experiment_agent"),
        )
        _trace_llm_io(_ACTIVE_STATE, "experiment_agent", "react agent", messages, result)
    except Exception as exc:
        mark_react_exception(_ACTIVE_STATE, "experiment_agent", exc)
        return _drain(_ACTIVE_STATE)

    msgs = result.get("messages") if isinstance(result, dict) else None
    metrics = ensure_react_node_metrics(_ACTIVE_STATE, "experiment_agent")
    if not metrics.get("react_token_callback_recorded"):
        react_usage = _extract_token_usage(result)
        _record_node_token_usage(_ACTIVE_STATE, "experiment_agent", react_usage)
    _commit_experiment_contract()
    success = bool(_ACTIVE_STATE.get("experiments"))
    finalize_react_node_metrics(
        _ACTIVE_STATE,
        "experiment_agent",
        msgs,
        success=success,
        termination_reason=None if success else "react_experiment_incomplete",
    )
    return _drain(_ACTIVE_STATE)
