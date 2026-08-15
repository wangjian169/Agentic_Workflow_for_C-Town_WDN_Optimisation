"""ReAct-mode implementation of the running_node.

The agent can run optimisations and finalise results. It has no warm-start,
no retries, and no orchestrator LLM calls, keeping comparison with workflow
mode clean.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import wntr
from langchain_core.tools import tool

from tools.react_runtime import (
    bump_react_tool_call,
    ensure_react_node_metrics,
    finalize_react_node_metrics,
    mark_react_exception,
    react_invoke_config,
)

_ACTIVE_STATE: Dict[str, Any] = {}
# Cache of pymoo result objects keyed by experiment name (kept module-local
# because they don't serialise into the LangGraph state cleanly).
_RUN_CACHE: Dict[str, Any] = {}


def _state() -> Dict[str, Any]:
    return _ACTIVE_STATE


def _expected_run_names() -> List[str]:
    experiments = _state().get("experiments") or []
    if not experiments:
        return ["optimization"]
    return [str(e.get("name") or "optimization") for e in experiments]


def _bump_tool_call(tool_name: str, payload: Optional[Dict[str, Any]] = None) -> None:
    bump_react_tool_call(_state(), "running_node", tool_name, payload)


def _run_optimization_contract(name: str) -> Dict[str, Any]:
    state = _state()
    experiments = state.get("experiments") or []
    exp = next((e for e in experiments if e.get("name") == name), None)
    if exp is None and not experiments and name in ("optimization", "baseline"):
        exp = {
            "name": name,
            "objective_parameters": state.get("objective_parameters") or {},
            "variable_parameters": state.get("variable_parameters") or [],
        }
    if exp is None:
        return {"ok": False, "error": f"no experiment named '{name}'"}

    from tools.tools import normalize_objective_for_runtime, run_optimization_from_json
    from graph_workflow import NETWORK_DIR, _apply_state_random_seed, _trace_tool_call

    obj = normalize_objective_for_runtime(exp.get("objective_parameters") or {}, NETWORK_DIR)
    obj = _apply_state_random_seed(obj, state)
    var = exp.get("variable_parameters") or []
    _trace_tool_call(
        state,
        "running_node",
        "run_optimization_from_json",
        {
            "name": name,
            "inp_path": obj.get("inp_path"),
            "objectives": obj.get("objectives") or [],
            "algorithm": obj.get("algorithm"),
            "termination": obj.get("termination"),
            "seed": obj.get("seed"),
            "n_variables": len(var),
        },
    )
    res, vars_out = run_optimization_from_json(obj, var)
    _RUN_CACHE[name] = {"res": res, "varspecs": vars_out, "objective_parameters": obj,
                       "variable_parameters": var, "objectives": obj.get("objectives") or []}

    F = getattr(res, "F", None)
    summary: Dict[str, Any] = {"ok": True, "name": name}
    try:
        import numpy as _np
        Fa = _np.asarray(F)
        if Fa.ndim == 1:
            Fa = Fa.reshape(1, -1)
        summary["n_solutions"] = int(Fa.shape[0])
        summary["F_shape"] = list(Fa.shape)
        summary["F_min"] = _np.min(Fa, axis=0).tolist()
        summary["F_mean"] = _np.mean(Fa, axis=0).tolist()
    except Exception as exc:
        summary["F_summary_error"] = str(exc)
    _trace_tool_call(state, "running_node", "run_optimization_from_json", result=summary)
    return summary


@tool
def run_optimization(name: str) -> Dict[str, Any]:
    """Execute the named experiment exactly once and return a result summary.

    The full ``res`` object is cached internally for downstream tools. Returns
    a dict with: name, n_solutions, F_shape, F_min (per objective), F_mean.
    """
    _bump_tool_call("run_optimization", {"name": name})
    return _run_optimization_contract(name)


def _finalize_results_contract() -> Dict[str, Any]:
    state = _state()
    run_names = _expected_run_names()
    for name in run_names:
        if name not in _RUN_CACHE:
            _run_optimization_contract(name)

    all_results: List[Dict[str, Any]] = []
    for name in run_names:
        cached = _RUN_CACHE.get(name)
        if cached is None:
            continue
        all_results.append({
            "name": name,
            "objective_parameters": cached["objective_parameters"],
            "variable_parameters": cached["variable_parameters"],
            "results": getattr(cached["res"], "F", None),
        })

    figure_info: List[Dict[str, Any]] = []
    if _RUN_CACHE:
        from graph_workflow import _plot_selected_timeseries_variables, _run_label, _should_plot_variable_timeseries, units
        from tools.tools import OptimizationWDNPlotter

        ordered_cache = [(name, _RUN_CACHE[name]) for name in run_names if name in _RUN_CACHE]
        first_obj_params = ordered_cache[0][1]["objective_parameters"]
        first_var_params = ordered_cache[0][1]["variable_parameters"]
        wn = wntr.network.WaterNetworkModel(first_obj_params["inp_path"])
        output_dir = state.get("run_output_dir") or "."
        plotter = OptimizationWDNPlotter(wn=wn, units=units, save_dir=str(output_dir))

        res_list = [cached["res"] for _, cached in ordered_cache]
        objective_lists = [list(cached["objectives"]) for _, cached in ordered_cache]
        labels = [
            _run_label(
                name,
                objective_parameters=cached["objective_parameters"],
                variable_parameters=cached["variable_parameters"],
                baseline_objective_parameters=first_obj_params,
                baseline_variable_parameters=first_var_params,
            )
            for name, cached in ordered_cache
        ]
        objective_meta = objective_lists[0] if len(objective_lists) == 1 else objective_lists
        objective_paths = plotter.plot_optimization_results(
            res=res_list,
            objectives=objective_meta,
            labels=labels,
        ) or []
        if isinstance(objective_paths, str):
            objective_paths = [objective_paths]
        for path in objective_paths:
            figure_info.append(
                {
                    "path": os.path.abspath(path),
                    "category": "objective",
                    "description": "Objective values / convergence curves or Pareto fronts.",
                }
            )

        entries = [
            {
                "base_name": name,
                "run_name": label,
                "suffix": None,
                "res": cached["res"],
                "varspecs": cached["varspecs"],
            }
            for (name, cached), label in zip(ordered_cache, labels)
        ]
        var_paths = (
            _plot_selected_timeseries_variables(
                entries,
                wn,
                output_dir=str(output_dir),
                prefix="vars",
                seed=int(state.get("random_seed", 0) or 0),
            )
            if _should_plot_variable_timeseries(state)
            else []
        ) or []
        if isinstance(var_paths, str):
            var_paths = [var_paths]
        for path in var_paths:
            figure_info.append(
                {
                    "path": os.path.abspath(path),
                    "category": "variables",
                    "description": "Representative time-series trajectories for optimised decision variables.",
                }
            )

    state["experiment_results"] = all_results
    state["figure_info"] = figure_info
    try:
        from graph_workflow import _trace_tool_call

        _trace_tool_call(
            state,
            "running_node",
            "finalize_results",
            result={"n_runs": len(all_results), "figures": [f.get("path") for f in figure_info]},
        )
    except Exception:
        pass
    return {"ok": True, "n_runs": len(all_results)}


@tool
def finalize_results() -> Dict[str, Any]:
    """Emit experiment_results + figure_info into state so report_agent can run."""
    _bump_tool_call("finalize_results", None)
    return _finalize_results_contract()


RUNNING_TOOLS = [run_optimization, finalize_results]


SYSTEM_PROMPT = """You are the ReAct execution agent for a WDN-optimisation study.

You will execute each committed experiment exactly once, then finalise. The finalisation step produces the same figures requested in state["output_requirements"] as workflow mode. Do not retry, do not warm-start, do not adjust the configuration.

Process:
  1. Look at the experiments list (it is implicit in state; try
     run_optimization with the most likely names: "baseline", "optimization",
     or any name the experiment_agent emitted).
  2. Call run_optimization once per experiment.
  3. Call finalize_results.
"""


def _drain(out_state: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(out_state)
    for k in ("experiment_results", "figure_info", "node_metrics"):
        if k in _ACTIVE_STATE:
            out[k] = _ACTIVE_STATE[k]
    return out


def running_node_react(state):
    global _ACTIVE_STATE, _RUN_CACHE
    _ACTIVE_STATE = dict(state)
    _RUN_CACHE = {}
    ensure_react_node_metrics(_ACTIVE_STATE, "running_node")

    from graph_workflow import _extract_token_usage, _record_node_token_usage, _require_main_model, _trace_llm_io
    from langgraph.prebuilt import create_react_agent

    agent = create_react_agent(_require_main_model(), tools=RUNNING_TOOLS, prompt=SYSTEM_PROMPT)

    experiments = _ACTIVE_STATE.get("experiments") or []
    if not experiments:
        experiments = [{"name": "optimization"}]
    names = [e.get("name", "optimization") for e in experiments]
    messages = [
        {"role": "user", "content": (
            f"Experiments to execute, in order: {names}.\n"
            f"Run each one exactly once, plot results, then finalise."
        )},
    ]
    try:
        _trace_llm_io(_ACTIVE_STATE, "running_node", "react agent", messages)
        result = agent.invoke(
            {"messages": messages},
            config=react_invoke_config(_ACTIVE_STATE, "running_node"),
        )
        _trace_llm_io(_ACTIVE_STATE, "running_node", "react agent", messages, result)
    except Exception as exc:
        mark_react_exception(_ACTIVE_STATE, "running_node", exc)
        return _drain(_ACTIVE_STATE)

    msgs = result.get("messages") if isinstance(result, dict) else None
    metrics = ensure_react_node_metrics(_ACTIVE_STATE, "running_node")
    if not metrics.get("react_token_callback_recorded"):
        react_usage = _extract_token_usage(result)
        _record_node_token_usage(_ACTIVE_STATE, "running_node", react_usage)
    if not _ACTIVE_STATE.get("experiment_results") or not _ACTIVE_STATE.get("figure_info"):
        _finalize_results_contract()
    success = bool(_ACTIVE_STATE.get("experiment_results"))
    finalize_react_node_metrics(
        _ACTIVE_STATE,
        "running_node",
        msgs,
        success=success,
        termination_reason=None if success else "react_running_incomplete",
    )
    return _drain(_ACTIVE_STATE)
