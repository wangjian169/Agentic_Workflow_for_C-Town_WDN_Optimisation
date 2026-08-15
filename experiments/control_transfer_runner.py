"""Control-transfer study runner.

Runs selected mode combinations (W/R for parameter_agent, experiment_agent,
running_node) across seed/repeat settings on the C-Town case study, and
persists per-run telemetry under the selected results root, e.g.
``results/deepseek/<spec_id>/<model_label>/<combo>/seed_<seed>/rep_<repeat>/``.

Usage::

    python -m experiments.control_transfer_runner \\
        --spec combined_obj_ctown \\
        --combo WWW WWR RRR \\
        --n-repeats 3

    # full sweep
    python -m experiments.control_transfer_runner --all
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

REFERENCE_SPECS_DIR = BASE_DIR / "reference_specs"

TOGGLEABLE = ("parameter_agent", "experiment_agent", "running_node")
NODE_ORDER = (
    "planning_agent",
    "parameter_agent",
    "experiment_agent",
    "running_node",
    "report_agent",
)
CORE_NODES = ("planning_agent", "parameter_agent", "experiment_agent", "running_node")
REPORT_NODES = ("report_agent",)
RUN_CONSOLE_TRACE = "run_console_trace.txt"
gw = None


def _workflow_module():
    global gw
    if gw is None:
        import graph_workflow as graph_workflow_module  # noqa: E402

        gw = graph_workflow_module
    return gw


def _combo_to_modes(combo: str) -> Dict[str, str]:
    """Convert a 3-letter combo like 'WRW' to a node-to-mode mapping."""
    if len(combo) != 3 or not all(c in "WR" for c in combo.upper()):
        raise ValueError(f"combo must be 3 letters from {{W,R}}, got {combo!r}")
    return {node: ("workflow" if c.upper() == "W" else "react")
            for node, c in zip(TOGGLEABLE, combo.upper())}


def _all_combos() -> List[str]:
    return ["".join(c) for c in product("WR", repeat=3)]


def _parse_combos(values: Optional[Any]) -> List[str]:
    if isinstance(values, str):
        values = [values]
    combos: List[str] = []
    for value in values or []:
        for raw in str(value).split(","):
            combo = raw.strip().upper()
            if not combo:
                continue
            _combo_to_modes(combo)
            if combo not in combos:
                combos.append(combo)
    return combos


def _build_run_plan(args: argparse.Namespace) -> List[Tuple[str, str, int, int]]:
    if args.n_repeats < 1:
        raise ValueError("--n-repeats must be >= 1")

    if args.all:
        specs = ["single_obj_ctown", "multi_obj_ctown", "combined_obj_ctown"]
        combos = _all_combos()
        n_seed_slots = args.n_seeds
        use_random_seeds = True
    else:
        if not args.spec:
            raise ValueError("--spec is required unless --all is given")
        combos = _parse_combos(args.combo)
        if not combos:
            raise ValueError("--combo must include at least one 3-letter value unless --all is given")
        specs = [args.spec]
        n_seed_slots = args.seeds or 1
        use_random_seeds = bool(args.seeds) or args.n_repeats > 1

    if n_seed_slots < 1:
        raise ValueError("--n-seeds/--seeds must be >= 1")

    if use_random_seeds:
        seed_table = _random_seed_table(n_seed_slots, args.n_repeats)
    else:
        seed_table = {(1, 1): int(args.seed)}

    return [
        (spec, combo, seed_table[(seed_slot, repeat)], repeat)
        for spec in specs
        for combo in combos
        for seed_slot in range(1, n_seed_slots + 1)
        for repeat in range(1, args.n_repeats + 1)
    ]


def _load_reference_spec(spec_id: str) -> Dict[str, Any]:
    p = REFERENCE_SPECS_DIR / f"{spec_id}.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _slugify(value: Any, default: str = "default") -> str:
    text = str(value or default).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text).strip("._-")
    return text or default


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_model_label(model_name: Any) -> str:
    text = str(model_name or "main_model").split(":")[-1].split("/")[-1]
    return _slugify(text, "main_model")


def _random_seed_table(n_seed_slots: int, n_repeats: int) -> Dict[Tuple[int, int], int]:
    rng = random.SystemRandom()
    table: Dict[Tuple[int, int], int] = {}
    used: set[int] = set()
    for seed_slot in range(1, n_seed_slots + 1):
        for repeat in range(1, n_repeats + 1):
            while True:
                seed = rng.randint(1, 2_147_483_647)
                if seed not in used:
                    used.add(seed)
                    table[(seed_slot, repeat)] = seed
                    break
    return table


def _model_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    label = (
        args.model_label
        or os.environ.get("CONTROL_MODEL_LABEL")
        or _default_model_label(args.model)
    )
    return {
        "label": str(label),
        "slug": _slugify(label, "default_model"),
        "profile": getattr(args, "model_profile", None),
        "model": args.model,
        "model_provider": args.model_provider,
        "backend": args.model_backend,
        "base_url": args.model_base_url,
        "timeout": args.model_timeout,
        "extra_body": _parse_model_extra_body(args.model_extra_body_json),
        "api_key_env": args.model_api_key_env,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }


def _parse_model_extra_body(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("--model-extra-body-json must decode to a JSON object.")
    return payload


def _configure_model(args: argparse.Namespace) -> None:
    """Configure graph_workflow.model without touching its multimodal model."""
    workflow = _workflow_module()
    kwargs: Dict[str, Any] = {}
    if args.model_provider:
        kwargs["model_provider"] = args.model_provider
    if args.model_backend:
        kwargs["backend"] = args.model_backend
    if args.model_base_url:
        kwargs["base_url"] = args.model_base_url
    if args.model_timeout is not None:
        kwargs["timeout"] = args.model_timeout
    extra_body = _parse_model_extra_body(args.model_extra_body_json)
    if extra_body:
        kwargs["extra_body"] = extra_body
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.max_tokens is not None:
        kwargs["max_tokens"] = args.max_tokens
    api_key = os.environ.get(args.model_api_key_env) if args.model_api_key_env else None
    if api_key:
        kwargs["api_key"] = api_key
    workflow.model = workflow.init_chat_model(args.model, **kwargs)


def _run_id(spec_id: str, model_slug: str, combo: str, seed: int, repeat: int) -> str:
    return f"{spec_id}__{model_slug}__{combo.upper()}__seed_{seed}__rep_{repeat:03d}"


def _unique_out_dir(base: Path) -> Path:
    if not base.exists():
        return base
    if not any(base.iterdir()):
        return base
    stamped = base.with_name(f"{base.name}__rerun_{_utc_stamp()}")
    idx = 2
    candidate = stamped
    while candidate.exists():
        candidate = base.with_name(f"{stamped.name}_{idx}")
        idx += 1
    return candidate


def _run_out_dir(
    spec_id: str,
    model_slug: str,
    combo: str,
    seed: int,
    repeat: int,
    root: Path,
) -> Path:
    base = root / spec_id / model_slug / combo.upper() / f"seed_{seed}" / f"rep_{repeat:03d}"
    return _unique_out_dir(base)


def _build_user_query(
    spec: Dict[str, Any],
    seed: int,
    single_task_description: str,
    multi_task_description: str,
    combined_task_description: str,
) -> str:
    obj = spec["objective_parameters"]
    objectives = obj["objectives"]
    spec_id = str(spec.get("spec_id") or "")
    task_kind = str(spec.get("expected_problem_kind") or "")
    if spec_id == "combined_obj_ctown" or task_kind == "combined_single_multi":
        template = combined_task_description
    else:
        template = multi_task_description if len(objectives) >= 2 else single_task_description
    return str(template).replace("{seed}", str(int(seed)))


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _console_trace_path(out_dir: Path) -> Path:
    return out_dir / RUN_CONSOLE_TRACE


class _TeeStream:
    """Write output to a per-run text file, optionally echoing to the console."""

    def __init__(self, stream, file_handle, *, echo: bool):
        self._stream = stream
        self._file = file_handle
        self._echo = bool(echo)
        self.encoding = getattr(stream, "encoding", "utf-8")

    def write(self, text):
        if self._echo:
            self._stream.write(text)
        self._file.write(text)
        return len(text)

    def flush(self):
        if self._echo:
            self._stream.flush()
        self._file.flush()

    def isatty(self):
        return bool(getattr(self._stream, "isatty", lambda: False)())


class _RunConsoleTrace:
    """Capture stdout/stderr emitted during one run, with optional console echo."""

    def __init__(self, path: Path, *, echo: bool = True):
        self.path = path
        self.echo = bool(echo)
        self._file = None
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._file = open(self.path, "w", encoding="utf-8", buffering=1)
        sys.stdout = _TeeStream(self._stdout, self._file, echo=self.echo)
        sys.stderr = _TeeStream(self._stderr, self._file, echo=self.echo)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout = self._stdout
            sys.stderr = self._stderr
            if self._file is not None:
                self._file.close()
        return False


def _print_run_trace_header(
    *,
    spec: str,
    combo: str,
    seed: int,
    repeat: int,
    model_label: str,
    out_dir: Path,
) -> None:
    width = 96
    print("=" * width)
    print("RUN CONSOLE TRACE")
    print("-" * width)
    print(f"started_at_utc : {_utc_stamp()}")
    print(f"spec           : {spec}")
    print(f"model          : {model_label}")
    print(f"combo          : {combo}")
    print(f"seed           : {seed}")
    print(f"repeat         : {repeat}")
    print(f"output_dir     : {out_dir}")
    print("=" * width)
    print()


def _print_run_trace_footer(summary: Dict[str, Any]) -> None:
    width = 96
    print()
    print("=" * width)
    print("RUN SUMMARY")
    print("-" * width)
    print(f"finished_at_utc: {_utc_stamp()}")
    print(f"wall_clock_s   : {float(summary.get('wall_clock_s') or 0.0):.2f}")
    print(f"success        : {summary.get('success')}")
    if summary.get("error"):
        print(f"error          : {summary.get('error')}")
    failures = summary.get("node_failures") or []
    if failures:
        print(f"node_failures  : {', '.join(failures)}")
    print("=" * width)


def _as_array(value: Any):
    try:
        import numpy as _np

        arr = _np.asarray(value, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
    except Exception:
        return None


def _matrix_summary(value: Any) -> Dict[str, Any]:
    arr = _as_array(value)
    if arr is None:
        return {"shape": [0, 0], "min": [], "mean": [], "finite": False}
    try:
        import numpy as _np

        finite = bool(_np.isfinite(arr).all()) if arr.size else False
        return {
            "shape": list(arr.shape),
            "min": _np.nanmin(arr, axis=0).tolist() if arr.size else [],
            "mean": _np.nanmean(arr, axis=0).tolist() if arr.size else [],
            "finite": finite,
        }
    except Exception:
        return {"shape": list(getattr(arr, "shape", (0, 0))), "min": [], "mean": [], "finite": False}


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _empty_usage() -> Dict[str, int]:
    return {
        "input_tokens": 0,
        "input_cache_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def _normalise_usage(value: Any) -> Dict[str, int]:
    usage = _empty_usage()
    if isinstance(value, dict):
        for key in usage:
            usage[key] = _int_value(value.get(key))
    if not usage["total_tokens"]:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


def _add_usage(base: Dict[str, int], update: Any) -> Dict[str, int]:
    other = _normalise_usage(update)
    return {key: _int_value(base.get(key)) + other[key] for key in _empty_usage()}


def _build_node_telemetry(
    node_metrics: Dict[str, Any],
    token_breakdown: Dict[str, Any],
) -> Dict[str, Any]:
    """Return per-node telemetry plus core/report grouped totals."""
    present_nodes = [
        node for node in NODE_ORDER
        if node in (node_metrics or {}) or node in (token_breakdown or {})
    ]

    by_node: Dict[str, Dict[str, Any]] = {}
    for node in present_nodes:
        metrics = dict((node_metrics or {}).get(node) or {})
        token_usage = _normalise_usage(
            metrics.get("token_usage") or (token_breakdown or {}).get(node)
        )
        by_node[node] = {
            "mode": metrics.get("mode"),
            "success": metrics.get("success"),
            "wall_clock_s": metrics.get("wall_clock_s", 0.0),
            "n_llm_calls": _int_value(metrics.get("n_llm_calls")),
            "n_tool_calls": _int_value(metrics.get("n_tool_calls")),
            "n_user_interactions": _int_value(metrics.get("n_user_interactions")),
            "trajectory_steps": _int_value(metrics.get("trajectory_steps")),
            "token_usage": token_usage,
            "tool_call_log": metrics.get("tool_call_log") or [],
            "termination_reason": metrics.get("termination_reason"),
            "error": metrics.get("error"),
        }
        if "react_recursion_limit" in metrics:
            by_node[node]["react_recursion_limit"] = metrics.get("react_recursion_limit")

    def _totals(nodes: List[str]) -> Dict[str, Any]:
        totals = {
            "nodes": [node for node in nodes if node in by_node],
            "wall_clock_s": 0.0,
            "n_llm_calls": 0,
            "n_tool_calls": 0,
            "n_user_interactions": 0,
            "trajectory_steps": 0,
            "token_usage": _empty_usage(),
        }
        for node in totals["nodes"]:
            record = by_node[node]
            totals["wall_clock_s"] += float(record.get("wall_clock_s") or 0.0)
            totals["n_llm_calls"] += _int_value(record.get("n_llm_calls"))
            totals["n_tool_calls"] += _int_value(record.get("n_tool_calls"))
            totals["n_user_interactions"] += _int_value(record.get("n_user_interactions"))
            totals["trajectory_steps"] += _int_value(record.get("trajectory_steps"))
            totals["token_usage"] = _add_usage(totals["token_usage"], record.get("token_usage"))
        return totals

    groups = {
        "core": _totals(list(CORE_NODES)),
        "all": _totals([*CORE_NODES, *REPORT_NODES]),
    }
    if "report_agent" in by_node:
        groups["report"] = _totals(list(REPORT_NODES))
    return {
        "by_node": by_node,
        "groups": groups,
    }


def _experiment_result_record(record: Dict[str, Any]) -> Dict[str, Any]:
    F = record.get("results")
    summary = _matrix_summary(F)
    arr = _as_array(F)
    return {
        "name": record.get("name"),
        "objectives": (record.get("objective_parameters") or {}).get("objectives"),
        "objective_parameters": _strip_non_serialisable(record.get("objective_parameters") or {}),
        "variable_parameters": _strip_non_serialisable(record.get("variable_parameters") or []),
        "F": arr.tolist() if arr is not None else _strip_non_serialisable(F),
        "F_shape": summary["shape"],
        "F_min": summary["min"],
        "F_mean": summary["mean"],
        "F_finite": summary["finite"],
    }


def _experiment_result_meta(record: Dict[str, Any]) -> Dict[str, Any]:
    summary = _matrix_summary(record.get("results"))
    return {
        "name": record.get("name"),
        "objectives": (record.get("objective_parameters") or {}).get("objectives"),
        "n_solutions": summary["shape"][0] if summary.get("shape") else 0,
        "F_shape": summary["shape"],
        "F_min": summary["min"],
        "F_mean": summary["mean"],
        "F_finite": summary["finite"],
    }


def _output_completeness(out_state: Dict[str, Any]) -> Dict[str, Any]:
    experiment_results = out_state.get("experiment_results") or []
    figure_info = out_state.get("figure_info") or []
    categories = {str(f.get("category")) for f in figure_info if isinstance(f, dict)}
    output_requirements = out_state.get("output_requirements") or {}
    required_categories = list(
        output_requirements.get("required_figure_categories")
        or ["objective", "variables"]
    )
    n_objective_figures = sum(
        1 for f in figure_info
        if isinstance(f, dict) and str(f.get("category")) == "objective"
    )
    n_variable_figures = sum(
        1 for f in figure_info
        if isinstance(f, dict) and str(f.get("category")) == "variables"
    )
    min_objective_figures = int(
        output_requirements.get("min_objective_figures")
        if "min_objective_figures" in output_requirements
        else 1
    )
    min_variable_figures = int(
        output_requirements.get("min_variable_figures")
        if "min_variable_figures" in output_requirements
        else 1
    )
    figures_complete = all(category in categories for category in required_categories)
    if "objective" in required_categories:
        figures_complete = figures_complete and n_objective_figures >= min_objective_figures
    if "variables" in required_categories:
        figures_complete = figures_complete and n_variable_figures >= min_variable_figures
    return {
        "has_experiment_results": bool(experiment_results),
        "n_experiment_results": len(experiment_results),
        "has_objective_figure": "objective" in categories,
        "has_variable_figure": "variables" in categories,
        "n_objective_figures": n_objective_figures,
        "n_variable_figures": n_variable_figures,
        "n_figures": len(figure_info),
        "required_figure_categories": required_categories,
        "min_objective_figures": min_objective_figures,
        "min_variable_figures": min_variable_figures,
        "report_present": bool(out_state.get("report")),
        "complete": bool(experiment_results) and figures_complete,
    }


def _empty_final_config() -> Dict[str, Any]:
    return {
        "objective_parameters": {},
        "variable_parameters": [],
        "experiments": [],
        "parameter_status": None,
        "missing_keys": [],
        "parameter_questions": [],
        "reference_fallback_used": False,
    }


def _empty_engineering_outputs() -> Dict[str, Any]:
    return {
        "experiment_results": [],
        "figure_info": [],
        "output_completeness": {
            "has_experiment_results": False,
            "n_experiment_results": 0,
            "has_objective_figure": False,
            "has_variable_figure": False,
            "n_objective_figures": 0,
            "n_variable_figures": 0,
            "n_figures": 0,
            "required_figure_categories": ["objective", "variables"],
            "min_objective_figures": 1,
            "min_variable_figures": 1,
            "report_present": False,
            "complete": False,
        },
    }


def _empty_process_outputs() -> Dict[str, Any]:
    return {
        "user_query": None,
        "task_description_template": None,
        "plan": None,
        "task_category": None,
        "needs_experiments": None,
        "use_experiment_agent": None,
        "planner_reason": None,
        "hypothesis_text": None,
        "mechanism_rationale": None,
        "testable_prediction": None,
        "parameter_helper_message": None,
        "experiment_helper_message": None,
        "report_path": None,
        "console_trace_path": None,
        "output_requirements": None,
    }


def _common_run_fields(
    *,
    rid: str,
    spec_id: str,
    task_kind: Optional[str],
    combo: str,
    modes: Dict[str, str],
    seed: int,
    repeat: int,
    model_meta: Dict[str, Any],
    out_dir: Path,
    include_report_agent: bool,
    allow_human_input: bool,
    trace_enabled: bool,
    trace_max_chars: int,
    react_recursion_limit: int,
) -> Dict[str, Any]:
    return {
        "run_id": rid,
        "spec_id": spec_id,
        "task_kind": task_kind,
        "combo": combo.upper(),
        "modes": modes,
        "seed": int(seed),
        "repeat": int(repeat),
        "model": model_meta,
        "run_output_dir": str(out_dir),
        "console_trace_path": str(_console_trace_path(out_dir)),
        "include_report_agent": bool(include_report_agent),
        "allow_human_input": bool(allow_human_input),
        "trace_enabled": bool(trace_enabled),
        "trace_llm_io": bool(trace_enabled),
        "trace_max_chars": int(trace_max_chars),
        "react_recursion_limit": int(react_recursion_limit),
    }


def _write_run_artifacts(
    out_dir: Path,
    *,
    summary: Dict[str, Any],
    run_metadata: Dict[str, Any],
    final_config: Dict[str, Any],
    node_metrics: Dict[str, Any],
    node_telemetry: Dict[str, Any],
    configuration_trace: List[Dict[str, Any]],
    engineering_outputs: Dict[str, Any],
    process_outputs: Dict[str, Any],
    out_state: Optional[Dict[str, Any]] = None,
    traceback_text: Optional[str] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "summary.json": summary,
        "run_metadata.json": run_metadata,
        "final_config.json": final_config,
        "node_metrics.json": node_metrics,
        "node_telemetry.json": node_telemetry,
        "configuration_trace.json": configuration_trace,
        "engineering_outputs.json": engineering_outputs,
        "process_outputs.json": _strip_non_serialisable(process_outputs),
    }
    for name, payload in artifacts.items():
        _write_json(out_dir / name, payload)

    if out_state and out_state.get("report"):
        _write_text(out_dir / "report.md", str(out_state.get("report")))
    if traceback_text:
        _write_text(out_dir / "error.txt", traceback_text)
    if out_state is not None:
        try:
            with open(out_dir / "state.pkl", "wb") as f:
                pickle.dump(out_state, f)
        except Exception:
            pass


def _write_isolated_failure(
    spec_id: str,
    combo: str,
    seed: int,
    repeat: int,
    out_dir: Path,
    model_meta: Dict[str, Any],
    exc: BaseException,
    *,
    allow_human_input: bool,
    include_report_agent: bool,
    trace_enabled: bool,
    trace_max_chars: int,
    react_recursion_limit: int,
) -> Dict[str, Any]:
    """Persist a minimal failed-run record when runner-level code escapes _run_one."""
    rid = _run_id(spec_id, model_meta["slug"], combo, seed, repeat)
    error = f"{type(exc).__name__}: {exc}"
    tb = traceback.format_exc()
    try:
        modes = _combo_to_modes(combo)
    except Exception:
        modes = {}
    finished_at = _utc_stamp()

    final_config = _empty_final_config()
    engineering_outputs = _empty_engineering_outputs()
    process_outputs = _empty_process_outputs()
    process_outputs["console_trace_path"] = str(_console_trace_path(out_dir))
    empty_telemetry = _build_node_telemetry({}, {})
    common = _common_run_fields(
        rid=rid,
        spec_id=spec_id,
        task_kind=None,
        combo=combo,
        modes=modes,
        seed=seed,
        repeat=repeat,
        model_meta=model_meta,
        out_dir=out_dir,
        include_report_agent=include_report_agent,
        allow_human_input=allow_human_input,
        trace_enabled=trace_enabled,
        trace_max_chars=trace_max_chars,
        react_recursion_limit=react_recursion_limit,
    )
    run_metadata = {
        **common,
        "started_at_utc": None,
        "finished_at_utc": finished_at,
        "wall_clock_s": 0.0,
        "success": False,
        "error": error,
        "node_failures": [],
        "failure_stage": "runner_isolation",
    }
    summary = {
        **common,
        "wall_clock_s": 0.0,
        "success": False,
        "error": error,
        "traceback": tb,
        "node_failures": [],
        "node_metrics": {},
        "workflow_token_usage": {},
        "workflow_token_breakdown": {},
        "node_telemetry": empty_telemetry,
        "telemetry_by_group": empty_telemetry["groups"],
        **final_config,
        "experiment_results_meta": [],
        "figure_info": [],
        "output_completeness": engineering_outputs["output_completeness"],
        "configuration_trace": [],
        "output_requirements": None,
        "report_present": False,
        "process_outputs_present": True,
        "planner_task_category": None,
        "planner_needs_experiments": None,
        "planner_use_experiment_agent": None,
        "failure_stage": "runner_isolation",
    }

    try:
        _write_run_artifacts(
            out_dir,
            summary=summary,
            run_metadata=run_metadata,
            final_config=final_config,
            node_metrics={},
            node_telemetry=empty_telemetry,
            configuration_trace=[],
            engineering_outputs=engineering_outputs,
            process_outputs=process_outputs,
            traceback_text=tb,
        )
    except Exception as artifact_exc:
        summary["artifact_write_error"] = f"{type(artifact_exc).__name__}: {artifact_exc}"
    return summary


def _run_one(
    spec_id: str,
    combo: str,
    seed: int,
    repeat: int,
    out_dir: Path,
    model_meta: Dict[str, Any],
    allow_human_input: bool = False,
    include_report_agent: bool = True,
    trace_enabled: bool = True,
    trace_max_chars: int = 5000,
    react_recursion_limit: int = 50,
    single_task_description: str = "",
    multi_task_description: str = "",
    combined_task_description: str = "",
) -> Dict[str, Any]:
    spec = _load_reference_spec(spec_id)
    modes = _combo_to_modes(combo)
    rid = _run_id(spec_id, model_meta["slug"], combo, seed, repeat)
    task_kind = spec.get("expected_problem_kind") or (
        "multi_objective"
        if len((spec.get("objective_parameters") or {}).get("objectives") or []) > 1
        else "single_objective"
    )
    common = _common_run_fields(
        rid=rid,
        spec_id=spec_id,
        task_kind=task_kind,
        combo=combo,
        modes=modes,
        seed=seed,
        repeat=repeat,
        model_meta=model_meta,
        out_dir=out_dir,
        include_report_agent=include_report_agent,
        allow_human_input=allow_human_input,
        trace_enabled=trace_enabled,
        trace_max_chars=trace_max_chars,
        react_recursion_limit=react_recursion_limit,
    )

    initial_state = {
        "user_query": _build_user_query(
            spec,
            seed,
            single_task_description,
            multi_task_description,
            combined_task_description,
        ),
        "allow_human_input": bool(allow_human_input),
        "trace_enabled": bool(trace_enabled),
        "trace_llm_io": bool(trace_enabled),
        "trace_max_chars": int(trace_max_chars),
        "react_recursion_limit": int(react_recursion_limit),
        "node_modes": modes,
        "node_metrics": {},
        "configuration_trace": [],
        "reference_spec_id": spec_id,
        "reference_objective_parameters": _strip_non_serialisable(
            spec.get("objective_parameters") or {}
        ),
        "reference_variable_parameters": _strip_non_serialisable(
            spec.get("variable_parameters") or []
        ),
        "output_requirements": _strip_non_serialisable(
            spec.get("output_requirements") or {}
        ),
        "random_seed": int(seed),
        "run_id": rid,
        "run_output_dir": str(out_dir),
        "objective_parameters": {},
        "variable_parameters": [],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    run_started_at = _utc_stamp()
    t0 = time.perf_counter()
    try:
        app = _workflow_module().build_workflow_graph(include_report_agent=include_report_agent)
        out_state = app.invoke(
            initial_state,
            config={"configurable": {"thread_id": rid}},
        )
        success = True
        error = None
        tb = None
    except Exception as exc:
        out_state = dict(initial_state)
        out_state["error"] = f"{type(exc).__name__}: {exc}"
        out_state["traceback"] = traceback.format_exc()
        success = False
        error = out_state["error"]
        tb = out_state["traceback"]
    elapsed = time.perf_counter() - t0
    run_finished_at = _utc_stamp()

    final_config = {
        "objective_parameters": _strip_non_serialisable(out_state.get("objective_parameters") or {}),
        "variable_parameters": _strip_non_serialisable(out_state.get("variable_parameters") or []),
        "experiments": _strip_non_serialisable(out_state.get("experiments") or []),
        "parameter_status": out_state.get("parameter_status"),
        "missing_keys": _strip_non_serialisable(out_state.get("missing_keys") or []),
        "parameter_questions": _strip_non_serialisable(out_state.get("parameter_questions") or []),
        "reference_fallback_used": bool(out_state.get("reference_fallback_used")),
    }
    node_metrics = _strip_non_serialisable(out_state.get("node_metrics") or {})
    workflow_token_breakdown = _strip_non_serialisable(out_state.get("workflow_token_breakdown") or {})
    node_telemetry = _build_node_telemetry(node_metrics, workflow_token_breakdown)
    configuration_trace = _strip_non_serialisable(out_state.get("configuration_trace") or [])
    engineering_outputs = {
        "experiment_results": [
            _experiment_result_record(r)
            for r in (out_state.get("experiment_results") or [])
            if isinstance(r, dict)
        ],
        "figure_info": _strip_non_serialisable(out_state.get("figure_info") or []),
        "output_completeness": _output_completeness(out_state),
    }
    node_failures = [
        node for node, metrics in node_metrics.items()
        if isinstance(metrics, dict) and metrics.get("success") is False
    ]
    output_complete = bool(engineering_outputs["output_completeness"].get("complete"))
    if success and (node_failures or not output_complete):
        success = False
        if node_failures:
            error = f"Run completed with failed node(s): {', '.join(node_failures)}"
        else:
            error = "Run completed but required engineering outputs are incomplete."
    process_outputs = {
        "user_query": out_state.get("user_query") or initial_state.get("user_query"),
        "task_description_template": (
            combined_task_description
            if task_kind == "combined_single_multi"
            else (
                multi_task_description
                if len((spec.get("objective_parameters") or {}).get("objectives") or []) >= 2
                else single_task_description
            )
        ),
        "plan": out_state.get("plan"),
        "task_category": out_state.get("task_category"),
        "needs_experiments": out_state.get("needs_experiments"),
        "use_experiment_agent": out_state.get("use_experiment_agent"),
        "planner_reason": out_state.get("planner_reason"),
        "hypothesis_text": out_state.get("hypothesis_text"),
        "mechanism_rationale": out_state.get("mechanism_rationale"),
        "testable_prediction": out_state.get("testable_prediction"),
        "parameter_helper_message": out_state.get("helper_message"),
        "experiment_helper_message": out_state.get("experiment_helper_message"),
        "report_path": str(out_dir / "report.md") if out_state.get("report") else None,
        "console_trace_path": str(_console_trace_path(out_dir)),
        "output_requirements": out_state.get("output_requirements") or initial_state.get("output_requirements"),
    }
    run_metadata = {
        **common,
        "started_at_utc": run_started_at,
        "finished_at_utc": run_finished_at,
        "wall_clock_s": elapsed,
        "success": success,
        "error": error,
        "node_failures": node_failures,
    }

    summary = {
        **common,
        "wall_clock_s": elapsed,
        "success": success,
        "error": error,
        "traceback": tb,
        "node_failures": node_failures,
        "node_metrics": node_metrics,
        "workflow_token_usage": out_state.get("workflow_token_usage") or {},
        "workflow_token_breakdown": workflow_token_breakdown,
        "node_telemetry": node_telemetry,
        "telemetry_by_group": node_telemetry["groups"],
        **final_config,
        "experiment_results_meta": [
            _experiment_result_meta(r)
            for r in (out_state.get("experiment_results") or [])
            if isinstance(r, dict)
        ],
        "figure_info": engineering_outputs["figure_info"],
        "output_completeness": engineering_outputs["output_completeness"],
        "configuration_trace": configuration_trace,
        "output_requirements": out_state.get("output_requirements") or initial_state.get("output_requirements"),
        "report_present": bool(out_state.get("report")),
        "process_outputs_present": True,
        "planner_task_category": out_state.get("task_category"),
        "planner_needs_experiments": out_state.get("needs_experiments"),
        "planner_use_experiment_agent": out_state.get("use_experiment_agent"),
    }

    _write_run_artifacts(
        out_dir,
        summary=summary,
        run_metadata=run_metadata,
        final_config=final_config,
        node_metrics=node_metrics,
        node_telemetry=node_telemetry,
        configuration_trace=configuration_trace,
        engineering_outputs=engineering_outputs,
        process_outputs=process_outputs,
        out_state=out_state,
        traceback_text=tb,
    )
    return summary


def _strip_non_serialisable(obj):
    """Recursively coerce values into JSON-friendly types."""
    if hasattr(obj, "tolist"):
        try:
            return _strip_non_serialisable(obj.tolist())
        except Exception:
            pass
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): _strip_non_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strip_non_serialisable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return repr(obj)


def main():
    model_profiles = {
        "deepseek": {
            "results_dir": "deepseek",
            "label": "deepseek-chat",
            "model": "deepseek:deepseek-chat",
            "provider": None,
            "backend": None,
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "",
            "timeout": 300.0,
            "extra_body_json": "",
        },
        "aliyun_qwen": {
            "results_dir": "Qwen_235B_instruct",
            "label": "qwen3-235b-a22b",
            "model": "qwen3-235b-a22b",
            "provider": "openai",
            "backend": None,
            "api_key_env": "DASHSCOPE_API_KEY",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "timeout": 300.0,
            "extra_body_json": '{"enable_thinking": false}',
        },
        "ollama_qwen": {
            "results_dir": "Qwen_4B_instruct",
            "label": "qwen3-4b-instruct-2507-server",
            "model": "kamekichi128/qwen3-4b-instruct-2507:latest",
            "provider": "openai",
            "backend": None,
            "api_key_env": "OLLAMA_API_KEY",
            "base_url": "http://localhost:11435/v1",
            "timeout": 300.0,
            "extra_body_json": "",
        },
    }
    model_profile = "deepseek"  # choose: deepseek, aliyun_qwen, ollama_qwen
    model_defaults = model_profiles[model_profile]

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--spec",
        choices=["single_obj_ctown", "multi_obj_ctown", "combined_obj_ctown"],
        default="combined_obj_ctown",
    )
    ap.add_argument("--single-task-description",
        default=(
            "For the C-Town water distribution network, can pump-speed control reduce "
            "pump energy over one operating cycle? Please compare the baseline setting "
            "with two compact what-if cases, such as changing the service-pressure "
            "requirement or tightening the pump-speed operating range. Seed={seed}."
        ), help="Task prompt used when --spec single_obj_ctown is selected. Use {seed} as the seed placeholder.")
    ap.add_argument("--multi-task-description",
        default=(
            "For the C-Town water distribution network, does pump-speed control reveal "
            "a trade-off between pump energy and operational resilience over one operating "
            "cycle? Please compare the baseline setting with two compact what-if cases, "
            "such as changing the service-pressure requirement or tightening the pump-speed "
            "operating range. Seed={seed}."
        ), help="Task prompt used when --spec multi_obj_ctown is selected. Use {seed} as the seed placeholder.")
    ap.add_argument("--combined-task-description",
        default=(
            "Optimise the C-Town water distribution network using "
            "ctown.inp. Configure the baseline as a single-objective pump-energy "
            "minimisation with pump speed time series for ALL pumps, bounds 0.5 to 1.5, "
            "PDD required pressure 20 m, GA pop_size=5, and 5 generations. Then run a "
            "matched multi-objective comparison using pump_energy and modified_resilience_index, "
            "NSGA-II pop_size=5, 5 generations, and MRI Pstar=20 m. Seed={seed}. "
            "For this combined comparison experiment, produce the single-objective convergence plot "
            "and the two-objective Pareto solution distribution plot."
        ), help="Task prompt used when --spec combined_obj_ctown is selected. Use {seed} as the seed placeholder.")
    ap.add_argument("--combo", nargs="+", default=["RWW","RRR",], help="One or more 3-letter combinations, e.g. WWW WWR RRR or WWW,WWR,RRR.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=None, help="If set, schedules N random seed slots for each chosen combo/spec.")
    ap.add_argument("--all", action="store_true", default=False, help="Full sweep 8 combos x N seeds x 3 specs.")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-repeats", "--repeats", dest="n_repeats", type=int, default=10)
    ap.add_argument("--model-profile", choices=list(model_profiles), default=model_profile)
    ap.add_argument("--results-root", default=str(Path(__file__).resolve().parent.parent / "results" / model_defaults["results_dir"]))
    ap.add_argument("--model-label", default=model_defaults["label"], help="Short label used in result folders, e.g. deepseek_chat or qwen3_4b.")
    ap.add_argument("--model", default=model_defaults["model"], help="Primary init_chat_model model string.")
    ap.add_argument("--model-provider", default=model_defaults["provider"])
    ap.add_argument("--model-backend", default=model_defaults["backend"])
    ap.add_argument("--model-api-key-env", default=model_defaults["api_key_env"])
    ap.add_argument("--model-base-url", default=model_defaults["base_url"])
    ap.add_argument("--model-timeout", type=float, default=model_defaults["timeout"])
    ap.add_argument("--model-extra-body-json", default=model_defaults["extra_body_json"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=3000)
    human_group = ap.add_mutually_exclusive_group()
    human_group.add_argument("--allow-human-input", dest="allow_human_input", action="store_true", default=False, help="Allow parameter_agent to ask for missing inputs interactively.")
    human_group.add_argument("--no-human-input", dest="allow_human_input", action="store_false", help="Disable interactive parameter questions; missing fields use the reference fallback when available.")
    ap.add_argument("--quiet", action="store_true", default=False, help="Do not echo per-run trace to terminal; full trace is still saved to run_console_trace.txt.")
    ap.add_argument("--trace-max-chars", type=int, default=5000, help="Maximum characters printed per traced LLM input/output block.")
    ap.add_argument("--react-recursion-limit", type=int, default=50, help="Maximum LangGraph steps allowed inside each ReAct node before it is marked as failed.")
    ap.add_argument("--no-report", dest="include_report", action="store_false", default=False, help="Skip report_agent.")
    args = ap.parse_args()
    explicit_args = {arg.split("=", 1)[0] for arg in sys.argv[1:] if arg.startswith("--")}
    if args.model_profile != model_profile:
        selected = model_profiles[args.model_profile]
        profile_defaults = {
            "--results-root": ("results_root", str(Path(__file__).resolve().parent.parent / "results" / selected["results_dir"])),
            "--model-label": ("model_label", selected["label"]),
            "--model": ("model", selected["model"]),
            "--model-provider": ("model_provider", selected["provider"]),
            "--model-backend": ("model_backend", selected["backend"]),
            "--model-api-key-env": ("model_api_key_env", selected["api_key_env"]),
            "--model-base-url": ("model_base_url", selected["base_url"]),
            "--model-timeout": ("model_timeout", selected["timeout"]),
            "--model-extra-body-json": ("model_extra_body_json", selected["extra_body_json"]),
        }
        for flag, (attr, value) in profile_defaults.items():
            if flag not in explicit_args:
                setattr(args, attr, value)

    try:
        plan = _build_run_plan(args)
    except ValueError as exc:
        ap.error(str(exc))

    _configure_model(args)
    model_meta = _model_metadata(args)
    results_root = Path(args.results_root)
    include_report = bool(args.include_report)
    trace_enabled = True
    echo_run_console = (not args.quiet) or bool(args.allow_human_input)

    print(f"[runner] {len(plan)} runs scheduled.")
    aggregated: List[Dict[str, Any]] = []
    for spec, combo, seed, repeat in plan:
        out_dir = results_root / "_failed_runs" / _run_id(
            spec, model_meta["slug"], combo, seed, repeat
        )
        try:
            out_dir = _run_out_dir(spec, model_meta["slug"], combo, seed, repeat, results_root)
            with _RunConsoleTrace(_console_trace_path(out_dir), echo=echo_run_console):
                _print_run_trace_header(
                    spec=spec,
                    combo=combo,
                    seed=seed,
                    repeat=repeat,
                    model_label=model_meta["label"],
                    out_dir=out_dir,
                )
                print(
                    f"[runner] >>> spec={spec} model={model_meta['label']} "
                    f"combo={combo} seed={seed} repeat={repeat}"
                )
                try:
                    summary = _run_one(
                        spec,
                        combo,
                        seed,
                        repeat,
                        out_dir,
                        model_meta,
                        allow_human_input=args.allow_human_input,
                        include_report_agent=include_report,
                        trace_enabled=trace_enabled,
                        trace_max_chars=args.trace_max_chars,
                        react_recursion_limit=args.react_recursion_limit,
                        single_task_description=args.single_task_description,
                        multi_task_description=args.multi_task_description,
                        combined_task_description=args.combined_task_description,
                    )
                except Exception as exc:
                    summary = _write_isolated_failure(
                        spec,
                        combo,
                        seed,
                        repeat,
                        out_dir,
                        model_meta,
                        exc,
                        allow_human_input=args.allow_human_input,
                        include_report_agent=include_report,
                        trace_enabled=trace_enabled,
                        trace_max_chars=args.trace_max_chars,
                        react_recursion_limit=args.react_recursion_limit,
                    )
                    print(f"[runner]     isolated failure: {summary['error']}")
                print(f"[runner]     wall={summary['wall_clock_s']:.1f}s success={summary['success']}")
                _print_run_trace_footer(summary)
        except Exception as exc:
            with _RunConsoleTrace(_console_trace_path(out_dir), echo=echo_run_console):
                _print_run_trace_header(
                    spec=spec,
                    combo=combo,
                    seed=seed,
                    repeat=repeat,
                    model_label=model_meta["label"],
                    out_dir=out_dir,
                )
                summary = _write_isolated_failure(
                    spec,
                    combo,
                    seed,
                    repeat,
                    out_dir,
                    model_meta,
                    exc,
                    allow_human_input=args.allow_human_input,
                    include_report_agent=include_report,
                    trace_enabled=trace_enabled,
                    trace_max_chars=args.trace_max_chars,
                    react_recursion_limit=args.react_recursion_limit,
                )
                print(f"[runner]     isolated failure: {summary['error']}")
                print(f"[runner]     wall={summary['wall_clock_s']:.1f}s success={summary['success']}")
                _print_run_trace_footer(summary)
        aggregated.append(summary)

    agg_path = results_root / "all_runs.jsonl"
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(agg_path, "a", encoding="utf-8") as f:
        for row in aggregated:
            f.write(json.dumps(row, default=str) + "\n")
    print(f"[runner] appended {len(aggregated)} rows to {agg_path}")


if __name__ == "__main__":
    main()
