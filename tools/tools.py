from __future__ import annotations

import json
import importlib, pathlib
import re, os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.callback import Callback
import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from autogen_agentchat.base import TaskResult
from autogen_agentchat.teams import SelectorGroupChat
import asyncio
from tools.problem import DEFAULT_MINIMUM_SERVICE_PRESSURE_M, FlexibleWDNProblem
from tools.decision_variables import *  # VarSpec + all make_setter_* factories

warnings.filterwarnings(
    "ignore",
    message=r'Not all curves were used in ".*"; added with type None, units conversion left to user',
    category=UserWarning,
    module=r"wntr\.epanet\.io",
)


# =========================================================
# Unified registry: name -> {kind, attr, factory}
#   kind:   "link" | "node" | None        (None => no prior)
#   attr:   str | [str,...] | None        (for prior extraction)
#   factory: callable(**kwargs) -> setter
# =========================================================
SETTERS: Dict[str, Dict[str, Any]] = {
    # Selection-only
    "random_selection_mask": {
        "kind": None, "attr": None, "factory": make_setter_random_selection_mask
    },

    # Link/node scalar setters
    "pipe_roughness_masked": {
        "kind": "link", "attr": "roughness", "factory": make_setter_pipe_roughness_masked
    },
    "pipe_diameter_masked": {
        "kind": "link", "attr": "diameter", "factory": make_setter_pipe_diameter_masked
    },
    "pipe_status_masked": {
        "kind": "link", "attr": ["status", "initial_status"], "factory": make_setter_pipe_status_masked
    },
    "valve_status_masked": {
        "kind": "link", "attr": ["status", "initial_status"], "factory": make_setter_valve_status_masked
    },
    "valve_setting_masked": {
        "kind": "link", "attr": "setting", "factory": make_setter_valve_setting_masked
    },
    "pump_speed_masked": {
        "kind": "link", "attr": ["speed", "initial_speed"], "factory": make_setter_pump_speed_masked
    },
    "node_emitter_masked": {
        "kind": "node", "attr": ["emitter_coefficient", "emitter"], "factory": make_setter_node_emitter_masked
    },
    "demand_multiplier_masked": {
        "kind": "node", "attr": ["demand_multiplier", "pattern_multiplier"], "factory": make_setter_demand_multiplier_masked
    },
    "junction_initial_quality_masked": {
        "kind": "node", "attr": "initial_quality", "factory": make_setter_junction_initial_quality_masked
    },
    "reservoir_head_masked": {
        "kind": "node", "attr": ["head", "base_head", "initial_head"], "factory": make_setter_reservoir_head_masked
    },

    # Time series / misc
    "source_strength_timeseries": {
        "kind": "node", "attr": ["source_strength", "source_quality"], "factory": make_setter_source_strength_timeseries
    },
    "tank_initial_level_masked": {
        "kind": "node", "attr": ["initial_level", "init_level"], "factory": make_setter_tank_initial_level_masked
    },
    "tank_min_level_masked": {
        "kind": "node", "attr": "min_level", "factory": make_setter_tank_min_level_masked
    },
    "tank_max_level_masked": {
        "kind": "node", "attr": "max_level", "factory": make_setter_tank_max_level_masked
    },
    "tank_volume_curve": {
        "kind": None, "attr": None, "factory": make_setter_tank_volume_curve
    },
    "randomize_existing_source_types": {
        "kind": None, "attr": None, "factory": make_setter_randomize_existing_source_types
    },
}


# =========================
# Items resolution
# =========================
def _pool_by_keyword(wn: wntr.network.WaterNetworkModel, key: str) -> List[str]:
    """
    Map a keyword to a list of item names.
    Accepts both 'junctions_all' and 'ALL_JUNCTIONS' styles.
    """
    k = key.strip().lower()
    if k in ("junctions_all", "all_junctions"):   return list(wn.junction_name_list)
    if k in ("pipes_all",     "all_pipes"):       return list(wn.pipe_name_list)
    if k in ("pumps_all",     "all_pumps"):       return list(wn.pump_name_list)
    if k in ("valves_all",    "all_valves"):      return list(wn.valve_name_list)
    if k in ("reservoirs_all","all_reservoirs"):  return list(wn.reservoir_name_list)
    if k in ("tanks_all",     "all_tanks"):       return list(wn.tank_name_list)
    if k in ("sources_all",   "all_sources"):     return list(wn.source_name_list)
    raise ValueError(f"Unknown items keyword: {key}")


def _resolve_items(wn: wntr.network.WaterNetworkModel, spec: Dict[str, Any]) -> List[str]:
    """
    'items' supports:
      - list: ["P1","P2",...]
      - keyword str: "pipes_all" / "junctions_all" / ...
      - dict with regex: {"from": "junctions_all", "regex": "J.*"}
    """
    items_spec = spec.get("items")
    if isinstance(items_spec, list):
        return list(items_spec)
    if isinstance(items_spec, str):
        return _pool_by_keyword(wn, items_spec)
    if isinstance(items_spec, dict) and "regex" in items_spec:
        pool = _pool_by_keyword(wn, items_spec.get("from", "junctions_all"))
        pat = re.compile(items_spec["regex"])
        return [x for x in pool if pat.match(x)]
    raise ValueError(f"Unable to resolve items for VarSpec: {spec.get('name')}")


# =========================
# Prior / bounds / kwargs helpers
# =========================
def _get_attr_first(obj: Any, names: Any, default: float = np.nan) -> float:
    """Return first available attribute among names; default if none exists."""
    if names is None:
        return default
    if isinstance(names, (list, tuple)):
        for n in names:
            if hasattr(obj, n):
                return getattr(obj, n)
        return default
    return getattr(obj, names, default)


def _build_prior_from_registry(setter_key: str,
                               wn: wntr.network.WaterNetworkModel,
                               items: List[str]) -> Optional[np.ndarray]:
    """Use SETTERS to build prior vector; None if no attr defined."""
    entry = SETTERS.get(setter_key)
    if not entry or entry["attr"] is None or entry["kind"] is None:
        return None
    getter = wn.get_link if entry["kind"] == "link" else wn.get_node
    attr = entry["attr"]
    try:
        return np.array([float(_get_attr_first(getter(nm), attr)) for nm in items], float)
    except Exception:
        return None


def _parse_bounds(bounds_cfg: Any, n_items: int) -> List[Tuple[float, float]]:
    """
    Normalize bounds into per-item list of (lo, hi).
    Accepts:
      - dict: {'lb':..,'ub':..} or {'lo':..,'hi':..} -> replicated
      - pair: [lo,hi]/(lo,hi)                         -> replicated
      - list of pairs/dicts: [[lo,hi], ...]           -> validated length
    """
    if isinstance(bounds_cfg, dict):
        lo = float(bounds_cfg.get("lb", bounds_cfg.get("lo")))
        hi = float(bounds_cfg.get("ub", bounds_cfg.get("hi")))
        return [(lo, hi)] * n_items

    if isinstance(bounds_cfg, (list, tuple)) and len(bounds_cfg) == 2 and not isinstance(bounds_cfg[0], (list, tuple, dict)):
        lo, hi = map(float, bounds_cfg)
        return [(lo, hi)] * n_items

    if isinstance(bounds_cfg, (list, tuple)):
        out: List[Tuple[float, float]] = []
        for it in bounds_cfg:
            if isinstance(it, dict):
                lo = float(it.get("lb", it.get("lo")))
                hi = float(it.get("ub", it.get("hi")))
            else:
                lo, hi = map(float, it)
            out.append((lo, hi))
        if len(out) != n_items:
            raise ValueError(f"Bounds length {len(out)} != n_items {n_items}")
        return out

    raise TypeError(f"Unsupported bounds: {bounds_cfg!r}")


def _parse_setter_kwargs(raw: Any) -> Dict[str, Any]:
    """
    Normalize setter_kwargs:
      - dict -> as is
      - JSON string -> parsed dict
      - plain string -> {'mask_key': <string>}
      - None -> {}
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        return {"mask_key": s}
    raise TypeError(f"setter_kwargs must be dict/str/None, got {type(raw)}")


def _build_varspecs(wn: wntr.network.WaterNetworkModel, var_cfgs: List[Dict[str, Any]]) -> List[VarSpec]:
    """Build VarSpec list from config blocks using the unified registry."""
    specs: List[VarSpec] = []
    for cfg in var_cfgs:
        name = cfg["name"]
        setter_key = cfg["setter"]
        entry = SETTERS.get(setter_key)
        if entry is None:
            raise ValueError(f"Unknown setter '{setter_key}'")

        items = _resolve_items(wn, cfg)
        bounds = _parse_bounds(cfg["bounds"], len(items))
        setter = entry["factory"](**_parse_setter_kwargs(cfg.get("setter_kwargs")))
        prior = _build_prior_from_registry(setter_key, wn, items)
        timeseries = bool(cfg.get("timeseries", False))

        specs.append(VarSpec(
            name=name,
            items=items,
            bounds=bounds,
            setter=setter,
            prior=prior,
            timeseries=timeseries,
        ))
    return specs


# =========================
# Observed data loader
# =========================
def _load_observed(obs_cfg: Optional[Dict[str, Any]]
                   ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]]]:
    """Return (observed, inferred_selected_names) from CSV definitions."""
    observed: Dict[str, pd.DataFrame] = {}
    inferred: Dict[str, List[str]] = {}
    if not obs_cfg:
        return observed, inferred

    for k, c in obs_cfg.items():
        df = pd.read_csv(
            c["csv"],
            index_col=c.get("index_col", 0),
            parse_dates=c.get("parse_dates", False),
        )
        observed[k] = df
        inferred[k] = [str(col) for col in df.columns]
    return observed, inferred


# =========================
# Algorithm loader (pymoo)
# =========================
_ALGO_NAME_HINTS = {
    "GA": "pymoo.algorithms.soo.nonconvex.ga.GA",
    "NSGA2": "pymoo.algorithms.moo.nsga2.NSGA2",
    "SMSEMOA": "pymoo.algorithms.moo.sms.SMSEMOA",
    "MOEAD": "pymoo.algorithms.moo.moead.MOEAD",
    "AGEMOEA": "pymoo.algorithms.moo.age.AGEMOEA",
    "RVEA": "pymoo.algorithms.moo.rvea.RVEA",
    "DE": "pymoo.algorithms.soo.nonconvex.de.DE",
    "CMAES": "pymoo.algorithms.soo.nonconvex.cmaes.CMAES",
    "PSO": "pymoo.algorithms.soo.nonconvex.pso.PSO",
}
_SINGLE_OBJECTIVE_ALGOS = {"GA", "DE", "CMAES", "PSO"}
_MULTI_OBJECTIVE_ALGOS = {"NSGA2", "SMSEMOA", "MOEAD", "AGEMOEA", "RVEA"}
_ELIMINATE_DUPLICATES_ALGOS = {"GA", "NSGA2", "SMSEMOA", "AGEMOEA"}
# MO algorithms that require a reference-direction set as a constructor argument.
_REF_DIR_REQUIRED_ALGOS = {"MOEAD", "RVEA"}

def _load_class(path: str):
    mod, cls = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)

def _normalize_name(name: str) -> str:
    return str(name).strip().replace("-", "").replace("_", "").upper()


def _validate_algorithm_family(name_norm: str, n_objectives: Optional[int]):
    if n_objectives is None:
        return
    allowed = _MULTI_OBJECTIVE_ALGOS if int(n_objectives) > 1 else _SINGLE_OBJECTIVE_ALGOS
    if name_norm not in allowed:
        family = "multi-objective" if int(n_objectives) > 1 else "single-objective"
        supported = ", ".join(sorted(allowed))
        raise ValueError(
            f"Algorithm {name_norm} is not supported for {family} runs. "
            f"Supported algorithms: {supported}"
        )


def _sanitize_algorithm_kwargs(name_norm: str, kwargs: dict) -> dict:
    """Drop cross-algorithm kwargs known to conflict with pymoo constructors."""
    raw = dict(kwargs or {})
    if name_norm not in _ELIMINATE_DUPLICATES_ALGOS and "eliminate_duplicates" in raw:
        raw.pop("eliminate_duplicates", None)
        print(f"[algorithm kwargs] dropped unsupported kwargs for {name_norm}: ['eliminate_duplicates']")
    return raw


def _default_ref_dirs(n_obj: int, pop_size: Optional[int] = None):
    """Return a sensible default reference-direction set for MOEAD/RVEA."""
    from pymoo.util.ref_dirs import get_reference_directions
    if n_obj <= 1:
        raise ValueError(f"ref_dirs require n_obj >= 2, got {n_obj}")
    # Das-Dennis partition for 2-obj gives n_partitions+1 directions.
    n_partitions = 12 if n_obj == 2 else 6
    return get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)


def _build_supported_algorithm(name: str, kwargs: dict, n_objectives: Optional[int]):
    name_norm = _normalize_name(name)
    path = _ALGO_NAME_HINTS.get(name_norm)
    if not path:
        supported = ", ".join(sorted(_ALGO_NAME_HINTS))
        raise ValueError(
            f"Unsupported algorithm name: {name}. Supported algorithms: {supported}"
        )
    _validate_algorithm_family(name_norm, n_objectives)
    sanitized = _sanitize_algorithm_kwargs(name_norm, kwargs)
    cls = _load_class(path)

    if name_norm in _REF_DIR_REQUIRED_ALGOS:
        if "ref_dirs" not in sanitized:
            if n_objectives is None or int(n_objectives) < 2:
                raise ValueError(
                    f"{name_norm} requires reference directions and thus n_objectives>=2"
                )
            sanitized["ref_dirs"] = _default_ref_dirs(int(n_objectives), sanitized.get("pop_size"))
        # MOEAD/RVEA constructors don't accept 'pop_size'; population size is
        # determined by the number of reference directions.
        if name_norm == "MOEAD":
            sanitized.pop("pop_size", None)

    return cls(**sanitized)


def _choose_algorithm(
    algo_cfg: Optional[dict | str],
    algo_kwargs: Optional[dict] = None,
    n_objectives: Optional[int] = None,
):
    """
    Accepts:
      - "ga" / "nsga2" / "smsemoa" / "de" / "cmaes" / "pso"
      - {"name": "ga", "kwargs": {...}}
    Merge order: algo_kwargs < per-block kwargs (latter wins).
    """
    if algo_cfg is None:
        algo_cfg = "ga"
    base_kwargs = dict(algo_kwargs or {})

    # string shorthand
    if isinstance(algo_cfg, str):
        return _build_supported_algorithm(algo_cfg, base_kwargs, n_objectives)

    # dict spec
    if isinstance(algo_cfg, dict):
        kw = dict(base_kwargs)
        kw.update(algo_cfg.get("kwargs") or {})
        if algo_cfg.get("class"):
            raise ValueError(
                "Class-based algorithm specs are disabled in this workflow. "
                "Use one of the supported algorithm names instead."
            )
        if algo_cfg.get("name"):
            return _build_supported_algorithm(algo_cfg["name"], kw, n_objectives)
    raise ValueError(f"Invalid algorithm spec: {algo_cfg!r}")


class HistoryRecorder(Callback):
    """Capture objective matrices at the end of each pymoo generation."""

    def __init__(self):
        super().__init__()
        self.history_F: List[np.ndarray] = []

    def notify(self, algorithm):
        pop = getattr(algorithm, "pop", None)
        if pop is None:
            return
        try:
            F = pop.get("F")
        except Exception:
            return
        if F is None:
            return
        arr = np.atleast_2d(np.asarray(F, dtype=float))
        if arr.size == 0:
            return
        arr = arr[np.all(np.isfinite(arr), axis=1)]
        if arr.size:
            self.history_F.append(arr.copy())


def _extract_history_F_from_result(res) -> List[np.ndarray]:
    """Best-effort extraction of generation objective matrices from a pymoo result."""
    explicit = getattr(res, "history_F", None)
    if explicit is not None:
        return [np.atleast_2d(np.asarray(f, dtype=float)).copy() for f in explicit]

    out: List[np.ndarray] = []
    for alg in getattr(res, "history", []) or []:
        pop = getattr(alg, "pop", None)
        if pop is None:
            continue
        try:
            F = pop.get("F")
        except Exception:
            continue
        if F is None:
            continue
        arr = np.atleast_2d(np.asarray(F, dtype=float))
        arr = arr[np.all(np.isfinite(arr), axis=1)]
        if arr.size:
            out.append(arr.copy())
    return out


# =========================
# Main
# =========================
def run_optimization_from_json(objective_parameters, variable_parameters):
    """
    objective JSON: inp_path, objectives, observed, demand_model, etc.
    variable  JSON: variables (see _build_varspecs)
    """
    # with open('results/json_files/objective_parameters.json', "r", encoding="utf-8") as f:
    #     objective_parameters = json.load(f)
    # with open('results/json_files/variable_parameters.json', "r", encoding="utf-8") as f:
    #     variable_parameters = json.load(f)

    # 1) Network
    inp_path = objective_parameters["inp_path"]
    wn = wntr.network.WaterNetworkModel(inp_path)

    # 2) Variables
    var_specs = _build_varspecs(wn, variable_parameters)

    # 3) Objectives
    objectives: List[str] = objective_parameters["objectives"]
    objective_weights = objective_parameters.get("objective_weights")

    # 4) Observed & selected_names
    observed, selected_names = _load_observed(objective_parameters.get("observed"))

    # 5) Optional pricing inputs
    price_pattern = None
    if objective_parameters.get("price_pattern"):
        price_pattern = pd.read_csv(
            objective_parameters["price_pattern"]["csv"],
            index_col=0,
            parse_dates=objective_parameters["price_pattern"].get("parse_dates", False),
        )

    # 6) Other params
    demand_model = objective_parameters.get("demand_model", "DD")
    use_epanet_toolkit = bool(objective_parameters.get("use_epanet_toolkit", True))
    minimum_service_pressure = objective_parameters.get("minimum_service_pressure")
    if minimum_service_pressure is None:
        minimum_service_pressure = objective_parameters.get("service_pressure_min")
    if minimum_service_pressure is None:
        minimum_service_pressure = DEFAULT_MINIMUM_SERVICE_PRESSURE_M
    pressure_min_constraint = objective_parameters.get("pressure_min_constraint", 0.0)
    detection_limit = objective_parameters.get("detection_limit", 1e-6)
    hydraulic_minimum_pressure = objective_parameters.get("hydraulic_minimum_pressure", 0.0)
    simulation_duration_hours = objective_parameters.get("simulation_duration_hours")
    pressure_constraint_nodes = objective_parameters.get("pressure_constraint_nodes", "all")

    # 7) Problem
    problem = FlexibleWDNProblem(
        inp_path=inp_path,
        var_specs=var_specs,
        objectives=objectives,
        objective_weights=objective_weights,
        observed=observed,
        selected_names=selected_names,
        price_pattern=price_pattern,
        use_epanet_toolkit=use_epanet_toolkit,
        pressure_min_constraint=pressure_min_constraint,
        minimum_service_pressure=minimum_service_pressure,
        demand_model=demand_model,
        hydraulic_minimum_pressure=hydraulic_minimum_pressure,
        simulation_duration_hours=simulation_duration_hours,
        pressure_constraint_nodes=pressure_constraint_nodes,
        detection_limit=detection_limit,
    )

    # 8) Algorithm & termination
    algo = _choose_algorithm(
        objective_parameters.get("algorithm"),
        objective_parameters.get("algorithm_kwargs"),
        n_objectives=len(objectives),
    )
    term_cfg = objective_parameters.get("termination", {"type": "n_gen", "value": 20})
    termination = get_termination(term_cfg["type"], term_cfg.get("value"))

    seed = objective_parameters.get("seed", 1)
    verbose = bool(objective_parameters.get("verbose", True))

    # 9) Optimize
    history_recorder = HistoryRecorder()
    if len(objectives)==1:
        res = minimize(
            problem,
            algo,
            termination,
            seed=seed,
            verbose=verbose,
            save_history=True,
            callback=history_recorder,
        )
    else:
        res = minimize(
            problem,
            algo,
            termination,
            seed=seed,
            save_history=True,
            callback=history_recorder,
        )
    res.history_F = history_recorder.history_F or _extract_history_F_from_result(res)

    # 10) Brief report
    if getattr(res, "X", None) is None:
        pop = res.pop
        CV, F = pop.get("CV"), pop.get("F")
        i = int(np.argmin(CV))
        print("No feasible X yet. Best by CV:", CV[i], "F:", F[i])
    else:
        print("Best F:", res.F, "x-len:", len(res.X))

    return res, var_specs


def _atomic_write_json(path: pathlib.Path, obj: dict, encoding="utf-8"):
    tmp = pathlib.Path(str(path) + ".tmp")
    with tmp.open("w", encoding=encoding) as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def normalize_objective_for_runtime(obj: dict, NETWORK_DIR:str) -> dict:
    """
    Make sure inp_path and observed[*].csv are absolute paths before running
    optimisation or loading the network.
    """
    if not obj:
        return obj

    obj = dict(obj)  # shallow copy

    def _resolve_path(path_value: str) -> str:
        if os.path.isabs(path_value):
            return path_value

        candidates = [
            path_value,
            os.path.join(os.getcwd(), path_value),
            os.path.join(NETWORK_DIR, path_value),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
        return os.path.join(NETWORK_DIR, path_value)

    # inp_path
    inp = obj.get("inp_path")
    if isinstance(inp, str) and not os.path.isabs(inp):
        obj["inp_path"] = _resolve_path(inp)

    observed = obj.get("observed") or {}
    new_observed = {}
    for key, series_cfg in observed.items():
        if isinstance(series_cfg, dict):
            series_cfg = dict(series_cfg)
            csv_path = series_cfg.get("csv")
            if isinstance(csv_path, str) and not os.path.isabs(csv_path):
                series_cfg["csv"] = _resolve_path(csv_path)
            new_observed[key] = series_cfg
        else:
            new_observed[key] = series_cfg

    if new_observed:
        obj["observed"] = new_observed

    return obj


def run_team_sync(team: SelectorGroupChat, task: str):
    async def _run():
        history = []

        async for message in team.run_stream(task=task):
            if isinstance(message, TaskResult):
                print("\n=== TaskResult ===")
                print("Stop Reason:", message.stop_reason)
            else:
                print(f"\n=== {message.source} ===")
                print(message.content)
                if message.source!='user':
                    history.append(message)
        return history

    return asyncio.run(_run())




class OptimizationWDNPlotter:
    """
    Combined plotting utility for:
      1) Optimisation result diagnostics (convergence, Pareto fronts)
      2) Variable visualisation (time series + topology heatmaps)
      3) Single decision solution visualisation from a Pareto set

    Images are always saved to disk (no plt.show) and each public
    method returns the list of file paths that were generated.
    """

    def __init__(
        self,
        wn: Optional[wntr.network.model.WaterNetworkModel] = None,
        units: Optional[Dict[str, str]] = None,
        time_index: Optional[Sequence[Any]] = None,
        save_dir: str = ".",
        topology_node_size: float = 6.0,
    ):
        self.wn = wn
        self.units = units or {}
        self.time_index = time_index
        self.save_dir = save_dir
        self.topology_node_size = topology_node_size
        os.makedirs(self.save_dir, exist_ok=True)

    @staticmethod
    def _shorten_text(text: Any, max_len: int = 34) -> str:
        s = str(text)
        return s if len(s) <= max_len else s[: max_len - 3].rstrip() + "..."

    @staticmethod
    def _compact_legend_label(text: Any, max_len: int = 16) -> str:
        """Keep experiment labels short enough to be useful in plot legends."""
        raw = str(text).strip()
        normalized = re.sub(r"[^a-z0-9.]+", "_", raw.lower()).strip("_")
        numbers = []
        for value in re.findall(r"\d+(?:\.\d+)?", normalized):
            try:
                number = float(value)
                numbers.append(str(int(number)) if number.is_integer() else f"{number:.3g}")
            except Exception:
                numbers.append(value)

        if "baseline" in normalized or normalized == "base" or normalized.startswith("base_"):
            return "Baseline"

        if "speed" in normalized:
            prefix = "Tight speed" if re.search(r"tight|narrow|bounded|range", normalized) else "Speed"
            if len(numbers) >= 2:
                try:
                    lo, hi = float(numbers[0]), float(numbers[1])
                    if lo > 2 and hi > 2:
                        lo /= 10.0
                        hi /= 10.0
                    lo_s = str(int(lo)) if lo.is_integer() else f"{lo:.3g}"
                    hi_s = str(int(hi)) if hi.is_integer() else f"{hi:.3g}"
                except Exception:
                    lo_s, hi_s = numbers[0], numbers[1]
                return f"{prefix} ({lo_s}-{hi_s})"
            return prefix

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

        label = re.sub(
            r"expected[\s_-]*demand[\s_-]*served[\s_-]*ratio",
            "service",
            raw,
            flags=re.IGNORECASE,
        )
        label = re.sub(
            r"pump[\s_-]*energy",
            "energy",
            label,
            flags=re.IGNORECASE,
        )
        label = re.sub(r"[\s_-]+", " ", label).strip()
        label = re.sub(r"\s+", " ", label)
        if len(label) <= max_len:
            return label
        return label[: max_len - 3].rstrip() + "..."

    @staticmethod
    def _legend_ncols(n_items: int, max_cols: int = 4, target_rows: int = 4) -> int:
        if n_items <= 0:
            return 1
        return max(1, min(max_cols, math.ceil(n_items / target_rows)))

    def _place_legend_below(
        self,
        fig,
        ax,
        *,
        fontsize: int = 8,
        max_cols: int = 4,
        target_rows: int = 4,
    ) -> float:
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return 0.12
        ncol = self._legend_ncols(len(handles), max_cols=max_cols, target_rows=target_rows)
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=ncol,
            frameon=False,
            fontsize=fontsize,
            handlelength=1.8,
            columnspacing=1.0,
            borderaxespad=0.0,
        )
        rows = math.ceil(len(handles) / ncol)
        return min(0.34, 0.16 + 0.045 * max(0, rows - 1))

    def _place_legend_inside_bottom(
        self,
        ax,
        *,
        fontsize: int = 8,
        max_cols: int = 3,
        target_rows: int = 2,
    ) -> None:
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return
        ncol = self._legend_ncols(len(handles), max_cols=max_cols, target_rows=target_rows)
        ax.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.03),
            ncol=ncol,
            frameon=True,
            framealpha=0.86,
            facecolor="white",
            edgecolor="#d0d0d0",
            fontsize=fontsize,
            handlelength=1.8,
            columnspacing=0.9,
            borderpad=0.45,
        )

    @staticmethod
    def _reserve_inner_legend_space(ax, values: Sequence[float], fraction: float = 0.28) -> None:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        ymin = float(np.min(arr))
        ymax = float(np.max(arr))
        span = ymax - ymin
        if span <= 0:
            pad = max(abs(ymin) * 0.01, 1.0)
        else:
            pad = span * fraction
        ax.set_ylim(ymin - pad, ymax + span * 0.06 + pad * 0.12)

    @staticmethod
    def _display_name(name: str) -> str:
        mapping = {
            "expected_demand_served_ratio": "demand service",
            "modified_resilience_index": "modified resilience index",
            "pump_energy": "pump energy",
        }
        return mapping.get(str(name), str(name).replace("_", " "))

    def _axis_label(self, name: str, unit: Optional[str] = None) -> str:
        label = self._display_name(name)
        return label + (f" ({unit})" if unit else "")

    @staticmethod
    def _display_objective_values(name: str, values: Any) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if str(name) in {"expected_demand_served_ratio", "modified_resilience_index"}:
            return np.clip(-arr, 0.0, 1.0)
        return arr

    @staticmethod
    def _best_display_objective_value(name: str, values: Any) -> float:
        display_values = OptimizationWDNPlotter._display_objective_values(name, values)
        display_values = display_values[np.isfinite(display_values)]
        if display_values.size == 0:
            return float("nan")
        if str(name) in {"expected_demand_served_ratio", "modified_resilience_index"}:
            return float(np.max(display_values))
        return float(np.min(display_values))

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    def _ensure_dir(self):
        os.makedirs(self.save_dir, exist_ok=True)

    def _get_T(self) -> int:
        """Infer number of pattern steps from self.wn."""
        if self.wn is None:
            raise ValueError("WaterNetworkModel (wn) is not set on the plotter.")
        try:
            from tools.help_functions import _pattern_len as _pattern_len_impl  # type: ignore
            return int(_pattern_len_impl(self.wn))
        except Exception:
            pstep = int(self.wn.options.time.pattern_timestep)
            return int(self.wn.options.time.duration // pstep) + 1

    def _is_node(self, name_: str) -> bool:
        if self.wn is None:
            raise ValueError("WaterNetworkModel (wn) is not set on the plotter.")
        try:
            self.wn.get_node(name_)
            return True
        except Exception:
            return False

    def _is_link(self, name_: str) -> bool:
        if self.wn is None:
            raise ValueError("WaterNetworkModel (wn) is not set on the plotter.")
        try:
            self.wn.get_link(name_)
            return True
        except Exception:
            return False

    def _save_fig(self, fig, fname: str) -> str:
        """Save figure into save_dir with fname and close."""
        self._ensure_dir()
        fpath = os.path.join(self.save_dir, fname)
        fig.savefig(fpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return fpath

    def _topology_plot_kwargs(self) -> Dict[str, Any]:
        return {
            "show_plot": False,
            "node_size": self.topology_node_size,
        }

    def _build_time_axis(self, T_local: int) -> (np.ndarray, str):
        """Get x axis (xi) and label for time-series plots."""
        if self.time_index is None:
            xi = np.arange(T_local)
            xlab = "Time step"
        else:
            xi = np.asarray(self.time_index)
            if xi.size != T_local:
                raise ValueError(
                    f"time_index length {xi.size} != T {T_local}"
                )
            xlab = "Time"
        return xi, xlab

    def _normalize_meta(
        self,
        meta: Optional[Union[Sequence[str], Sequence[Sequence[str]]]],
        name: str,
        n_res: int,
    ) -> List[Optional[List[str]]]:
        """Normalise objectives/units into per-result lists."""
        if meta is None:
            return [None] * n_res
        meta = list(meta)  # type: ignore[arg-type]
        if meta and isinstance(meta[0], str):
            return [list(meta)] * n_res
        if len(meta) != n_res:
            raise ValueError(
                f"Length of {name} must match number of results or be a single list."
            )
        out: List[Optional[List[str]]] = []
        for m in meta:
            if m is None:
                out.append(None)
            else:
                out.append(list(m))  # type: ignore[arg-type]
        return out

    # ------------------------------------------------------------------
    # Common topology helpers (used by variables + decision-solution)
    # ------------------------------------------------------------------
    def _check_topology_items(
        self,
        var_name: str,
        items_list: List[str],
        node_flags: List[bool],
        link_flags: List[bool],
    ):
        """Check if items are all nodes or all links, raise if invalid."""
        missing = [
            nm for nm, isN, isL in zip(items_list, node_flags, link_flags)
            if not isN and not isL
        ]
        if missing:
            raise ValueError(
                f"[topology] {var_name}: items not found in nodes/links: {missing}"
            )

        all_nodes = all(node_flags)
        all_links = all(link_flags)
        if not (all_nodes or all_links):
            raise ValueError(
                f"[topology] {var_name}: items must be ALL nodes or ALL links "
                f"(require all True). node_flags={node_flags}, link_flags={link_flags}"
            )
        return all_nodes, all_links

    def _plot_topology_single(
            self,
            var_name: str,
            items: List[str],
            vals: np.ndarray,
            node_flags: List[bool],
            link_flags: List[bool],
            unit: Optional[str],
            title_suffix: str,
            fname: str,
    ) -> str:
        """Single topology plot (used by decision solution)."""
        all_nodes, all_links = self._check_topology_items(
            var_name, items, node_flags, link_flags
        )
        vals = np.asarray(vals, float).ravel()
        if vals.size != len(items):
            raise ValueError(
                f"[topology] {var_name}: value length mismatch "
                f"({vals.size} != {len(items)})"
            )

        fig, ax = plt.subplots(figsize=(6, 5))
        attr = {nm: float(v) for nm, v in zip(items, vals)}


        if all_nodes:
            title = f"{var_name} (nodes)" + (f" [{unit}]" if unit else "")
            wntr.graphics.plot_network(
                self.wn,
                node_attribute=attr,
                title=title + title_suffix,
                ax=ax,
                **self._topology_plot_kwargs(),
            )
        elif all_links:
            title = f"{var_name} (links)" + (f" [{unit}]" if unit else "")
            wntr.graphics.plot_network(
                self.wn,
                link_attribute=attr,
                title=title + title_suffix,
                ax=ax,
                **self._topology_plot_kwargs(),
            )

        plt.tight_layout()
        return self._save_fig(fig, fname)

    def _plot_topology_multi_runs(
            self,
            var_name: str,
            items_list: List[str],
            group_entries: List[Dict[str, Any]],
            labels: Sequence[str],
            unit: Optional[str],
            fname: str,
    ) -> str:
        """Topology plots for multiple runs of the same variable (multi-subplot)."""
        meta0 = group_entries[0]["meta"]
        node_flags = meta0["node_flags"]
        link_flags = meta0["link_flags"]

        all_nodes, all_links = self._check_topology_items(
            var_name, items_list, node_flags, link_flags
        )

        n_runs_in_group = len(group_entries)
        ncols = min(2, n_runs_in_group)
        nrows = math.ceil(n_runs_in_group / ncols)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 4 * nrows)
        )
        if isinstance(axes, np.ndarray):
            axes_arr = axes.flatten()
        else:
            axes_arr = np.array([axes])

        n_items = len(items_list)

        for idx, entry in enumerate(group_entries):
            ax = axes_arr[idx]
            run_idx = entry["run_idx"]
            label = self._shorten_text(labels[run_idx], 24)
            vals = np.asarray(entry["values"], float).ravel()

            if vals.size != n_items:
                raise ValueError(
                    f"[topology] {var_name} run {label}: value length mismatch "
                    f"({vals.size} != {n_items})"
                )

            attr = {nm: float(v) for nm, v in zip(items_list, vals)}
            if all_nodes:
                title = f"{self._display_name(var_name)} (nodes) - {label}" + (f" [{unit}]" if unit else "")
                wntr.graphics.plot_network(
                    self.wn,
                    node_attribute=attr,
                    title=title,
                    ax=ax,
                    **self._topology_plot_kwargs(),
                )
            elif all_links:
                title = f"{self._display_name(var_name)} (links) - {label}" + (f" [{unit}]" if unit else "")
                wntr.graphics.plot_network(
                    self.wn,
                    link_attribute=attr,
                    title=title,
                    ax=ax,
                    **self._topology_plot_kwargs(),
                )

        for j in range(len(group_entries), len(axes_arr)):
            fig.delaxes(axes_arr[j])

        plt.tight_layout()
        return self._save_fig(fig, fname)

    # ------------------------------------------------------------------
    # Common time-series helpers (used by variables + decision-solution)
    # ------------------------------------------------------------------
    def _plot_timeseries_multi_runs(
        self,
        var_name: str,
        group_entries: List[Dict[str, Any]],
        items_list: List[str],
        labels: Sequence[str],
        unit: Optional[str],
        fname: str,
    ) -> str:
        """Time-series for multiple runs of same variable."""
        fig, ax = plt.subplots(figsize=(9.5, 4.8))

        for entry in group_entries:
            run_idx = entry["run_idx"]
            label = self._compact_legend_label(labels[run_idx], 14)
            arr = entry["values"]  # shape (T, n_items)
            T_local = arr.shape[0]
            xi, xlab = self._build_time_axis(T_local)

            for j, item_id in enumerate(items_list):
                y = arr[:, j]
                line_label = self._shorten_text(f"{label}:{item_id}", 22)
                ax.plot(xi, y, label=line_label)

        title = self._axis_label(var_name, unit)
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel(title)
        ax.grid(True, linestyle="--", alpha=0.25)
        bottom = self._place_legend_below(fig, ax, fontsize=7, max_cols=4, target_rows=4)
        fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=bottom)

        return self._save_fig(fig, fname)

    def _plot_timeseries_single_solution(
        self,
        var_name: str,
        arr: np.ndarray,
        items: List[str],
        unit: Optional[str],
        sol_idx: int,
        fname: str,
    ) -> str:
        """Time-series for a single solution (one run, one x)."""
        T_local, n_items = arr.shape
        xi, xlab = self._build_time_axis(T_local)

        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        for j, item_id in enumerate(items):
            y = arr[:, j]
            ax.plot(xi, y, label=self._shorten_text(str(item_id), 22))

        title = self._axis_label(var_name, unit)
        ax.set_title(title + f" [solution #{sol_idx}]")
        ax.set_xlabel(xlab)
        ax.set_ylabel(title)
        ax.grid(True, linestyle="--", alpha=0.25)
        bottom = self._place_legend_below(fig, ax, fontsize=7, max_cols=4, target_rows=4)
        fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=bottom)

        return self._save_fig(fig, fname)

    # ------------------------------------------------------------------
    # 1) Optimisation results (formerly plot_optimization_results)
    # ------------------------------------------------------------------
    def _plot_single_objective_groups(
        self,
        groups: Dict[tuple, List[Dict[str, Any]]],
        prefix: str,
    ) -> List[str]:
        """Single-objective convergence curves for all groups in one figure."""
        if not groups:
            return []

        n_groups = len(groups)
        ncols = 1 if n_groups == 1 else 2
        nrows = math.ceil(n_groups / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(8.8 * ncols, 4.8 * nrows))
        if n_groups == 1:
            axes = np.array([axes])  # type: ignore[assignment]
        axes = np.asarray(axes).flatten()  # type: ignore[assignment]

        for ax_idx, (obj_key, exps) in enumerate(groups.items()):
            ax = axes[ax_idx]
            obj_name = obj_key[0]
            all_fvals: List[float] = []

            for e in exps:
                res = e["res"]
                label = self._compact_legend_label(e["label"], 16)
                units = e["units"]

                gens: List[int] = []
                fvals: List[float] = []
                if not hasattr(res, "history") or len(res.history) == 0:  # type: ignore[arg-type]
                    continue

                for g, algo in enumerate(res.history):  # type: ignore[attr-defined]
                    F = algo.pop.get("F")
                    if F is None or len(F) == 0:
                        continue
                    F_arr = np.asarray(F).reshape(len(F), -1)
                    fvals.append(self._best_display_objective_value(obj_name, F_arr[:, 0]))
                    gens.append(g)

                if not gens:
                    continue

                all_fvals.extend(fvals)
                ax.plot(
                    gens,
                    fvals,
                    marker="o",
                    markersize=4,
                    linewidth=1.8,
                    linestyle="-",
                    label=label,
                )

                ax.set_xlabel("Generation")
                ax.set_ylabel(self._axis_label(obj_name, units[0] if units else None))
                ax.set_title(f"Convergence: {self._display_name(obj_name)}")
                ax.grid(True, linestyle="--", alpha=0.3)
                ax.margins(x=0.03, y=0.08)

            self._reserve_inner_legend_space(ax, all_fvals, fraction=0.30)
            self._place_legend_inside_bottom(
                ax,
                fontsize=8,
                max_cols=3,
                target_rows=2,
            )

        # Delete unused subplots
        for j in range(len(groups), len(axes)):
            fig.delaxes(axes[j])

        fig.subplots_adjust(
            left=0.08,
            right=0.98,
            top=0.88,
            bottom=0.12,
            wspace=0.28,
            hspace=0.72,
        )
        fname = f"{prefix}__single_objectives.png"
        fpath = self._save_fig(fig, fname)
        return [fpath]

    def _plot_mixed_objective_groups(
        self,
        groups: Dict[tuple, List[Dict[str, Any]]],
        prefix: str,
    ) -> List[str]:
        """Handle mixed single/multi-objective groups."""
        saved_paths: List[str] = []

        single_obj_groups = {k: v for k, v in groups.items() if len(k) == 1}
        multi2_groups = {k: v for k, v in groups.items() if len(k) == 2}
        multi3plus_groups = {k: v for k, v in groups.items() if len(k) >= 3}

        # 1) Single-objective convergence
        if single_obj_groups:
            saved_paths.extend(
                self._plot_single_objective_groups(
                    single_obj_groups, prefix=f"{prefix}__single"
                )
            )

        # 2) Two-objective Pareto fronts
        if multi2_groups:
            n_groups = len(multi2_groups)
            ncols = 1 if n_groups == 1 else 2
            nrows = math.ceil(n_groups / ncols)

            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
            if n_groups == 1:
                axes = np.array([axes])  # type: ignore[assignment]
            axes = np.asarray(axes).flatten()  # type: ignore[assignment]

            for ax_idx, (obj_key, exps) in enumerate(multi2_groups.items()):
                ax = axes[ax_idx]
                obj1, obj2 = obj_key

                for e in exps:
                    F = np.asarray(e["F_final"], dtype=float).copy()
                    label = self._shorten_text(e["label"])
                    F[:, 0] = self._display_objective_values(obj1, F[:, 0])
                    F[:, 1] = self._display_objective_values(obj2, F[:, 1])
                    ax.scatter(F[:, 0], F[:, 1], s=25, alpha=0.7, label=label)

                units_any = exps[0]["units"] or [None, None]

                ax.set_xlabel(self._axis_label(obj1, units_any[0] if units_any else None))
                ax.set_ylabel(self._axis_label(obj2, units_any[1] if units_any and len(units_any) > 1 else None))
                ax.set_title("Pareto front")
                ax.grid(True)
                ax.legend(loc="best", frameon=True, framealpha=0.75, fontsize=8)

            for j in range(len(multi2_groups), len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            fname = f"{prefix}__pareto2.png"
            saved_paths.append(self._save_fig(fig, fname))

        # 3) 3+ objectives
        for obj_key, exps in multi3plus_groups.items():
            n_obj = len(obj_key)
            if n_obj < 3:
                continue

            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")

            obj1, obj2, obj3 = obj_key[:3]

            for e in exps:
                F = np.asarray(e["F_final"], dtype=float).copy()
                label = self._shorten_text(e["label"])
                for idx, obj_name in enumerate(obj_key[:3]):
                    F[:, idx] = self._display_objective_values(obj_name, F[:, idx])
                ax.scatter(F[:, 0], F[:, 1], F[:, 2], s=25, alpha=0.7, label=label)

            units_any = exps[0]["units"] or [None] * n_obj

            def _build_label(name: str, idx: int) -> str:
                return self._axis_label(
                    name,
                    units_any[idx] if units_any and len(units_any) > idx else None,
                )

            ax.set_title("Pareto 3D")
            ax.set_xlabel(_build_label(obj1, 0))
            ax.set_ylabel(_build_label(obj2, 1))
            ax.set_zlabel(_build_label(obj3, 2))
            ax.legend(loc="best", frameon=True, framealpha=0.75, fontsize=8)

            plt.tight_layout()
            obj_tag = "_".join(obj_key[:3])
            fname = f"{prefix}__pareto3d__{obj_tag}.png"
            saved_paths.append(self._save_fig(fig, fname))

        return saved_paths

    def _plot_timeseries_multi_vars_and_runs(
        self,
        ts_groups: Dict[tuple, List[Dict[str, Any]]],
        labels: Sequence[str],
        units: Dict[str, str],
        prefix: str,
    ) -> List[str]:
        """
        Draw all time-series variables in ONE figure with a grid of subplots.
        Each subplot = one variable; within each subplot, multiple runs are overlaid.
        """
        n_vars = len(ts_groups)
        if n_vars == 0:
            return []

        ncols = 1 if n_vars == 1 else 2
        nrows = math.ceil(n_vars / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
        if n_vars == 1:
            axes = np.array([axes])  # type: ignore[assignment]
        axes = np.asarray(axes).flatten()  # type: ignore[assignment]

        for ax_idx, (key, group_entries) in enumerate(ts_groups.items()):
            (var_name, _is_ts, items_tuple) = key
            ax = axes[ax_idx]
            items_list = list(items_tuple)
            unit = units.get(var_name)

            #  run 
            arr0 = np.asarray(group_entries[0]["values"])
            T_local = arr0.shape[0]
            xi, xlab = self._build_time_axis(T_local)

            for entry in group_entries:
                run_idx = entry["run_idx"]
                label = self._shorten_text(labels[run_idx], 24)
                arr = np.asarray(entry["values"])
                if arr.shape[0] != T_local:
                    raise ValueError(
                        f"{var_name}: different time length across runs "
                        f"({arr.shape[0]} vs {T_local})"
                    )
                for j, item_id in enumerate(items_list):
                    y = arr[:, j]
                    line_label = f"{label}:{item_id}"
                    ax.plot(xi, y, label=line_label)

            title = self._axis_label(var_name, unit)
            ax.set_title(title)
            ax.set_xlabel(xlab)
            ax.set_ylabel(title)
            ax.legend(loc="best", frameon=True, framealpha=0.75, fontsize=7)

        #  subplot
        for j in range(n_vars, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        fname = f"{prefix}__timeseries.png"
        fpath = self._save_fig(fig, fname)
        return [fpath]

    def plot_optimization_results(
            self,
            res: Union[Any, Sequence[Any]],
            objectives: Optional[Union[Sequence[str], Sequence[Sequence[str]]]] = None,
            labels: Optional[Sequence[str]] = None,
            prefix: str = "opt_results",
    ) -> List[str]:

        # Normalise res
        if not isinstance(res, (list, tuple)):
            res_list = [res]
        else:
            res_list = list(res)

        n_res = len(res_list)
        if n_res == 0:
            raise ValueError("No result provided.")

        # Labels
        if labels is None:
            labels = [f"run_{i + 1}" for i in range(n_res)]
        elif len(labels) != n_res:
            raise ValueError("Length of labels must match number of results.")

        # Objectives
        obj_lists = self._normalize_meta(objectives, "objectives", n_res)

        exps: List[Dict[str, Any]] = []
        for i, r in enumerate(res_list):
            if not hasattr(r, "F"):
                raise ValueError(f"Result #{i} has no attribute 'F'.")

            F_final = np.atleast_2d(np.asarray(r.F))
            n_obj = F_final.shape[1]

            if obj_lists[i] is None or len(obj_lists[i]) != n_obj:
                obj_names = [f"Objective {j + 1}" for j in range(n_obj)]
                obj_lists[i] = obj_names
            else:
                obj_names = obj_lists[i]  # type: ignore[assignment]

            unit_list = [
                self.units.get(name) if isinstance(self.units, dict) else None
                for name in obj_names
            ]

            exps.append(
                {
                    "res": r,
                    "F_final": F_final,
                    "n_obj": n_obj,
                    "objectives": obj_names,
                    "units": unit_list,
                    "label": labels[i],
                }
            )

        groups: Dict[tuple, List[Dict[str, Any]]] = {}
        for e in exps:
            key = tuple(e["objectives"])
            groups.setdefault(key, []).append(e)

        if all(e["n_obj"] == 1 for e in exps):
            return self._plot_single_objective_groups(groups, prefix=prefix)

        return self._plot_mixed_objective_groups(groups, prefix=prefix)

    # ------------------------------------------------------------------
    # 2) Variables (formerly plot_vars)
    # ------------------------------------------------------------------
    def plot_variables(
            self,
            res: Union[Any, Sequence[Any]],
            varspecs: Union[Sequence[Any], Sequence[Sequence[Any]]],
            labels: Optional[Sequence[str]] = None,
            prefix: str = "vars",
    ) -> List[str]:
        """
        Visualise optimisation variables for one or multiple experiments.

        Layout logic
        ------------
        1) All experiments share the SAME variable set
           (same var name, same timeseries flag, same items):
             - Time-series variables:
                 * ONE figure per variable.
                 * Inside each figure, all runs are overlaid in the same axes.
             - Topology variables:
                 * ONE big grid figure.
                 * Each subplot = (run, variable) pair.

        2) Experiments have DIFFERENT variable sets:
             - Each (run, variable) pair is drawn as its own figure.
        """
        if self.wn is None:
            raise ValueError("WaterNetworkModel (wn) must be set on the plotter.")

        self._ensure_dir()
        units = self.units

        # --- normalise res ------------------------------------------------
        if not isinstance(res, (list, tuple)):
            res_list = [res]
        else:
            res_list = list(res)

        n_res = len(res_list)
        if n_res == 0:
            raise ValueError("No results provided.")

        # --- normalise varspecs -------------------------------------------
        if not isinstance(varspecs, (list, tuple)):
            raise ValueError("varspecs must be a list or list of lists")

        if len(varspecs) == 0:
            raise ValueError("varspecs is empty")

        if hasattr(varspecs[0], "name"):
            varspecs_list = [list(varspecs)] * n_res  # type: ignore[arg-type]
        else:
            if len(varspecs) != n_res:
                raise ValueError(
                    "Length of varspecs must match number of res if you pass list of lists"
                )
            varspecs_list = [list(vs) for vs in varspecs]  # type: ignore[assignment]

        # labels
        if labels is None:
            labels = [f"run_{i + 1}" for i in range(n_res)]
        elif len(labels) != n_res:
            raise ValueError("Length of labels must match number of res")

        T = self._get_T()

        decoded_all: List[Dict[str, np.ndarray]] = []
        meta_all: List[Dict[str, Dict[str, Any]]] = []

        # --- 1. decode each result into variables -------------------------
        for i, (r, specs) in enumerate(zip(res_list, varspecs_list)):
            if not hasattr(r, "X"):
                raise ValueError(f"Result #{i} has no attribute 'X'")
            x = np.asarray(r.X, dtype=float).ravel()  # type: ignore[attr-defined]

            decoded_i: Dict[str, np.ndarray] = {}
            meta_i: Dict[str, Dict[str, Any]] = {}

            ofs = 0
            for spec in specs:
                items = list(spec.items)
                n_items = len(items)
                is_ts = bool(getattr(spec, "timeseries", False))
                need = n_items * T if is_ts else n_items

                sl = x[ofs:ofs + need]
                if sl.size != need:
                    raise ValueError(f"[decode] {spec.name}: expect {need}, got {sl.size}")
                ofs += need

                # mask vars only consume length
                name_lower = str(spec.name).lower()
                setter = getattr(spec, "setter", None)
                if "mask" in name_lower or setter == "random_selection_mask":
                    continue

                if is_ts:
                    arr = sl.reshape(T, n_items)
                else:
                    arr = sl.reshape(n_items, )
                decoded_i[spec.name] = arr

                node_flags = [self._is_node(nm) for nm in items]
                link_flags = [self._is_link(nm) for nm in items]
                meta_i[spec.name] = {
                    "items": tuple(items),
                    "timeseries": is_ts,
                    "node_flags": node_flags,
                    "link_flags": link_flags,
                }

            if ofs != x.size:
                raise ValueError(
                    f"[decode] Unused tail in res.X for run {i}: used {ofs}, total {x.size}"
                )

            decoded_all.append(decoded_i)
            meta_all.append(meta_i)

        # --- 2. decide whether variable sets are aligned across experiments ---
        def _keys_from_meta(meta_dict: Dict[str, Dict[str, Any]]) -> List[tuple]:
            keys: List[tuple] = []
            for name, info in meta_dict.items():
                keys.append((name, info["timeseries"], info["items"]))
            return keys

        keys0 = _keys_from_meta(meta_all[0])
        aligned = True
        for j in range(1, n_res):
            keysj = _keys_from_meta(meta_all[j])
            if len(keysj) != len(keys0) or set(keysj) != set(keys0):
                aligned = False
                break

        saved_paths: List[str] = []

        # ------------------------------------------------------------------
        # CASE 1: aligned variable sets
        # ------------------------------------------------------------------
        if aligned:
            ts_keys = [k for k in keys0 if k[1]]
            topo_keys = [k for k in keys0 if not k[1]]

            # ---- 1A. time-series: ONE FIGURE PER VARIABLE ----------------
            for (var_name, _is_ts, items_tuple) in ts_keys:
                items_list = list(items_tuple)
                unit = units.get(var_name)

                # collect group_entries across all runs
                group_entries: List[Dict[str, Any]] = []
                for run_idx in range(n_res):
                    arr = decoded_all[run_idx][var_name]
                    info = meta_all[run_idx][var_name]
                    group_entries.append(
                        {
                            "run_idx": run_idx,
                            "name": var_name,
                            "values": arr,
                            "meta": info,
                        }
                    )

                fname = f"{prefix}__{var_name}__timeseries.png"
                fpath = self._plot_timeseries_multi_runs(
                    var_name, group_entries, items_list, labels, unit, fname
                )
                saved_paths.append(fpath)

            # ---- 1B. topology: ONE big grid figure -----------------------
            if topo_keys:
                n_topo_vars = len(topo_keys)
                n_plots = n_topo_vars * n_res
                ncols = 1 if n_plots == 1 else 2
                nrows = math.ceil(n_plots / ncols)

                fig, axes = plt.subplots(
                    nrows, ncols,
                    figsize=(5 * ncols, 4 * nrows)
                )
                axes = np.atleast_1d(axes).flatten()

                plot_idx = 0
                for (var_name, _is_ts, items_tuple) in topo_keys:
                    items = list(items_tuple)
                    unit = units.get(var_name)

                    for run_idx in range(n_res):
                        ax = axes[plot_idx]
                        plot_idx += 1

                        vals = np.asarray(decoded_all[run_idx][var_name], float).ravel()
                        info = meta_all[run_idx][var_name]
                        node_flags = info["node_flags"]
                        link_flags = info["link_flags"]

                        all_nodes, all_links = self._check_topology_items(
                            var_name, items, node_flags, link_flags
                        )
                        attr = {nm: float(v) for nm, v in zip(items, vals)}

                        if all_nodes:
                            title = f"{self._display_name(var_name)} (nodes) - {self._shorten_text(labels[run_idx], 24)}"
                            if unit:
                                title += f" [{unit}]"
                            wntr.graphics.plot_network(
                                self.wn,
                                node_attribute=attr,
                                title=title,
                                ax=ax,
                                **self._topology_plot_kwargs(),
                            )
                        elif all_links:
                            title = f"{self._display_name(var_name)} (links) - {self._shorten_text(labels[run_idx], 24)}"
                            if unit:
                                title += f" [{unit}]"
                            wntr.graphics.plot_network(
                                self.wn,
                                link_attribute=attr,
                                title=title,
                                ax=ax,
                                **self._topology_plot_kwargs(),
                            )

                for j in range(plot_idx, len(axes)):
                    fig.delaxes(axes[j])

                plt.tight_layout()
                fpath = self._save_fig(fig, f"{prefix}__topology_grid.png")
                saved_paths.append(fpath)

            return saved_paths

        # ------------------------------------------------------------------
        # CASE 2: non-aligned variable sets  per (run, var) figure
        # ------------------------------------------------------------------
        for run_idx in range(n_res):
            label = labels[run_idx]
            dec_i = decoded_all[run_idx]
            meta_i = meta_all[run_idx]

            for var_name, arr in dec_i.items():
                info = meta_i[var_name]
                items = list(info["items"])
                is_ts = info["timeseries"]
                node_flags = info["node_flags"]
                link_flags = info["link_flags"]
                unit = units.get(var_name)

                if is_ts:
                    T_local, _ = arr.shape
                    xi, xlab = self._build_time_axis(T_local)

                    fig, ax = plt.subplots(figsize=(7, 4))
                    for j, item_id in enumerate(items):
                        y = arr[:, j]
                        ax.plot(xi, y, label=str(item_id))

                    title = f"{self._axis_label(var_name, unit)} - {self._shorten_text(label, 24)}"
                    ax.set_title(title)
                    ax.set_xlabel(xlab)
                    ax.set_ylabel(self._axis_label(var_name, unit))
                    ax.legend(loc="best", frameon=True, framealpha=0.75, fontsize=7)
                    plt.tight_layout()

                    fname = f"{prefix}__{var_name}__{label}__timeseries.png"
                    fpath = self._save_fig(fig, fname)
                    saved_paths.append(fpath)
                else:
                    vals = np.asarray(arr, float).ravel()
                    all_nodes, all_links = self._check_topology_items(
                        var_name, items, node_flags, link_flags
                    )
                    fig, ax = plt.subplots(figsize=(6, 5))
                    attr = {nm: float(v) for nm, v in zip(items, vals)}

                    if all_nodes:
                        title = f"{self._display_name(var_name)} (nodes) - {self._shorten_text(label, 24)}"
                        if unit:
                            title += f" [{unit}]"
                        wntr.graphics.plot_network(
                            self.wn,
                            node_attribute=attr,
                            title=title,
                            ax=ax,
                            **self._topology_plot_kwargs(),
                        )
                    elif all_links:
                        title = f"{self._display_name(var_name)} (links) - {self._shorten_text(label, 24)}"
                        if unit:
                            title += f" [{unit}]"
                        wntr.graphics.plot_network(
                            self.wn,
                            link_attribute=attr,
                            title=title,
                            ax=ax,
                            **self._topology_plot_kwargs(),
                        )

                    plt.tight_layout()
                    fname = f"{prefix}__{var_name}__{label}__topology.png"
                    fpath = self._save_fig(fig, fname)
                    saved_paths.append(fpath)

        return saved_paths

    # ------------------------------------------------------------------
    # 3) Decision solution (formerly plot_decision_solution)
    # ------------------------------------------------------------------
    def plot_decision_solution(
        self,
        res: Any,
        varspecs: Sequence[Any],
        solution_index: Optional[int] = None,
        solution_F: Optional[Sequence[float]] = None,
        prefix: str = "decision",
    ) -> List[str]:
        """
        Corresponds to the original `plot_decision_solution`.
        """
        if self.wn is None:
            raise ValueError("WaterNetworkModel (wn) must be set on the plotter.")

        self._ensure_dir()
        units = self.units

        if not hasattr(res, "X"):
            raise ValueError("Result has no attribute 'X'")

        X_all = np.asarray(res.X, dtype=float)  # type: ignore[attr-defined]
        if X_all.ndim == 1:
            X_all = X_all.reshape(1, -1)
        n_solutions, _ = X_all.shape

        # Choose solution index
        if solution_index is not None:
            if not (0 <= solution_index < n_solutions):
                raise ValueError(
                    f"solution_index {solution_index} out of range [0, {n_solutions-1}]"
                )
            idx = int(solution_index)
        elif solution_F is not None:
            if not hasattr(res, "F"):
                raise ValueError("Result has no attribute 'F'; cannot match by solution_F")
            F_all = np.atleast_2d(np.asarray(res.F, dtype=float))  # type: ignore[attr-defined]
            target = np.asarray(solution_F, dtype=float).ravel()
            if F_all.shape[1] != target.size:
                raise ValueError(
                    f"solution_F dimension {target.size} != res.F.shape[1] {F_all.shape[1]}"
                )
            dists = np.linalg.norm(F_all - target, axis=1)
            idx = int(np.argmin(dists))
            print(
                f"[plot_decision_solution] Matched solution index {idx} "
                f"with F={F_all[idx]} (target={target})"
            )
        else:
            if n_solutions == 1:
                idx = 0
            else:
                raise ValueError(
                    f"Result has {n_solutions} solutions in X; please specify "
                    "`solution_index` or `solution_F`."
                )

        x = X_all[idx].ravel()
        print(f"[plot_decision_solution] Using solution index {idx}, x-len={len(x)}")

        T = self._get_T()

        # Decode this x into variables
        decoded: Dict[str, np.ndarray] = {}
        meta: Dict[str, Dict[str, Any]] = {}

        ofs = 0
        for spec in varspecs:
            items = list(spec.items)
            n_items = len(items)
            is_ts = bool(getattr(spec, "timeseries", False))

            need = n_items * T if is_ts else n_items
            sl = x[ofs:ofs + need]
            if sl.size != need:
                raise ValueError(
                    f"[decode] {spec.name}: expect {need}, got {sl.size} "
                    f"(solution index {idx})"
                )
            ofs += need

            name_lower = str(spec.name).lower()
            setter = getattr(spec, "setter", None)
            if "mask" in name_lower or setter == "random_selection_mask":
                continue

            if is_ts:
                arr = sl.reshape(T, n_items)
            else:
                arr = sl.reshape(n_items,)

            decoded[spec.name] = arr

            node_flags = [self._is_node(nm) for nm in items]
            link_flags = [self._is_link(nm) for nm in items]
            meta[spec.name] = {
                "items": items,
                "timeseries": is_ts,
                "node_flags": node_flags,
                "link_flags": link_flags,
            }

        if ofs != x.size:
            raise ValueError(
                f"[decode] Unused tail in x for solution index {idx}: "
                f"used {ofs}, total {x.size}"
            )

        # Plot each variable
        saved_paths: List[str] = []

        for var_name, arr in decoded.items():
            info = meta[var_name]
            items = info["items"]
            is_ts = info["timeseries"]
            node_flags = info["node_flags"]
            link_flags = info["link_flags"]
            unit = units.get(var_name, None)

            if is_ts:
                fname = f"{prefix}__{var_name}__sol{idx}__timeseries.png"
                fpath = self._plot_timeseries_single_solution(
                    var_name, arr, items, unit, idx, fname
                )
                saved_paths.append(fpath)
                continue

            fname = f"{prefix}__{var_name}__sol{idx}__topology.png"
            vals = np.asarray(arr, float).ravel()
            fpath = self._plot_topology_single(
                var_name, items, vals, node_flags, link_flags,
                unit, title_suffix=f"  [solution #{idx}]",
                fname=fname,
            )
            saved_paths.append(fpath)

        return saved_paths


