import warnings
import wntr
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import List, Dict, Optional
from pymoo.core.problem import ElementwiseProblem
from tools.decision_variables import VarSpec
from tools.help_functions import _pattern_len
from tools.objective_functions import ObjectiveContext, OBJECTIVE_REGISTRY

DEFAULT_MINIMUM_SERVICE_PRESSURE_M = 20.0

warnings.filterwarnings(
    "ignore",
    message=r'Not all curves were used in ".*"; added with type None, units conversion left to user',
    category=UserWarning,
    module=r"wntr\.epanet\.io",
)


# =========================
# Generic composable problem
# =========================
class FlexibleWDNProblem(ElementwiseProblem):
    """
        - Variables: `VarSpec`, with an optional `timeseries=True` -> dimension = T * n_items
                     (where T is determined by duration / pattern_timestep)
        - Objectives: selected from `OBJECTIVE_REGISTRY`; current workflow
                      objectives are pump_energy and modified_resilience_index
        - Constraints: hard minimum pressure over all junctions
    """

    def __init__(self,
                 inp_path: str,
                 var_specs: List[VarSpec],
                 objectives: List[str],
                 detection_limit: float,
                 objective_weights: Optional[List[float]] = None,
                 observed: Optional[Dict[str, pd.DataFrame]] = None,
                 price_pattern: Optional[pd.DataFrame] = None,
                 selected_names: Optional[Dict[str, List[str]]] = None,
                 use_epanet_toolkit: bool = True,
                 pressure_min_constraint: Optional[float] = None,
                 minimum_service_pressure: Optional[float] = None,
                 demand_model: Optional[str] = "DD",
                 hydraulic_minimum_pressure: Optional[float] = 0.0,
                 simulation_duration_hours: Optional[float] = None,
                 pressure_constraint_nodes: str = "all",):

        self._base_wn = wntr.network.WaterNetworkModel(inp_path)
        if simulation_duration_hours is not None:
            self._base_wn.options.time.duration = float(simulation_duration_hours) * 3600.0
        self.use_epanet_toolkit = use_epanet_toolkit
        self.demand_model = demand_model
        self.requires_water_age = "water_quality_risk" in objectives
        self.price_pattern = price_pattern
        self.detection_limit = detection_limit
        self.Pmin_constr = 0.0 if pressure_min_constraint is None else float(pressure_min_constraint)
        self.hydraulic_minimum_pressure = (
            0.0 if hydraulic_minimum_pressure is None else float(hydraulic_minimum_pressure)
        )
        if pressure_constraint_nodes not in ("all", "demand"):
            raise ValueError("pressure_constraint_nodes must be 'all' or 'demand'")
        self.pressure_constraint_nodes = pressure_constraint_nodes
        self.minimum_service_pressure = (
            DEFAULT_MINIMUM_SERVICE_PRESSURE_M
            if minimum_service_pressure is None
            else float(minimum_service_pressure)
        )
        # EPANET enforces a lower bound of 0.1 for required_pressure.
        self.required_pressure = max(0.1, self.minimum_service_pressure)


        # Variable concatenation -> global x (supports timeseries dimension expansion)
        self.var_specs = var_specs
        self.groups: Dict[str, slice] = {}
        xl_list, xu_list, x0_parts = [], [], []
        start = 0

        T_global = _pattern_len(self._base_wn)  # Global T (number of pattern steps)
        for spec in var_specs:
            T = T_global if spec.timeseries else 1
            n_effect = spec.n * T
            self.groups[spec.name] = slice(start, start + n_effect)
            start += n_effect

            # bounds
            if isinstance(spec.bounds, tuple):
                lo, hi = float(spec.bounds[0]), float(spec.bounds[1])
                xl_list.append(np.full(n_effect, lo, float))
                xu_list.append(np.full(n_effect, hi, float))
            else:
                lo = np.array([b[0] for b in spec.bounds], float)
                hi = np.array([b[1] for b in spec.bounds], float)
                if T > 1:
                    lo = np.tile(lo, T)   # Order: expand step by step (one row per step)
                    hi = np.tile(hi, T)
                xl_list.append(lo); xu_list.append(hi)

            # prior: support (n_items,), (T*n_items,), (T,n_items)
            if spec.prior is not None:
                p = np.asarray(spec.prior, float)
                if spec.timeseries:
                    if p.size == spec.n:
                        p = np.tile(p, T)
                    elif p.size == spec.n * T:
                        p = p.reshape(-1)
                    elif p.ndim == 2 and p.shape == (T, spec.n):
                        p = p.reshape(-1)
                    else:
                        warnings.warn(f"[{spec.name}] prior shape mismatch, ignored")
                        p = None
                x0_parts.append(p) if p is not None else None

        self.n_var_total = start
        xl = np.concatenate(xl_list) if xl_list else np.zeros((0,), float)
        xu = np.concatenate(xu_list) if xu_list else np.zeros((0,), float)
        self.x0 = np.concatenate(x0_parts) if x0_parts else None

        # objective setting: single/multiple objective
        self.objectives = objectives
        self.objective_weights = objective_weights
        if objective_weights is None and len(objectives) > 1:
            n_obj = len(objectives); self._multi_obj = True
        else:
            self._multi_obj = False; n_obj = 1

        # oberved/choose
        self.observed = (observed or {}).copy()
        for k, df in list(self.observed.items()):
            self.observed[k] = df.sort_index()
        self.selected_names = selected_names or {}



        # Constraint: hard minimum pressure over all junctions.
        n_constr = 1 if (self.Pmin_constr is not None) else 0

        super().__init__(
            n_var=self.n_var_total,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=xl,
            xu=xu,
            elementwise_evaluation=True
        )

    # ---------- simulation ----------
    def _simulate(self, wn: wntr.network.WaterNetworkModel):
        wn.options.hydraulic.demand_model = self.demand_model
        try:
            wn.options.hydraulic.minimum_pressure = float(self.hydraulic_minimum_pressure)
        except Exception:
            pass
        wn.options.hydraulic.required_pressure = self.required_pressure
        if self.requires_water_age:
            wn.options.quality.parameter = "AGE"
        sim = wntr.sim.EpanetSimulator(wn) if self.use_epanet_toolkit else wntr.sim.WNTRSimulator(wn)
        try:
            return sim.run_sim()
        except Exception as e:
            warnings.warn(f"Simulation failed: {e}")
            return None

    # ---------- apple decision variables ----------
    def _apply_decision(self, wn: wntr.network.WaterNetworkModel, x: np.ndarray):
        T = _pattern_len(wn)
        for spec in self.var_specs:
            sl = self.groups[spec.name]
            xseg = np.asarray(x[sl], float)
            if spec.timeseries:
                # Restore to shape (T, n_items) by reversing the step-by-step expansion
                vals = xseg.reshape(T, spec.n)
            else:
                vals = xseg
            spec.setter(wn, spec.items, vals)

    # ---------- calculate objective functions ----------
    def _calc_objectives(self, results, x: np.ndarray, wn: wntr.network.WaterNetworkModel):
        ctx = ObjectiveContext(
            wn=wn,
            sim=results,
            observed=self.observed,
            price=self.price_pattern,
            selected_names=self.selected_names,
            x=x,
            x0=self.x0,
            groups=self.groups,
            params={
                "pressure_violation": {"min_pressure": self.Pmin_constr},
                "service_pressure": {
                    "minimum_service_pressure": self.minimum_service_pressure
                } if self.minimum_service_pressure is not None else {},
                "Pstar": self.required_pressure,
                "water_security":{"detection_limit": self.detection_limit} if self.detection_limit is not None else {},
                "pressure_objective": {"nodes": self.pressure_constraint_nodes},
            }
        )
        vals = []
        for name in self.objectives:
            fn = OBJECTIVE_REGISTRY.get(name, None)
            if fn is None:
                raise ValueError(f"Unknown objective: {name}")
            vals.append(fn(ctx))
        if self._multi_obj:
            return np.array(vals, float)
        else:
            w = np.array(self.objective_weights if self.objective_weights is not None else [1.0]*len(vals), float)
            return float(np.sum(w * np.array(vals, float)))

    # ---------- Constraint: hard minimum pressure over all junctions ----------
    def _calc_constraints(self, results):
        if self.Pmin_constr is None:
            return np.zeros((0,), dtype=float)

        if results is None or "pressure" not in results.node:
            return np.array([1.0], float)

        # simP: rows = time steps, columns = node names
        simP = results.node["pressure"]

        try:
            junctions = list(self._base_wn.junction_name_list)
        except Exception:
            junctions = []

        if self.pressure_constraint_nodes == "demand":
            try:
                expected = wntr.metrics.hydraulic.expected_demand(self._base_wn)
                active = expected.max(axis=0) > 1e-12
                junctions = [name for name in junctions if name in active.index and bool(active.loc[name])]
            except Exception:
                pass

        cols = [c for c in junctions if c in simP.columns]
        if not cols:
            return np.array([1.0], float)

        deficit = (float(self.Pmin_constr) - simP[cols]).clip(lower=0.0)
        g = float(np.nanmax(deficit.values))
        return np.array([g], dtype=float)

    # ---------- pymoo hooks (callback mechanism for monitoring/modifying the optimization process) ----------
    def _evaluate(self, x, out, *args, **kwargs):
        wn = deepcopy(self._base_wn)
        self._apply_decision(wn, x)
        results = self._simulate(wn)
        if results is None:
            out["F"] = np.array([1e12]*len(self.objectives), float) if self._multi_obj else 1e12
            out["G"] = np.array([1.0], float) if self.Pmin_constr is not None else np.zeros((0,), float)
            return
        out["F"] = self._calc_objectives(results, x, wn)
        out["G"] = self._calc_constraints(results)
