import wntr
import numpy as np
from typing import List, Callable, Optional, Tuple, Sequence, Dict
from tools.help_functions import _pattern_len, _fit_to_len, _opt_ctx
from wntr.network import controls
import warnings

# =========================
# Variable Registry (Extensible)
# =========================

class VarSpec:
    """
    Define a family of variables (e.g., a set of pipe roughness coefficients, pump speeds, or valve settings).

    - name: The name of the variable family (e.g., 'pipe_roughness')
    - items: A list of entity names to be optimized within this family (e.g., a list of pipe IDs)
    - bounds: (lo, hi) or a list of (lo_i, hi_i) with the same length as `items`
    - setter: A callback function setter(wn, items, vals_decoded) -> None
              * If timeseries=False: vals_decoded has shape (n_items,)
              * If timeseries=True : vals_decoded has shape (T, n_items)
    - prior: (Optional) An array of prior values, used for the objective 'prior_deviation'
              * If timeseries=True, prior can be of shape (n_items,), (T * n_items,), or (T, n_items)
    - timeseries: Whether to model this variable family as a time series;
                  True -> the decision space dimension becomes T * n_items
    """

    def __init__(self,
                 name: str,
                 items: List[str],
                 bounds: Tuple[float, float] | List[Tuple[float, float]],
                 setter: Callable[[wntr.network.WaterNetworkModel, List[str], np.ndarray], None],
                 prior: Optional[np.ndarray] = None,
                 timeseries: bool = False):
        self.name = name
        self.items = items
        self.n = len(items)
        self.bounds = bounds
        self.setter = setter
        self.prior = prior
        self.timeseries = timeseries


# --------- Several generic setter implementations ---------

# Random select nodes/edges
def make_setter_random_selection_mask(group_name: str, choose_rate: float, *, candidate: Optional[Sequence[str]] = None, at_least_one: bool = True):
    """
    Returns a setter(wn, items, vals) that uses vals (in 0..1) as selection scores.
    Select the Top-k items by choose_rate, create a boolean mask, and store it in
    wn._opt_ctx[f"{group_name}:selected"]. No randomness is used.
    choose_rate in [0, 1]: select k = round(choose_rate * n) items; if at_least_one=True, ensure at least 1 item.
    vals length must match items; values are clipped to [0,1], and NaN is treated as 0.
    """

    if not (0.0 <= choose_rate <= 1.0):
        raise ValueError(f"[{group_name}] choose_rate must be within [0, 1]")

    key = f"{group_name}:selected"

    def _setter(wn: wntr.network.WaterNetworkModel, items: List[str], vals: np.ndarray):

        if candidate:
            cand_set = set(candidate)
            selected = np.array([name in cand_set for name in items], dtype=bool)
            _opt_ctx(wn)[key] = selected
            return

        n = len(items)
        arr = np.asarray(vals, float).ravel()

        if len(arr) != n:
            raise ValueError(f"[{group_name}] vals length ({len(arr)}) != items length ({n})")

        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 1.0)

        if choose_rate <= 0.0 or n == 0:
            k = 0
        elif choose_rate >= 1.0:
            k = n
        else:
            k = int(round(choose_rate * n))

        if at_least_one and n > 0:
            k = max(k, 1)

        if k == 0:
            selected = np.zeros(n, dtype=bool)
        elif k >= n:
            selected = np.ones(n, dtype=bool)
        else:
            order = np.argsort(-arr, kind="mergesort")
            idx = order[:k]
            selected = np.zeros(n, dtype=bool)
            selected[idx] = True

        _opt_ctx(wn)[key] = selected

    return _setter


# Pipe roughness (scalar vector)
def make_setter_pipe_roughness_masked(group_name: str = 'all'):
    key = f"{group_name}:selected"

    def _setter(wn: wntr.network.WaterNetworkModel, items: List[str], vals: np.ndarray):
        arr = np.asarray(vals, float).ravel()
        if len(arr) != len(items):
            raise ValueError(f"[{group_name}] roughness length != items")
        selected = _opt_ctx(wn).get(key, np.ones(len(items), dtype=bool))
        for i, (nm, v) in enumerate(zip(items, arr)):
            if selected[i]:
                wn.get_link(nm).roughness = float(v)

    return _setter

# Pipe diameter (scalar vector)
def make_setter_pipe_diameter_masked(group_name: str = 'all'):
    key = f"{group_name}:selected"

    def _setter(wn: wntr.network.WaterNetworkModel, items: List[str], vals: np.ndarray):
        arr = np.asarray(vals, float).ravel()
        step = 0.05
        arr = np.round(arr / step) * step
        arr = np.maximum(arr, step)
        if len(arr) != len(items):
            raise ValueError(f"[{group_name}] diameter length != items")
        selected = _opt_ctx(wn).get(key, np.ones(len(items), dtype=bool))
        for i, (nm, v) in enumerate(zip(items, arr)):
            if selected[i]:
                wn.get_link(nm).diameter = float(v)

    return _setter


# Node emitter coefficients (scalar vector)
def make_setter_node_emitter_masked(group_name: str = 'all'):
    key = f"{group_name}:selected"

    def _setter(wn: wntr.network.WaterNetworkModel, items: List[str], vals: np.ndarray):
        arr = np.asarray(vals, float).ravel()
        if len(arr) != len(items):
            raise ValueError(f"[{group_name}] emitter length != items")
        selected = _opt_ctx(wn).get(key, np.ones(len(items), dtype=bool))
        for i, (nm, v) in enumerate(zip(items, arr)):
            if selected[i]:
                wn.get_node(nm).emitter_coefficient = float(v)

    return _setter

# Global demand multiplier (scalar: items = ["global"])
def make_setter_demand_multiplier_masked(group_name: str = 'all', *, mode: str = "relative"):
    """
    mode:
      - "relative": j.base_demand *= v
      - "absolute": j.base_demand  = v
    """
    key = f"{group_name}:selected"
    if mode not in ("relative", "absolute"):
        raise ValueError("mode must be 'relative' or 'absolute'")

    def _setter(wn: wntr.network.WaterNetworkModel, items: List[str], vals: np.ndarray):
        arr = np.asarray(vals, float).ravel()
        if len(arr) != len(items):
            raise ValueError(f"[{group_name}] demand values length != items")
        selected = _opt_ctx(wn).get(key, np.ones(len(items), dtype=bool))
        for i, (nm, v) in enumerate(zip(items, arr)):
            if not selected[i]:
                continue
            j = wn.get_node(nm)

            if getattr(j, "base_demand", None) is None:
                continue
            if mode == "relative":
                j.demand_timeseries_list[0].base_value = float(j.base_demand) * float(v)
            else:  # absolute
                j.demand_timeseries_list[0].base_value = float(v)

    return _setter

# Initial quality
def make_setter_junction_initial_quality_masked(group_name: str = 'all'):
    key = f"{group_name}:selected"
    def _setter(wn, items, vals):
        arr = np.asarray(vals, float).ravel()
        if len(arr) != len(items):
            raise ValueError(f"[{group_name}] quality length != items")
        selected = _opt_ctx(wn).get(key, np.ones(len(items), dtype=bool))
        for i, (nm, v) in enumerate(zip(items, arr)):
            if selected[i]:
                wn.get_node(nm).initial_quality = float(v)
    return _setter



# source type
def make_setter_randomize_existing_source_types(group_name: str = 'all'):
    """
    Returns a setter(wn, items, vals):
      - Iterates over `wn.source_name_list` and randomly assigns a source type
        to each existing Source from the set {CONCEN, MASS, FLOWPACED, SETPOINT}.
    """


    TYPES = ("CONCEN", "MASS", "FLOWPACED", "SETPOINT")
    key = f"{group_name}:selected"

    def _setter(wn: wntr.network.WaterNetworkModel, items: List[str], vals: np.ndarray) -> None:

        names = list(items) if items else list(getattr(wn, "source_name_list", []))
        if not names:
            raise ValueError("Network has no sources to set types for.")
        selected = _opt_ctx(wn).get(key, np.ones(len(items), dtype=bool))
        names = names[selected]

        arr = np.asarray(vals, float).ravel()
        if len(arr) != len(names):
            raise ValueError(f"vals length ({len(arr)}) != number of sources ({len(names)})")

        arr = np.nan_to_num(arr, nan=0.0)
        idx = np.rint(arr).astype(int)
        idx = np.clip(idx, 0, len(TYPES) - 1)

        for sname, i in zip(names, idx):
            src = wn.get_source(sname)
            new_type = TYPES[i]
            old_type = (src.source_type or "").upper()
            if new_type != old_type:
                src.source_type = new_type

    return _setter


# water quality strenght timeseries
def make_setter_source_strength_timeseries(group_name: str = 'all', *, default_type: str = "CONCEN"):
    """
        Returns a setter(wn, items, vals):
          - vals: A 2D array of shape (T, n_items) representing the absolute intensity time series
                  (values may include 0).
          - For each selected node:
              * If no source exists -> create one with type = default_type
              * If a source already exists -> keep its original type unchanged
              * Write or update the pattern: set base_value = 1.0 and pattern = time series
                (automatically aligned to simulation time steps)
    """

    key_mask = f"{group_name}:selected"

    def _setter(wn: wntr.network.WaterNetworkModel, items: List[str], vals: np.ndarray):
        vals = np.asarray(vals, float)
        if vals.ndim != 2 or vals.shape[1] != len(items):
            raise ValueError(f"[{group_name}] strength timeseries must be (T, n_items)")

        selected = _opt_ctx(wn).get(key_mask, np.ones(len(items), dtype=bool))

        nstep = int(wn.options.time.duration // wn.options.time.pattern_timestep) + 1
        T = vals.shape[0]
        if T < nstep:
            last = vals[-1:, :]
            vals = np.vstack([vals, np.repeat(last, nstep - T, axis=0)])
        elif T > nstep:
            vals = vals[:nstep, :]

        for j, node in enumerate(items):
            if not selected[j]:
                continue

            sname = f"SRC_{node}"
            if sname not in wn.source_name_list:
                wn.add_source(sname, node, default_type, 1.0, None)

            series = np.clip(vals[:, j], 0.0, None)
            pname  = f"src_pat_{node}"
            # write/updata pattern
            if pname in wn.pattern_name_list:
                wn.get_pattern(pname).multipliers = series.tolist()
            else:
                wn.add_pattern(pname, series.tolist())

            # Set base = 1.0 so that the intensity equals the pattern values
            src = wn.get_source(sname)
            src.strength_timeseries.base_value = 1.0
            src.strength_timeseries.pattern_name = pname

    return _setter


# Tank initial_level (scalar vector)
def make_setter_tank_initial_level_masked(group_name: str = 'all'):
    key = f"{group_name}:selected"

    def _setter(wn: wntr.network.WaterNetworkModel, items: List[str], vals: np.ndarray):
        names = list(items) if items else list(getattr(wn, "tank_name_list", []))
        if not names:
            raise ValueError("Network has no tanks.")

        arr = np.asarray(vals, float).ravel()
        if len(arr) != len(items):
            raise ValueError(f"[{group_name}] initial level != items")
        selected = _opt_ctx(wn).get(key, np.ones(len(items), dtype=bool))
        for i, (nm, v) in enumerate(zip(items, arr)):
            if selected[i]:
                wn.get_node(nm).initial_level = float(v)

    return _setter

# Tank max_level (scalar vector)
def make_setter_tank_max_level_masked(group_name: str = 'all'):
    key = f"{group_name}:selected"

    def _setter(wn: wntr.network.WaterNetworkModel, items: List[str], vals: np.ndarray):
        names = list(items) if items else list(getattr(wn, "tank_name_list", []))
        if not names:
            raise ValueError("Network has no tanks.")

        arr = np.asarray(vals, float).ravel()
        if len(arr) != len(items):
            raise ValueError(f"[{group_name}] max level != items")
        selected = _opt_ctx(wn).get(key, np.ones(len(items), dtype=bool))
        for i, (nm, v) in enumerate(zip(items, arr)):
            if selected[i]:
                wn.get_node(nm).max_level = float(v)

    return _setter

# Tank min_level (scalar vector)
def make_setter_tank_min_level_masked(group_name: str = 'all'):
    key = f"{group_name}:selected"

    def _setter(wn: wntr.network.WaterNetworkModel, items: List[str], vals: np.ndarray):
        names = list(items) if items else list(getattr(wn, "tank_name_list", []))
        if not names:
            raise ValueError("Network has no tanks.")

        arr = np.asarray(vals, float).ravel()
        if len(arr) != len(items):
            raise ValueError(f"[{group_name}] min level != items")
        selected = _opt_ctx(wn).get(key, np.ones(len(items), dtype=bool))
        for i, (nm, v) in enumerate(zip(items, arr)):
            if selected[i]:
                wn.get_node(nm).min_level = float(v)

    return _setter

# Pipe status (scalar vector)
def make_setter_pipe_status_masked(group_name: str = 'all'):
    key = f"{group_name}:selected"

    def _setter(wn: wntr.network.WaterNetworkModel, items: List[str], vals: np.ndarray):

        arr = np.asarray(vals, float).ravel()
        if len(arr) != len(items):
            raise ValueError(f"[{group_name}] pipe status != items")
        selected = _opt_ctx(wn).get(key, np.ones(len(items), dtype=bool))
        for i, (nm, v) in enumerate(zip(items, arr)):
            if selected[i]:
                wn.get_link(nm).initial_status = wntr.network.elements.LinkStatus.Open if v >= 0.5 else wntr.network.elements.LinkStatus.Closed

    return _setter



# Valve status (scalar vector)
def make_setter_valve_status_masked(group_name: str = 'all'):
    key = f"{group_name}:selected"

    def _setter(wn: wntr.network.WaterNetworkModel, items: List[str], vals: np.ndarray):

        names = list(items) if items else list(getattr(wn, "valve_name_list", []))
        if not names:
            raise ValueError("Network has no valves.")

        arr = np.asarray(vals, float).ravel()
        if len(arr) != len(items):
            raise ValueError(f"[{group_name}] valve status != items")
        selected = _opt_ctx(wn).get(key, np.ones(len(items), dtype=bool))
        for i, (nm, v) in enumerate(zip(items, arr)):
            if selected[i]:
                wn.get_link(nm).initial_status = wntr.network.elements.LinkStatus.Open if v >= 0.5 else wntr.network.elements.LinkStatus.Closed

    return _setter


# Choose best tank volume curves in the curve candidates (scalar vector)
def make_setter_tank_volume_curve(group_name: str = 'all'):

    key = f"{group_name}:selected"

    def _setter(wn, items: List[str], vals: np.ndarray):
        if not items:
            raise ValueError("No tank items provided.")

        selected = np.asarray(_opt_ctx(wn).get(key, np.ones(len(items), dtype=bool)), dtype=bool)
        if selected.size != len(items):
            raise ValueError(f"[{group_name}] selection mask length ({selected.size}) != items length ({len(items)})")

        sel_idx = np.where(selected)[0]
        if sel_idx.size == 0:
            warnings.warn(f"No tanks selected in group '{group_name}'; nothing to do.")
            return


        V = np.asarray(vals, float).ravel()
        if V.size == len(items):
            V = V[selected]
        elif V.size == sel_idx.size:
            pass
        else:
            raise ValueError(
                f"vals length ({V.size}) must equal n_selected ({sel_idx.size}) or n_items ({len(items)})"
            )

        existing = set(getattr(wn, "curve_name_list", []))

        for Vmax, j in zip(V, sel_idx):
            nm = items[j]

            try:
                tank = wn.get_node(nm)
            except Exception:
                warnings.warn(f"Tank '{nm}' not found, skip.")
                continue

            H = float(getattr(tank, "max_level", 0.0)) + 0.3
            if not np.isfinite(H) or H <= 0.0:
                warnings.warn(f"Tank '{nm}' has non-positive height; use H=1.0.")
                H = 1.0

            if not np.isfinite(Vmax) or Vmax < 0:
                warnings.warn(f"Tank '{nm}' Vmax invalid ({Vmax}); set to 0.")
                Vmax = 0.0

            pts = [(0.0, 0.0), (float(H), float(Vmax))]

            cname = getattr(tank, "vol_curve_name", None)
            if cname and cname in existing:
                cv = wn.get_curve(cname)
                if hasattr(cv, "points"):
                    cv.points = pts
                else:
                    xs, ys = zip(*pts)
                    cv.curve_type = "VOLUME"
                    cv.x, cv.y = np.array(xs, float), np.array(ys, float)
            else:
                cname = f"VOL_{nm}"
                k = 1
                while cname in existing:
                    cname = f"VOL_{nm}_{k}"
                    k += 1
                wn.add_curve(cname, "VOLUME", pts)
                existing.add(cname)
                tank.vol_curve_name = cname

    return _setter



# Pump speeds (2D time series: shape = (T, n_pumps))
def make_setter_pump_speed_masked(group_name: str = 'all'):

    key = f"{group_name}:selected"

    def _setter(wn: wntr.network.WaterNetworkModel, items: List[str], vals: np.ndarray):
        """
        vals: 2D array of shape (T, n_pumps), representing the relative speed multiplier at each time step (recommended; used together with timeseries=True).
        Standard practice: set pump.base_speed = 1.0, and control the operating speed via a time pattern (speed_pattern_name = 'spd_<pump_name>').
        """
        if not items:
            raise ValueError("No pump items provided.")

        selected = np.asarray(_opt_ctx(wn).get(key, np.ones(len(items), dtype=bool)), dtype=bool)
        if selected.size != len(items):
            raise ValueError(f"[{group_name}] selection mask length ({selected.size}) != items length ({len(items)})")

        sel_idx = np.where(selected)[0]
        if sel_idx.size == 0:
            return

        vals = np.asarray(vals, float)
        if vals.ndim != 2:
            raise ValueError("pump speeds must be a 2D array of shape (T, n_pumps)")

        n_items = len(items)
        n_sel = sel_idx.size

        if vals.shape[1] == n_items:
            V = vals[:, selected]
        elif vals.shape[1] == n_sel:
            V = vals
        else:
            raise ValueError(
                f"vals shape mismatch: got {vals.shape}, expected (T,{n_items}) or (T,{n_sel})"
            )

        V = np.clip(V, 0.0, None)

        nstep = _pattern_len(wn)

        for j in sel_idx:
            pnm = items[j]
            pump = wn.get_link(pnm)
            pump.base_speed = 1.0

        for col_k, j in enumerate(sel_idx):
            pnm = items[j]
            series = V[:, col_k]
            mults = _fit_to_len(series, nstep)

            pname = f"spd_{pnm}"
            if pname in wn.pattern_name_list:
                wn.get_pattern(pname).multipliers = mults
            else:
                wn.add_pattern(pname, mults)
            wn.get_link(pnm).speed_pattern_name = pname

    return _setter

# reservor total heads (2D time series: shape = (T, n_reservors))
def make_setter_reservoir_head_masked(group_name: str = 'all'):

    key = f"{group_name}:selected"

    def _setter(wn, items, vals):
        """
        vals: 2D array of shape (T, n_reservoirs), representing the relative head multiplier at each time step (recommended; used together with timeseries=True).
        Standard practice: set reservoirs.base_head = 1.0, and control the operating head via a time pattern (speed_pattern_name = 'hpat_<reservoir_name>').
        """

        if not items:
            raise ValueError("No reservoir items provided.")

        selected = np.asarray(_opt_ctx(wn).get(key, np.ones(len(items), dtype=bool)), dtype=bool)
        if selected.size != len(items):
            raise ValueError(f"[{group_name}] selection mask length ({selected.size}) != items length ({len(items)})")

        sel_idx = np.where(selected)[0]
        if sel_idx.size == 0:
            return

        vals = np.asarray(vals, float)
        if vals.ndim != 2:
            raise ValueError("pump speeds must be a 2D array of shape (T, n_pumps)")

        n_items = len(items)
        n_sel = sel_idx.size

        if vals.shape[1] == n_items:
            V = vals[:, selected]
        elif vals.shape[1] == n_sel:
            V = vals
        else:
            raise ValueError(
                f"vals shape mismatch: got {vals.shape}, expected (T,{n_items}) or (T,{n_sel})"
            )

        V = np.clip(V, 0.0, None)

        nstep = _pattern_len(wn)

        for j in sel_idx:
            pnm = items[j]
            pump = wn.get_node(pnm)
            pump.base_head = 1.0

        for col_k, j in enumerate(sel_idx):
            pnm = items[j]
            series = V[:, col_k]
            mults = _fit_to_len(series, nstep)

            pname = f"hpat_{pnm}"
            if pname in wn.pattern_name_list:
                wn.get_pattern(pname).multipliers = mults
            else:
                wn.add_pattern(pname, mults)
            wn.get_node(pnm).head_pattern_name = pname

    return _setter


# Valve settings (constant or 2D time series; time-series control is applied via TimeControl updating the setting at each step)
def make_setter_valve_setting_masked(group_name: str = 'all'):

    key = f"{group_name}:selected"

    def _setter(wn: wntr.network.WaterNetworkModel, items: List[str], vals: np.ndarray):
        """
            vals:
              - 1D: A constant setting value for each valve.
              - 2D: An array of shape (T, n_valves), specifying the setting value at each time step
                    (used together with timeseries=True).
            Implementation: At the beginning of each simulation step, the valve setting is updated to the corresponding value using a combination of `Rule` and `SimTimeCondition`.
            Note: During `_evaluate`, the base network is deep-copied at each iteration, so manual cleanup of previous controls is not required.
        """

        selected = np.asarray(_opt_ctx(wn).get(key, np.ones(len(items), dtype=bool)), dtype=bool)
        if selected.size != len(items):
            raise ValueError(
                f"[{group_name}] selection mask length ({selected.size}) != items length ({len(items)})"
            )
        sel_idx = np.where(selected)[0]
        if sel_idx.size == 0:
            return

        vals = np.asarray(vals, float)
        n_items = len(items)
        n_sel = sel_idx.size

        if vals.ndim == 1:
            if vals.size == n_items:
                const = vals[selected]
            elif vals.size == n_sel:
                const = vals
            else:
                raise ValueError(
                    f"Constant settings length mismatch: got {vals.size}, expected {n_items} or {n_sel}"
                )
            for k, j in enumerate(sel_idx):
                vnm = items[j]
                wn.get_link(vnm).setting = float(const[k])
            return

        if vals.ndim != 2:
            raise ValueError("vals must be 1D (constant) or 2D (T, n_valves)")


        if vals.shape[1] == n_items:
            series_all = vals[:, selected]
        elif vals.shape[1] == n_sel:
            series_all = vals
        else:
            raise ValueError(
                f"Time-series shape mismatch: got {vals.shape}, expected (T,{n_items}) or (T,{n_sel})"
            )

        pstep_sec = int(wn.options.time.pattern_timestep)
        if pstep_sec <= 0:
            raise ValueError("pattern_timestep must be positive.")
        nstep = int(wn.options.time.duration // pstep_sec) + 1

        T = series_all.shape[0]
        if T < nstep:
            pad = np.repeat(series_all[-1:, :], nstep - T, axis=0)
            series_all = np.vstack([series_all, pad])
        elif T > nstep:
            series_all = series_all[:nstep, :]

        for col_k, j in enumerate(sel_idx):
            vnm = items[j]
            link = wn.get_link(vnm)
            series = series_all[:, col_k]
            for k, setpoint in enumerate(series):
                t_hours = (k * pstep_sec) / 3600.0
                cond = controls.SimTimeCondition(model=wn, relation="after", threshold=t_hours)
                act = controls.ControlAction(link, "setting", float(setpoint))
                rule = controls.Rule(condition=cond, then_actions=act)
                wn.add_control(f"set_{group_name}_{vnm}_{k}", rule)

    return _setter



