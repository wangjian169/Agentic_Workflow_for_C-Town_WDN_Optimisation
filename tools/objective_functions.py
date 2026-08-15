import numpy as np
import pandas as pd
import wntr
import scipy
from typing import List, Dict, Callable, Optional
from tools.help_functions import _mean_or_inf



# =========================
# Objective functions registry (extensible)
# =========================
class ObjectiveContext:
    def __init__(self, wn, sim, observed: Dict[str, pd.DataFrame],
                 selected_names: Dict[str, List[str]],
                 price: Optional[pd.DataFrame],
                 x: np.ndarray, x0: Optional[np.ndarray],
                 groups: Dict[str, slice],
                 params: Dict[str, dict]):
        self.wn = wn
        self.sim = sim
        self.observed = observed
        self.selected_names = selected_names
        self.x = x
        self.x0 = x0
        self.groups = groups
        self.params = params
        self.price = price


def obj_pressure_rmse(ctx: ObjectiveContext) -> float:
    if "pressure" not in ctx.sim.node:
        return np.inf
    sim_df = ctx.sim.node["pressure"]
    obs = ctx.observed.get("pressure", None)
    if obs is None: return np.inf

    # Column alignment: if nodes are not specified, use the common columns of both
    nodes_sel = ctx.selected_names.get("pressure_nodes", None)
    cols = nodes_sel if nodes_sel else sim_df.columns.intersection(obs.columns)
    if len(cols) == 0: return np.inf
    common_idx = sim_df.index.intersection(obs.index)
    if len(common_idx) == 0: return np.inf
    A = sim_df.loc[common_idx, cols]
    B = obs.loc[common_idx, cols]
    mse = np.mean((A.values - B.values) ** 2)
    return float(np.sqrt(mse))

def obj_head_rmse(ctx: ObjectiveContext) -> float:
    if "head" not in ctx.sim.node:
        return np.inf
    sim_df = ctx.sim.node["head"]
    obs = ctx.observed.get("head", None)
    if obs is None: return np.inf

    # Column alignment: if nodes are not specified, use the common columns of both
    nodes_sel = ctx.selected_names.get("head_nodes", None)
    cols = nodes_sel if nodes_sel else sim_df.columns.intersection(obs.columns)
    if len(cols) == 0: return np.inf
    common_idx = sim_df.index.intersection(obs.index)
    if len(common_idx) == 0: return np.inf
    A = sim_df.loc[common_idx, cols]
    B = obs.loc[common_idx, cols]
    mse = np.mean((A.values - B.values) ** 2)
    return float(np.sqrt(mse))

def obj_flow_rmse(ctx: ObjectiveContext) -> float:
    if "flowrate" not in ctx.sim.link:
        return np.inf
    sim_df = ctx.sim.link["flowrate"]
    obs = ctx.observed.get("flow", None)
    if obs is None: return np.inf

    # Column alignment: if links are not specified, use the common columns of both
    links_sel = ctx.selected_names.get("flow_links", None)
    cols = links_sel if links_sel else sim_df.columns.intersection(obs.columns)
    if len(cols) == 0: return np.inf
    common_idx = sim_df.index.intersection(obs.index)
    if len(common_idx) == 0: return np.inf
    A = sim_df.loc[common_idx, cols]
    B = obs.loc[common_idx, cols]
    mse = np.mean((A.values - B.values) ** 2)
    return float(np.sqrt(mse))

def obj_pump_switches(ctx: ObjectiveContext) -> float:
    if "status" not in ctx.sim.link:
        return np.inf
    st = ctx.sim.link["status"]
    pumps = ctx.wn.pump_name_list
    if len(pumps) == 0:
        raise ValueError("This network contains no pumps (pump_name_list is empty).")
    diffs = (st[pumps].diff().abs() > 0.5).sum().sum()
    return float(diffs)

def obj_prior_deviation(ctx: ObjectiveContext) -> float:
    if ctx.x0 is None:
        return 0.0
    return float(np.sum((ctx.x - ctx.x0) ** 2) / len(ctx.x))

def obj_smoothness(ctx: ObjectiveContext) -> float:
    if len(ctx.x) <= 1: return 0.0
    d = np.diff(ctx.x)
    return float(np.sqrt(np.sum(d * d) / len(d)))



# -------- Hydraulic (wntr.metrics.hydraulic) --------

def obj_expected_demand_served_ratio(ctx):
    """
    Water service availability is the ratio of delivered demand to the expected demand.
    Water Service Availability: delivered / expected (network-wide average)
    Returns a per-timestep, per-node ratio; here we use the global mean. Higher is better -> take the negative (for minimization).
    Must be computed under PDD (pressure-dependent demand) mode.
    """

    if "demand" not in ctx.sim.node:
        return np.inf

    try:
        demand = ctx.sim.node["demand"].loc[:, ctx.wn.junction_name_list]
        expected_demand = wntr.metrics.hydraulic.expected_demand(ctx.wn)
        if expected_demand is None:
            return np.inf

        common_idx = expected_demand.index.intersection(demand.index)
        common_cols = expected_demand.columns.intersection(demand.columns)
        if len(common_idx) == 0 or len(common_cols) == 0:
            return np.inf

        expected = expected_demand.loc[common_idx, common_cols].astype(float)
        delivered = demand.loc[common_idx, common_cols].astype(float)

        eps = 1e-12
        active_cols = expected.max(axis=0) > eps
        if not bool(active_cols.any()):
            return np.inf

        expected = expected.loc[:, active_cols]
        delivered = delivered.loc[:, active_cols]
        active_cells = expected > eps

        served_ratio = delivered.clip(lower=0.0).where(active_cells) / expected.where(active_cells)
        served_ratio = served_ratio.clip(lower=0.0, upper=1.0)
        value = float(np.nanmean(served_ratio.to_numpy(dtype=float)))
        if not np.isfinite(value):
            return np.inf
        return -value
    except Exception:
        return np.inf


def obj_undelivered_demand(ctx):
    """
    Total undelivered demand volume over the simulation period.

    WNTR demand results are flow rates, so the demand deficit is multiplied by
    the report timestep to obtain a total volume.
    """
    if ctx.wn is None:
        return np.inf

    if "demand" not in ctx.sim.node:
        return np.inf

    try:
        delivered = ctx.sim.node["demand"]
        expected_demand = wntr.metrics.hydraulic.expected_demand(ctx.wn)
        if expected_demand is None:
            return np.inf

        junctions = pd.Index(ctx.wn.junction_name_list)
        common_idx = expected_demand.index.intersection(delivered.index)
        common_cols = junctions.intersection(expected_demand.columns).intersection(delivered.columns)
        if len(common_idx) == 0 or len(common_cols) == 0:
            return np.inf

        expected = expected_demand.loc[common_idx, common_cols].astype(float)
        delivered = delivered.loc[common_idx, common_cols].astype(float).clip(lower=0.0)

        active = expected > 1e-12
        deficit = (expected - delivered).clip(lower=0.0).where(active, 0.0)
        total_volume = float(np.nansum(deficit.to_numpy(dtype=float))) * float(
            ctx.wn.options.time.report_timestep
        )
        if not np.isfinite(total_volume):
            return np.inf
        return total_volume
    except Exception:
        return np.inf


def obj_total_excessive_pressure(ctx):
    """
    Total excessive pressure.
    Minimize squared pressure above the required service pressure over all
    junctions and reporting timesteps.

    The paper also applies P_i >= P_ref as a constraint. For feasible
    solutions, this is equivalent to sum((P_i,t - P_ref)^2).
    """
    if ctx.wn is None:
        return np.inf

    if "pressure" not in ctx.sim.node:
        return np.inf

    try:
        pressure = ctx.sim.node["pressure"]
        junctions = list(ctx.wn.junction_name_list)
        node_mode = ctx.params.get("pressure_objective", {}).get("nodes", "all")
        if node_mode == "demand":
            expected = wntr.metrics.hydraulic.expected_demand(ctx.wn)
            active = expected.max(axis=0) > 1e-12
            junctions = [name for name in junctions if name in active.index and bool(active.loc[name])]

        cols = [name for name in junctions if name in pressure.columns]
        if not cols:
            return np.inf

        service_cfg = ctx.params.get("service_pressure", {})
        ref_pressure = service_cfg.get("minimum_service_pressure", None)
        if ref_pressure is None:
            ref_pressure = ctx.params.get("pressure_violation", {}).get("min_pressure", 0.0)
        ref_pressure = float(ref_pressure)

        excess_pressure = (pressure.loc[:, cols].astype(float) - ref_pressure).clip(lower=0.0)
        return float(np.nansum(np.square(excess_pressure.to_numpy(dtype=float))))
    except Exception:
        return np.inf


def obj_water_quality_risk(ctx):
    """
    Water quality risk based on water age.

    Risk_q = 1 - Re_q
    Re_q = sum(b_i,t * Q_avl_i,t) / sum(Q_req_i,t)
    where Q_avl is delivered demand under PDD and Q_req is expected demand.
    """
    if ctx.wn is None:
        return np.inf

    if "quality" not in ctx.sim.node or "demand" not in ctx.sim.node:
        return np.inf

    try:
        expected_demand = wntr.metrics.hydraulic.expected_demand(ctx.wn)
        if expected_demand is None:
            return np.inf

        delivered = ctx.sim.node["demand"]
        water_age = ctx.sim.node["quality"]
        junctions = pd.Index(ctx.wn.junction_name_list)

        common_idx = expected_demand.index.intersection(delivered.index).intersection(water_age.index)
        common_cols = junctions.intersection(expected_demand.columns)
        common_cols = common_cols.intersection(delivered.columns).intersection(water_age.columns)
        if len(common_idx) == 0 or len(common_cols) == 0:
            return np.inf

        q_req = expected_demand.loc[common_idx, common_cols].astype(float)
        q_avl = delivered.loc[common_idx, common_cols].astype(float)
        water_age = water_age.loc[common_idx, common_cols].astype(float)

        eps = 1e-12
        active = q_req > eps
        if not bool(active.any().any()):
            return np.inf

        q_avl = q_avl.clip(lower=0.0).where(active, 0.0)
        q_avl = q_avl.where(q_avl <= q_req, q_req)

        b = pd.DataFrame(0.0, index=water_age.index, columns=water_age.columns)
        b = b.mask(water_age <= 6.0, 1.0)
        transition = (water_age > 6.0) & (water_age < 10.0)
        b = b.mask(transition, -0.125 * (water_age - 6.0) + 1.0)
        b = b.clip(lower=0.0, upper=1.0).where(active, 0.0)

        numerator = float(np.nansum((b * q_avl).to_numpy(dtype=float)))
        denominator = float(np.nansum(q_req.where(active, 0.0).to_numpy(dtype=float)))
        if denominator <= eps:
            return np.inf

        reliability = numerator / denominator
        risk = 1.0 - reliability
        return float(np.clip(risk, 0.0, 1.0))
    except Exception:
        return np.inf

def obj_entropy(ctx):
    """
    Entropy is a measure of uncertainty in a random variable. In a water distribution network model,
    the random variable is flow in the pipes and entropy can be used to measure alternate flow paths when a network component fails.
    A network that carries maximum entropy flow is considered reliable with multiple alternate paths.
    Connectivity will change at each timestep, depending on the flow direction. The to_graph method can be used to generate a weighted graph.
    Entropy can be computed using the entropy method.
    Higher is better -> take the negative (for minimization)
    """
    if ctx.wn is None:
        return np.inf

    if "flowrate" not in ctx.sim.link:
        return np.inf
    flowrate = ctx.sim.link['flowrate']
    G = ctx.wn.to_graph()

    edge_refs = [(u, v, k, data.get("name", k)) for u, v, k, data in G.edges(keys=True, data=True)]


    S_list = []
    for t in flowrate.index:
        s = flowrate.loc[t, :]
        s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        for u, v, k, nm in edge_refs:
            G[u][v][k]["weight"] = float(s.get(nm, 0.0))

        try:
            _, S_sys = wntr.metrics.hydraulic.entropy(G)
            S_list.append(S_sys)
        except Exception:
            S_list.append(np.nan)

    return -_mean_or_inf(S_list)

def obj_todini_index(ctx):
    """
    The Todini index is related to the capability of a system to overcome failures while still meeting demands and pressures at nodes.
    The Todini index defines resilience at a specific time as a measure of surplus power at each node.
    Higher is better -> take the negative (for minimization)
    """

    if ctx.wn is None:
        return np.inf

    if "demand" or "head" or "pressure" not in ctx.sim.node:
        return np.inf

    demand = ctx.sim.node["demand"]
    head = ctx.sim.node["head"]
    pressure = ctx.sim.node["pressure"]

    pumps = list(ctx.wn.pump_name_list)
    if not pumps:
        raise ValueError("Network has no pumps.")

    flowrate = ctx.sim.link["flowrate"].loc[:,ctx.wn.pump_name_list]

    Pstar = ctx.params.get("Pstar", 15)

    try:
        TI = wntr.metrics.hydraulic.todini_index(head, pressure, demand, flowrate, ctx.wn, Pstar)
        return -_mean_or_inf(TI)
    except Exception:
        return np.inf

def obj_modified_resilience_index(ctx):
    """
    The modified resilience index is similar to the Todini index, but is only computed at junctions.
    The metric defines resilience at a specific time as a measure of surplus power at each junction or as a system average.
    Higher is better -> take the negative (for minimization)
    wntr.metrics.hydraulic.modified_resilience_index(wn, results)
    """

    if ctx.wn is None:
        return np.inf

    if "pressure" not in ctx.sim.node:
        return np.inf


    pressure = ctx.sim.node["pressure"].loc[:,ctx.wn.junction_name_list]
    elev_all = ctx.wn.query_node_attribute('elevation')
    elev_junc = elev_all.loc[ctx.wn.junction_name_list]

    Pstar = ctx.params.get("Pstar", None)
    if Pstar is None:
        Pstar = ctx.params.get("service_pressure", {}).get("minimum_service_pressure", 15.0)
    Pstar = float(Pstar)

    try:
        MRI = wntr.metrics.hydraulic.modified_resilience_index(pressure, elev_junc, Pstar)
        return -_mean_or_inf(MRI)
    except Exception:
        return np.inf


def obj_tank_capacity(ctx):
    """
    Tank capacity is the ratio of current water volume stored in tanks to the maximum volume of water that can be stored.
    This metric is measured at each tank as a function of time and ranges between 0 and 1.
    A value of 1 indicates that tank storage is maximized, while a value of 0 means there is no water stored in the tank.
    Higher is better -> take the negative (for minimization)
    """

    if ctx.wn is None:
        return np.inf

    if "pressure" not in ctx.sim.node:
        return np.inf

    tanks = list(ctx.wn.tank_name_list)
    if not tanks:
        raise ValueError("Network has no tanks.")


    pressure = ctx.sim.node["pressure"].loc[:, ctx.wn.tank_name_list]

    try:
        cap = wntr.metrics.hydraulic.tank_capacity(pressure, ctx.wn)
        return -_mean_or_inf(cap)
    except Exception:
        return np.inf


# -------- Economic (wntr.metrics.economic) --------

def obj_ghg_emissions(ctx):
    '''
    Greenhouse gas emissions is the annual emissions associated with pipes based on equations from the Battle of Water Networks II
    unit: kg-CO2-e/yr
    '''
    if ctx.wn is None:
        return np.inf
    try:
        ghg = wntr.metrics.economic.annual_ghg_emissions(ctx.wn)
        return float(ghg)
    except Exception:
        return np.inf


def obj_pump_cost(ctx):
    """
    pump cost, lower is better
    need to set
    """

    if ctx.wn is None:
        return np.inf

    if "head" not in ctx.sim.node:
        return np.inf

    head = ctx.sim.node["head"]

    pumps = list(ctx.wn.pump_name_list)
    if not pumps:
        raise ValueError("Network has no pumps.")

    flowrate = ctx.sim.link["flowrate"].loc[:, ctx.wn.pump_name_list]

    time = flowrate.index

    headloss = pd.DataFrame(data=None, index=time, columns=pumps)
    for pump_name, pump in ctx.wn.pumps():
        start_node = pump.start_node_name
        end_node = pump.end_node_name
        start_head = head.loc[:, start_node]
        end_head = head.loc[:, end_node]
        headloss.loc[:, pump_name] = end_head - start_head

    efficiency_dict = {}
    for pump_name, pump in ctx.wn.pumps():
        if pump.efficiency is None:
            efficiency_dict[pump_name] = [ctx.wn.options.energy.global_efficiency / 100.0 for i in time]
        else:
            curve = ctx.wn.get_curve(pump.efficiency)
            x = [point[0] for point in curve.points]
            y = [point[1] / 100.0 for point in curve.points]
            interp = scipy.interpolate.interp1d(x, y, kind='linear')
            efficiency_dict[pump_name] = interp(np.array(flowrate.loc[:, pump_name]))

    efficiency = pd.DataFrame(data=efficiency_dict, index=time, columns=pumps)

    power = 1000.0 * 9.81 * headloss * flowrate / efficiency  # Watts = J/s
    energy = power * ctx.wn.options.time.report_timestep / 3600000

    if ctx.price is None:
        price = 1

    try:
        c = energy * price
        return float(np.nansum(c))
    except Exception:
        return np.inf


def obj_pump_energy(ctx):
    """
    pump energy
    unit: kwh
    """
    if ctx.wn is None:
        return np.inf

    if "head" not in ctx.sim.node:
        return np.inf

    head = ctx.sim.node["head"]

    pumps = list(ctx.wn.pump_name_list)
    if not pumps:
        raise ValueError("Network has no pumps.")

    flowrate = ctx.sim.link["flowrate"].loc[:, ctx.wn.pump_name_list]
    try:
        time = flowrate.index

        headloss = pd.DataFrame(data=None, index=time, columns=pumps)
        for pump_name, pump in ctx.wn.pumps():
            start_node = pump.start_node_name
            end_node = pump.end_node_name
            start_head = head.loc[:, start_node]
            end_head = head.loc[:, end_node]
            headloss.loc[:, pump_name] = end_head - start_head

        efficiency_dict = {}
        for pump_name, pump in ctx.wn.pumps():
            if pump.efficiency is None:
                efficiency_dict[pump_name] = [ctx.wn.options.energy.global_efficiency / 100.0 for i in time]
            else:
                curve = ctx.wn.get_curve(pump.efficiency)
                x = [point[0] for point in curve.points]
                y = [point[1]/100.0 for point in curve.points]
                interp = scipy.interpolate.interp1d(x, y, kind='linear')
                efficiency_dict[pump_name] = interp(np.array(flowrate.loc[:, pump_name]))

        efficiency = pd.DataFrame(data=efficiency_dict, index=time, columns=pumps)

        power = 1000.0 * 9.81 * headloss * flowrate / efficiency  # Watts = J/s
        E = power * ctx.wn.options.time.report_timestep/3600000
        return _mean_or_inf(np.nansum(E))
    except Exception:
        return np.inf



# -------- Water security (wntr.metrics.water_security) --------
def obj_extent_contaminant(ctx):
    """
    Extent of contamination is the length of contaminated pipe at each node-time pair
    """

    if ctx.wn is None:
        return np.inf

    if "quality" not in ctx.sim.node:
        return np.inf

    if "flowrate" not in ctx.sim.link:
        return np.inf

    flowrate = ctx.sim.link["flowrate"].loc[:, ctx.wn.pipe_name_list]
    quality = ctx.sim.node["quality"]

    detection_limit = ctx.params.get("water_security", {}).get("detection_limit", 1e-6)
    try:
        ext = wntr.metrics.water_security.extent_contaminant(quality, flowrate, ctx.wn, detection_limit)
        return _mean_or_inf(ext)
    except Exception:
        return np.inf

def obj_mass_contaminant_consumed(ctx):
    """
    Mass consumed is the mass of a contaminant that exits the network via node demand at each node-time pair
    """

    if ctx.wn is None:
        return np.inf

    if "quality" or "demand" not in ctx.sim.node:
        return np.inf

    demand = ctx.sim.node["demand"].loc[:, ctx.wn.junction_name_list]
    quality = ctx.sim.node["quality"].loc[:, ctx.wn.junction_name_list]

    detection_limit = ctx.params.get("water_security", {}).get("detection_limit", 1e-6)
    try:
        m = wntr.metrics.water_security.mass_contaminant_consumed(demand, quality, detection_limit)
        return _mean_or_inf(m)
    except Exception:
        return np.inf

def obj_volume_contaminant_consumed(ctx):
    """
    Volume consumed is the volume of a contaminant that exits the network via node demand at each node-time pair
    """
    if ctx.wn is None:
        return np.inf

    if "quality" or "demand" not in ctx.sim.node:
        return np.inf

    demand = ctx.sim.node["demand"].loc[:, ctx.wn.junction_name_list]
    quality = ctx.sim.node["quality"].loc[:, ctx.wn.junction_name_list]

    detection_limit = ctx.params.get("water_security", {}).get("detection_limit", 1e-6)
    try:
        v = wntr.metrics.water_security.volume_contaminant_consumed(demand, quality, detection_limit)
        return _mean_or_inf(v)
    except Exception:
        return np.inf



# -------- Misc (wntr.metrics.misc.population_impacted) --------

OBJECTIVE_REGISTRY: Dict[str, Callable[[ObjectiveContext], float]] = {
    # Registry stays extensible; the workflow schema restricts the active
    # multi-objective pair to pump_energy and modified_resilience_index.
    "expected_demand_served_ratio": obj_expected_demand_served_ratio,
    "undelivered_demand": obj_undelivered_demand,
    "total_excessive_pressure": obj_total_excessive_pressure,
    "water_quality_risk": obj_water_quality_risk,
    "pump_energy": obj_pump_energy,
    "modified_resilience_index": obj_modified_resilience_index,
}


