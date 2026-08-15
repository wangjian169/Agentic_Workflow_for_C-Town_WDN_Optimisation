from __future__ import annotations

from typing import List, Dict, Any, Union, Literal, Optional, TypedDict
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

# =========================
# Common base class: JSON friendly
# =========================

class JsonModel(BaseModel):
    """Base class for all models that will be exported as JSON."""
    model_config = ConfigDict(
        use_enum_values=True,   # Enum -> use .value directly, e.g. AlgorithmName.GA -> "ga"
        extra="forbid",         # Forbid unexpected fields (change to "ignore" if needed)
    )

# -------------------------
# Variable / VarSpec section
# -------------------------

ItemsKeyword = Literal[
    "ALL_PIPES",
    "ALL_PUMPS",
    "ALL_VALVES",
    "ALL_TANKS",
    "ALL_JUNCTIONS",
    "ALL_SOURCES",
    "ALL_RESERVOIRS",
]

SetterName = Literal[
    "random_selection_mask",
    "pipe_roughness_masked",
    "pipe_diameter_masked",
    "node_emitter_masked",
    "demand_multiplier_masked",
    "junction_initial_quality_masked",
    "randomize_existing_source_types",
    "source_strength_timeseries",
    "tank_initial_level_masked",
    "tank_min_level_masked",
    "tank_max_level_masked",
    "pipe_status_masked",
    "valve_status_masked",
    "tank_volume_curve",
    "pump_speed_masked",
    "reservoir_head_masked",
    "valve_setting_masked",
]


class Bounds(JsonModel):
    lb: float
    ub: float


class VarSpec(JsonModel):
    """
    VarSpec used in variable_parameters_agent.

    - name: variable name, e.g. "pipe_roughness"
    - items: set of items, can be ALL_* placeholders or a list of IDs
    - setter: must be one of SetterName
    - setter_kwargs:
        * in most cases: {"group_name": "all"}
        * for random_selection_mask: {"group_name": "SRC", "choose_rate": float}
    - timeseries: whether this variable is time-series based
    - bounds: required, with lb < ub
    """
    name: str
    items: Union[ItemsKeyword, List[str]]
    setter: SetterName
    setter_kwargs: Dict[str, Any] = Field(default_factory=dict)
    timeseries: bool
    bounds: Bounds


class VariableParameters(JsonModel):
    """
    Legacy structure: a wrapper around a list of VarSpec.
    If you no longer use this wrapper, you can safely remove it.
    """
    var_specs: List[VarSpec]


# -------------------------
# Objective / plan section
# -------------------------

class DemandModel(str, Enum):
    PDD = "PDD"
    DDA = "DDA"   # Unified enum value for "DD"/"DDA" in prompts


class AlgorithmName(str, Enum):
    GA = "ga"
    NSGA2 = "nsga2"
    SMSEMOA = "smsemoa"
    DE = "de"
    CMAES = "cmaes"
    PSO = "pso"
    # Add new algorithms here when needed


class TerminationType(str, Enum):
    N_GEN = "n_gen"
    TIME = "time"


# Objective names: restrict to allowed strings via Enum
class ObjectiveName(str, Enum):
    PUMP_ENERGY = "pump_energy"
    MODIFIED_RESILIENCE_INDEX = "modified_resilience_index"
    # Workflow-restricted objective set:
    # - pump_energy
    # - modified_resilience_index


class AlgorithmConfig(JsonModel):
    """
    Simplified algorithm configuration:
    - name: one of AlgorithmName
    - kwargs: parameter dict for the pymoo algorithm (e.g. pop_size, etc.)
    """
    name: AlgorithmName
    kwargs: Dict[str, Any] = Field(default_factory=dict)


class TerminationConfig(JsonModel):
    """
    Termination condition configuration:
    - type: "n_gen" or "time"
    - value: positive float
    """
    type: TerminationType
    value: float


class ObservedSeries(JsonModel):
    """
    Represents a time series loaded from CSV, e.g.:
    {"csv": "...", "index_col": 0, "parse_dates": true}
    """
    csv: str
    index_col: int = 0
    parse_dates: bool = True


class ObservedBlock(JsonModel):
    """
    Block of observed data.
    It is kept optional for extensibility, but the current workflow objective set
    does not require observed time series.
    """
    pressure: Optional[ObservedSeries] = None
    flow: Optional[ObservedSeries] = None
    head: Optional[ObservedSeries] = None


class ObjectiveParameters(JsonModel):
    """
    A typical objective_parameters structure.

    Roughly corresponds to the `plan` field in objective_parameters.json /
    objective_parameters_agent.

    Required when status is COMPLETE:
    - inp_path
    - objectives (non-empty)
    - demand_model
    - minimum_service_pressure if modified_resilience_index is selected
    - algorithm
    - termination

    In the current workflow, supported objectives are model-based metrics and do
    not require observed data.
    """
    inp_path: str
    objectives: List[ObjectiveName]

    # Optional weights; if present, length must equal len(objectives)
    objective_weights: Optional[List[float]] = None

    observed: Optional[ObservedBlock] = None

    use_epanet_toolkit: bool = True
    demand_model: DemandModel

    # Hard minimum pressure checked over all junctions.
    pressure_min_constraint: float = 0.0
    # PDD required pressure written to the WNTR network before simulation.
    minimum_service_pressure: Optional[float] = 20.0
    detection_limit: float = 1.0

    algorithm: AlgorithmConfig
    termination: TerminationConfig

    seed: int = 1
    verbose: bool = True


class PartialBounds(JsonModel):
    """Partial bounds used while collecting parameters interactively."""
    lb: Optional[Union[float, str]] = None
    ub: Optional[Union[float, str]] = None


class PartialAlgorithmConfig(JsonModel):
    """Partial algorithm config used during intermediate dialogue turns."""
    name: Optional[str] = None
    kwargs: Dict[str, Any] = Field(default_factory=dict)


class PartialTerminationConfig(JsonModel):
    """Partial termination config used during intermediate dialogue turns."""
    type: Optional[str] = None
    value: Optional[Union[float, str]] = None


class PartialVarSpec(JsonModel):
    """Partial VarSpec used while the agent is still collecting fields."""
    name: Optional[str] = None
    items: Optional[Union[ItemsKeyword, str, List[str]]] = None
    setter: Optional[str] = None
    setter_kwargs: Dict[str, Any] = Field(default_factory=dict)
    timeseries: Optional[bool] = None
    bounds: Optional[PartialBounds] = None


class PartialObjectiveParameters(JsonModel):
    """Partial objective_parameters used while the template is incomplete."""
    inp_path: Optional[str] = None
    objectives: Optional[List[str]] = None
    objective_weights: Optional[List[float]] = None
    observed: Optional[ObservedBlock] = None
    use_epanet_toolkit: Optional[bool] = None
    demand_model: Optional[str] = None
    pressure_min_constraint: Optional[Union[float, str]] = None
    minimum_service_pressure: Optional[Union[float, str]] = None
    detection_limit: Optional[Union[float, str]] = None
    algorithm: Optional[PartialAlgorithmConfig] = None
    termination: Optional[PartialTerminationConfig] = None
    seed: Optional[int] = None
    verbose: Optional[bool] = None

class ParameterCandidate(JsonModel):
    """A complete set of configurations for comparison and selection"""
    name: str
    objective: ObjectiveParameters
    variables: List[VarSpec]

class ParameterCandidate(JsonModel):
    """A complete experiment configuration for comparison."""
    name: str
    objective: ObjectiveParameters
    variables: List[VarSpec]
    description: str | None = None  # short explanation of how it differs from baseline

class ExperimentsDesign(JsonModel):
    """Set of experiment candidates built on top of the baseline."""
    helper_message: str | None = None
    experiments: List[ParameterCandidate]


class PlannerDecision(BaseModel):
    """Planner's task classification and whether experiments are required."""
    category: Literal["OPTIMIZATION", "COMPARISON", "HYPOTHESIS"]
    experiment_required: bool
    reason: str


class HypothesisBundle(JsonModel):
    """Explicit hypothesis content for scientific workflow tasks."""
    hypothesis: str
    mechanism_rationale: str
    testable_prediction: str



# -------------------------
# Unified output of ParameterAgent
# -------------------------

class ParameterStatus(str, Enum):
    """
    Status for structured output:
    - COLLECTING: still missing parameters; more input from the user is required
    - COMPLETE:   all required fields are filled; optimisation can be executed
    """
    COLLECTING = "COLLECTING"
    COMPLETE = "COMPLETE"


class ParametersState(JsonModel):
    """
    Structured output for each turn of the Parameter Agent.

    Field description:
    - status:
        * COLLECTING: required fields in objective/variables are missing or invalid
        * COMPLETE:   objective & variables both exist and pass validation
    - objective:    partial objective_parameters accumulated so far, or None
    - variables:    partial VarSpec list accumulated so far, or None
    - missing_keys: list of missing/uncertain field paths (e.g. "objective.inp_path")
    - questions_to_user: concrete questions for the user (ideally prefixed with field path)
    - suggestions:  suggested safe defaults or optional enhancements
    - helper_message: short summary of what has been done in this step

    Note: By default you should produce a single coherent configuration using `objective` and `variables`.
            If the user explicitly asks to **compare different variable sets or hyperparameter configurations**, you may additionally populate the `candidates` field with 2-4 alternative configurations.
            Each candidate must be internally consistent and runnable.
    """
    status: ParameterStatus

    objective: Optional[PartialObjectiveParameters] = None
    variables: Optional[List[PartialVarSpec]] = None  # Flattened partial VarSpec list

    missing_keys: List[str] = Field(default_factory=list)
    questions_to_user: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)

    helper_message: str = ""


class WorkflowState(TypedDict, total=False):
    """
    High-level state container for the workflow.
    This is passed between agents/nodes in the graph.
    """

    # ---- user input ----
    user_query: str
    allow_human_input: bool
    trace_enabled: bool
    trace_llm_io: bool
    trace_max_chars: int
    react_recursion_limit: int

    # ---- planning_agent ----
    plan: str
    task_category: Literal["OPTIMIZATION", "COMPARISON", "HYPOTHESIS"]
    needs_experiments: bool
    use_experiment_agent: bool
    planner_reason: str
    hypothesis_text: str
    mechanism_rationale: str
    testable_prediction: str

    # ---- parameter_agent ----
    objective_parameters: Dict[str, Any]
    variable_parameters: List[Dict[str, Any]]

    parameter_status: Literal["COLLECTING", "COMPLETE"]
    parameter_questions: List[str]
    helper_message: str
    missing_keys: List[str]
    minimum_service_pressure_confirmed: bool
    random_seed: int
    reference_objective_parameters: Dict[str, Any]
    reference_variable_parameters: List[Dict[str, Any]]
    reference_fallback_used: bool

    # ---- experiment_agent ----
    experiments: List[Dict[str, Any]]
    experiment_helper_message: str

    # ---- running_node ----
    experiment_results: List[Dict[str, Any]]
    figure_info: List[Dict[str, Any]]          # [{path, category, description}, ...]

    # ---- report_agent ----
    report: str

    workflow_token_usage: Dict[str, int]
    workflow_token_breakdown: Dict[str, Dict[str, int]]

    visualization_notes: str
    output_requirements: Dict[str, Any]

    # ---- control-transfer study (workflow ↔ ReAct autonomy axis) ----
    # node_modes: per-node mode, e.g. {"parameter_agent": "react",
    # "experiment_agent": "workflow", "running_node": "react"}.
    # Missing entries default to "workflow" in the dispatcher.
    node_modes: Dict[str, Literal["workflow", "react"]]

    # node_metrics: telemetry filled by the @record_node_metrics decorator.
    # See record_node_metrics in graph_workflow.py for the schema.
    node_metrics: Dict[str, Dict[str, Any]]

    # configuration_trace: append-only log of (field, value, source, node, t)
    # written by parameter / experiment ReAct tools (and the workflow versions
    # for parity), so the post-hoc evaluator can reconstruct who set what.
    configuration_trace: List[Dict[str, Any]]

    # reference_spec_id: identifier of the reference task specification.
    # Examples: "single_obj_ctown", "multi_obj_ctown", "combined_obj_ctown".
    reference_spec_id: Optional[str]

    # Per-run persistence metadata injected by experiments/control_transfer_runner.py.
    run_id: str
    run_output_dir: str
