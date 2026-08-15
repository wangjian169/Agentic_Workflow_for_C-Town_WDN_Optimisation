# Agentic Workflow for C-Town WDN Optimisation

This project implements a multi-agent workflow for water distribution network
(WDN) optimisation on the C-Town EPANET model. A LangGraph-based agent pipeline
turns a high-level optimisation request into a runnable configuration, optional
comparison experiments, optimisation runs, plots, telemetry, and summaries.

The workflow is designed for controlled experiments on agent autonomy. The
parameter configuration, experiment design, and execution nodes can each run in
either structured workflow mode or ReAct mode, making it possible to compare
different control-transfer settings while keeping the same WDN optimisation
problem.

## Features

- C-Town EPANET model included in `networks/ctown.inp`.
- Single-objective pump-energy optimisation.
- Multi-objective energy/resilience optimisation.
- Matched single- vs multi-objective comparison runs.
- Configurable agent execution modes: workflow (`W`) or ReAct (`R`).
- Reproducible batch execution across seeds, repeats, model profiles, and task
  specifications.
- Per-run JSON summaries, configuration traces, node telemetry, console traces,
  and generated plots.

## Project Structure

```text
.
|-- graph_workflow.py                 # Main multi-agent workflow
|-- structured_output.py              # Pydantic schemas and workflow state types
|-- experiments/
|   |-- control_transfer_runner.py    # Command-line runner
|   `-- __init__.py
|-- tools/                            # Optimisation, plotting, and ReAct tools
|-- networks/
|   `-- ctown.inp                     # C-Town EPANET input model
|-- reference_specs/                  # Baseline task specifications
|-- results/                          # Example and generated run outputs
|-- requirements.txt
`-- README.md
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Model Configuration

The runner reads model credentials from environment variables. No API key is
stored in the source code.

PowerShell example:

```powershell
$env:OPENAI_API_KEY = "your_key_here"
```

Command Prompt example:

```bat
set OPENAI_API_KEY=your_key_here
```

The default model profile is `deepseek`. Additional model profiles are defined
in `experiments/control_transfer_runner.py`:

```text
deepseek       -> DeepSeek chat model, using OPENAI_API_KEY
aliyun_qwen    -> DashScope/OpenAI-compatible Qwen endpoint, using DASHSCOPE_API_KEY
ollama_qwen    -> local OpenAI-compatible endpoint, using OLLAMA_API_KEY
```

You can also override model settings directly with command-line arguments such
as `--model`, `--model-provider`, `--model-base-url`, `--model-api-key-env`,
`--temperature`, and `--max-tokens`.

## Optimisation Problem Options

The default C-Town tasks use pressure-dependent demand (`PDD`), pump-speed time
series variables for all pumps, and a required service pressure of 20 m. These
defaults are stored in `reference_specs/`.

### Objectives

The current workflow schema focuses on the following objective names:

```text
pump_energy
modified_resilience_index
```

`pump_energy` is used for single-objective optimisation. The pair
`pump_energy` and `modified_resilience_index` is used for the default
multi-objective trade-off task. The lower-level objective registry in
`tools/objective_functions.py` also contains additional engineering metrics
such as expected demand served ratio, undelivered demand, excessive pressure,
and water quality risk, but the validated workflow interface is centred on the
energy/resilience tasks above.

### Decision Variables

The default experiments optimise:

```text
pump_speed with setter pump_speed_masked
items = ALL_PUMPS
timeseries = true
bounds = 0.5 to 1.5
```

The variable schema and setter registry also support other WDN controls and
network attributes, including:

```text
pipe_roughness_masked
pipe_diameter_masked
pipe_status_masked
valve_status_masked
valve_setting_masked
pump_speed_masked
node_emitter_masked
demand_multiplier_masked
junction_initial_quality_masked
reservoir_head_masked
source_strength_timeseries
tank_initial_level_masked
tank_min_level_masked
tank_max_level_masked
tank_volume_curve
random_selection_mask
randomize_existing_source_types
```

Supported item selectors include `ALL_PIPES`, `ALL_PUMPS`, `ALL_VALVES`,
`ALL_TANKS`, `ALL_JUNCTIONS`, `ALL_SOURCES`, and `ALL_RESERVOIRS`, or explicit
lists of network element IDs.

### Algorithms

The runner uses `pymoo` algorithms through the runtime loader in `tools/tools.py`.
The validated schema exposes:

```text
Single-objective: GA, DE, CMAES, PSO
Multi-objective: NSGA2, SMSEMOA
```

The runtime loader also contains entries for additional multi-objective
algorithms such as `MOEAD`, `AGEMOEA`, and `RVEA`, but those are not part of the
default reference specifications.

Common algorithm settings are passed through `algorithm.kwargs`, for example:

```json
{"pop_size": 20, "eliminate_duplicates": true}
```

Termination is configured with:

```json
{"type": "n_gen", "value": 20}
```

## Task Specifications

Three reference task specifications are included:

```text
single_obj_ctown     -> pump_energy minimisation with GA
multi_obj_ctown      -> pump_energy + modified_resilience_index with NSGA-II
combined_obj_ctown   -> matched single-objective and multi-objective comparison
```

These are stored as JSON files in `reference_specs/`. They provide the fallback
configuration used when human input is disabled or when the runner executes
batch experiments.

## Running the Workflow

Run the default combined optimisation comparison:

```bash
python -m experiments.control_transfer_runner
```

Run a single-objective C-Town task:

```bash
python -m experiments.control_transfer_runner --spec single_obj_ctown --combo WWW --seed 1 --n-repeats 1
```

Run a multi-objective C-Town task:

```bash
python -m experiments.control_transfer_runner --spec multi_obj_ctown --combo RRR --seed 1 --n-repeats 1
```

Run a selected set of mode combinations:

```bash
python -m experiments.control_transfer_runner --spec combined_obj_ctown --combo WWW WWR RWW RRR --n-repeats 3
```

Run all configured task specifications and all eight mode combinations:

```bash
python -m experiments.control_transfer_runner --all
```

Write outputs to a custom directory:

```bash
python -m experiments.control_transfer_runner --results-root results/my_run --n-repeats 1
```

Use another model profile:

```bash
python -m experiments.control_transfer_runner --model-profile aliyun_qwen --n-repeats 1
```

## Mode Combinations

The `--combo` argument is a three-letter code controlling these nodes:

```text
parameter_agent, experiment_agent, running_node
```

Each letter is either:

```text
W = structured workflow mode
R = ReAct mode
```

For example:

```text
WWW -> all three toggleable nodes run in workflow mode
RWW -> parameter_agent runs in ReAct mode
WRW -> experiment_agent runs in ReAct mode
WWR -> running_node runs in ReAct mode
RRR -> all three toggleable nodes run in ReAct mode
```

The planning node is always workflow-based. The report node is optional and can
be skipped with `--no-report`.

## Useful Runner Arguments

```text
--spec                 single_obj_ctown, multi_obj_ctown, or combined_obj_ctown
--combo                one or more W/R mode combinations
--seed                 fixed seed for a single run
--seeds                number of random seed slots for a selected spec/combo set
--n-repeats            repeat count per seed slot
--all                  full sweep over all specs and all mode combinations
--results-root         output directory
--model-profile        model preset
--allow-human-input    allow interactive parameter questions
--no-human-input       use reference fallback for missing fields
--quiet                save traces without echoing full run output to terminal
--trace-max-chars      max characters stored per traced LLM input/output block
--react-recursion-limit
--no-report            skip report generation
```

Task prompts can also be overridden with `--single-task-description`,
`--multi-task-description`, or `--combined-task-description`. Use `{seed}` in a
custom prompt if the run seed should be inserted into the task text.

## Outputs

Run outputs are written under the selected `results` root. A typical run folder
contains:

```text
summary.json
run_metadata.json
run_console_trace.txt
final_config.json
engineering_outputs.json
process_outputs.json
node_metrics.json
node_telemetry.json
report.md
state.pkl
*.png
```

The aggregate file `all_runs.jsonl` is appended under the selected results root.
Per-run folders are grouped by task specification, model label, mode
combination, seed, and repeat index.

Example layout:

```text
results/deepseek/combined_obj_ctown/deepseek-chat/RRR/seed_1/rep_001/
```

## Notes

- The bundled network file is `networks/ctown.inp`.
- Reference task specifications are stored in `reference_specs/`.
- Existing example outputs are kept under `results/`.
- Temporary Python caches, local IDE settings, local environment files, and
  private supporting trace files are ignored by `.gitignore`.
- Do not commit real API keys or local credential files.
