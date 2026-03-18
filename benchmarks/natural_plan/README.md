# NaturalPlan Benchmark

Calendar scheduling, meeting planning, trip planning — 3 tasks × 5 levels (zeroshot_structured, zeroshot_unstructured, workflow, pot, react).

## Setup

```bash
cd secretagent && uv sync
export TOGETHER_API_KEY="your-key"
```

## Flow

1. `expt.py run` loads data → implements ptools via config → evaluates interface on dataset
2. Results go to `results/{task}/{timestamp}.{expt_name}/` (csv, config, run_summary.json)
3. `make report` aggregates runs into a table

## Examples

```bash
cd benchmarks/natural_plan

# Quick test (2 samples)
uv run python expt.py run --config-file conf/calendar.yaml \
  evaluate.expt_name=cal_workflow ptools.calendar_scheduling.method=direct \
  ptools.calendar_scheduling.fn=ptools_calendar.calendar_workflow dataset.n=2

# Or Makefile
make cal_workflow

# 50 stratified samples (AgentProject-compatible)
uv run python expt.py run --config-file conf/calendar.yaml \
  evaluate.expt_name=cal_workflow ptools.calendar_scheduling.method=direct \
  ptools.calendar_scheduling.fn=ptools_calendar.calendar_workflow \
  dataset.stratified=true dataset.sample_n=50 dataset.n=50

# All 15 combinations, n=5 each
make run_all_15

# View results (writes report.md, grouped by task)
make report
```

## Config

| Param | Default | Description |
|-------|---------|-------------|
| `llm.model` | `together_ai/deepseek-ai/DeepSeek-V3.1` | Model string (litellm) |
| `dataset.split` | - | `calendar` / `meeting` / `trip` |
| `dataset.n` | 4 | Sample limit |
| `dataset.prompt_mode` | 5shot | `5shot` / `0shot` |
| `dataset.stratified` | false | Stratified sampling |
| `dataset.sample_n` | 50 | Samples when stratified (50 cal, 50 meet, 48 trip) |
| `dataset.sample_seed` | 42 | Seed for stratified |
| `evaluate.expt_name` | - | Run tag |
| `ptools.{entry}.method` | - | `simulate` / `direct` / `prompt_llm` / `program_of_thought` / `simulate_pydantic` |

## Scripts

- `scripts/sample_ids.py` — export stratified IDs to `data/sampled_ids.json`
- `scripts/report.py` — table by task → report.md
- `scripts/analyze_wrong.py` — analyze wrong predictions
- `scripts/clean_test_runs.py` — delete test run dirs (cal_test, cal_trace_test, meet_trace_test)

**Prompt trace** 保存在每个 run 目录下：`results/.../YYYYMMDD.HHMMSS.expt_name/prompt_trace.jsonl`。需加 `evaluate.prompt_trace=true` 或 `run_all_15.py --trace` 才会生成。
