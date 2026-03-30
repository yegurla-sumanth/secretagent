# Path to the RuleArena repo clone — only needed to run 'make prepare'
RULEARENA_DIR ?= ../../../RuleArena

#
# One-time data preparation (run before any experiment)
#

prepare:
	uv run python data/prepare.py $(RULEARENA_DIR)

#
# L0: Oracle baseline (Python calculators with ground-truth parameters, no LLM)
#

l0_airline:
	uv run expt.py run evaluate.expt_name=l0_airline dataset.domain=airline \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.l0_oracle_workflow

l0_nba:
	uv run expt.py run evaluate.expt_name=l0_nba dataset.domain=nba \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.l0_oracle_workflow

l0_tax:
	uv run expt.py run evaluate.expt_name=l0_tax dataset.domain=tax \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.l0_oracle_workflow

#
# L0F: Chain-of-thought baseline (single LLM call per instance, no structure)
#

l0f_airline:
	uv run expt.py run evaluate.expt_name=l0f_airline dataset.domain=airline \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.l0f_cot_workflow

l0f_nba:
	uv run expt.py run evaluate.expt_name=l0f_nba dataset.domain=nba \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.l0f_cot_workflow

l0f_tax:
	uv run expt.py run evaluate.expt_name=l0f_tax dataset.domain=tax \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.l0f_cot_workflow

#
# L1: PTool extraction + Python calculation
#

l1_airline:
	uv run expt.py run evaluate.expt_name=l1_airline dataset.domain=airline \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.l1_extract_workflow

l1_nba:
	uv run expt.py run evaluate.expt_name=l1_nba dataset.domain=nba \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.l1_extract_workflow

l1_tax:
	uv run expt.py run evaluate.expt_name=l1_tax dataset.domain=tax \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.l1_extract_workflow

#
# L3: ReAct autonomous agent (simulate_pydantic with tools)
#

l3_airline:
	uv run expt.py run evaluate.expt_name=l3_airline dataset.domain=airline \
	  ptools.compute_rulearena_answer.method=simulate_pydantic \
	  "ptools.compute_rulearena_answer.tools=[ptools.extract_airline_params,ptools.compute_airline_calculator]"

l3_nba:
	uv run expt.py run evaluate.expt_name=l3_nba dataset.domain=nba \
	  ptools.compute_rulearena_answer.method=simulate_pydantic \
	  "ptools.compute_rulearena_answer.tools=[ptools.extract_nba_params]"

l3_tax:
	uv run expt.py run evaluate.expt_name=l3_tax dataset.domain=tax \
	  ptools.compute_rulearena_answer.method=simulate_pydantic \
	  "ptools.compute_rulearena_answer.tools=[ptools.extract_tax_params,ptools.compute_tax_calculator]"

#
# Learning pipeline (airline domain, l1 extract workflow)
#

RECORD_OPTS = dataset.split=train evaluate.record_details=true evaluate.result_dir=recordings

learn_pipeline: record_l1_airline learn_extract_airline evaluate_learned_l1_airline avg

record_l1_airline:
	uv run expt.py run evaluate.expt_name=l1_airline dataset.domain=airline \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.l1_extract_workflow \
	  $(RECORD_OPTS)

learn_extract_airline:
	uv run python -m secretagent.cli.learn rote --interface extract_airline_params \
	  --learned-dir training recordings/*

evaluate_learned_l1_airline:
	uv run expt.py run evaluate.expt_name=l1_airline_learned dataset.domain=airline \
	  learn.train_dir=training \
	  ptools.extract_airline_params.method=learned_code \
	  ptools.extract_airline_params.learner=rote \
	  ptools.extract_airline_params.backoff=True \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.l1_extract_workflow

#
# View results of past experiments
#

list:
	uv run python -m secretagent.cli.results list results/*

avg:
	uv run python -m secretagent.cli.results average --metric correct --metric correct_tolerance --metric cost results/*

pair:
	uv run python -m secretagent.cli.results pair --metric correct --metric cost results/*

compare:
	uv run python -m secretagent.cli.results compare-configs results/*

costs:
	uv run python -m secretagent.cli.costs llm_cache
