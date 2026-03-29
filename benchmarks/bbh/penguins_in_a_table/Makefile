#
# Run experiments - all on the validation data
#

# the hand-coded workflow
workflow:
	uv run expt.py run evaluate.expt_name=workflow \
	  ptools.answer_penguin_question.method=direct \
	  ptools.answer_penguin_question.fn=ptools.penguins_workflow

# program of thought
pot:
	uv run expt.py run evaluate.expt_name=pot \
	  ptools.answer_penguin_question.method=program_of_thought \
	  ptools.answer_penguin_question.tools=[ptools.analyze_input,ptools.table_operation,ptools.answer_question,ptools.choose_response]

# ReAct using simulate_pydantic (which runs ReAct if there are tools)
react:
	uv run expt.py run evaluate.expt_name=react \
	  ptools.answer_penguin_question.method=simulate_pydantic \
	  ptools.answer_penguin_question.tools=[ptools.analyze_input,ptools.table_operation,ptools.answer_question,ptools.choose_response]

# zero-shot structured queries
zeroshot_structured:
	uv run expt.py run evaluate.expt_name=zeroshot_structured \
	  ptools.answer_penguin_question.method=simulate

# zero-shot unstructured queries with a vanilla prompt
zeroshot_unstructured:
	uv run expt.py run evaluate.expt_name=zeroshot_unstructured \
	  ptools.answer_penguin_question.method=direct \
	  ptools.answer_penguin_question.fn=ptools.zeroshot_unstructured_workflow

#
# View results of past experiments
#

list:
	uv run python -m secretagent.cli.results list results/*

avg:
	uv run python -m secretagent.cli.results average --metric correct --metric cost results/*

pair:
	uv run python -m secretagent.cli.results pair --metric correct --metric cost results/*

compare:
	uv run python -m secretagent.cli.results compare-configs results/*

costs:
	uv run python -m secretagent.cli.costs llm_cache
