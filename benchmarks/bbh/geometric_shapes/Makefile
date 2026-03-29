#
# Run experiments - all on the validation data
#

# the hand-coded workflow
workflow:
	uv run expt.py run evaluate.expt_name=workflow \
	  ptools.identify_shape.method=direct \
	  ptools.identify_shape.fn=ptools.geometric_shapes_workflow dataset.n=3

# quick test of the workflow on a single example (override with example_name=valid.003)
quick-test:
	uv run expt.py quick-test \
	  ptools.identify_shape.method=direct \
	  ptools.identify_shape.fn=ptools.geometric_shapes_workflow \
	  $(if $(example_name),--example-name $(example_name),)

# program of thoughts
pot:
	uv run expt.py run evaluate.expt_name=pot \
	  ptools.identify_shape.method=program_of_thought \
	  "ptools.identify_shape.tools=[ptools.extract_path,ptools.extract_options,ptools.explain_path,ptools.decompose_path,ptools.summarize_decomposed_path,ptools.summary_matches_option,ptools.compute_length_clusters,ptools.relate_length_clusters_to_option,ptools.length_clusters_match_option,ptools.is_unique_answer]"

# ReAct using simulate_pydantic (which runs ReAct if there are tools)
react:
	uv run expt.py run evaluate.expt_name=react \
	  ptools.identify_shape.method=simulate_pydantic \
	  "ptools.identify_shape.tools=[ptools.extract_path,ptools.extract_options,ptools.explain_path,ptools.decompose_path,ptools.summarize_decomposed_path,ptools.summary_matches_option,ptools.compute_length_clusters,ptools.relate_length_clusters_to_option,ptools.length_clusters_match_option,ptools.is_unique_answer]"

# zero-shot structured queries
zeroshot_structured:
	uv run expt.py run evaluate.expt_name=zeroshot_structured \
	  ptools.identify_shape.method=simulate

# zero-shot unstructured queries with a vanilla prompt
zeroshot_unstructured:
	uv run expt.py run evaluate.expt_name=zeroshot_unstructured \
	  ptools.identify_shape.method=direct \
	  ptools.identify_shape.fn=ptools.zeroshot_unstructured_workflow

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
