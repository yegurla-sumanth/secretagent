"""Sports understanding benchmark experiment.

Example CLI commands:

    # run with defaults from conf/conf.yaml
    uv run python expt.py run

    # run first 6 examples only
    uv run python expt.py run dataset.n=6

    # override model and experiment name
    uv run python expt.py run llm.model=gpt-4o evaluate.expt_name=gpt4o_test

    # record detailed results with rollouts
    uv run python expt.py run evaluate.record_details=true dataset.n=6
"""

import pandas as pd
from pathlib import Path
import pprint

import typer

from secretagent import record, config
from secretagent.core import implement_via_config, Interface
import secretagent.implement_pydantic  # noqa: F401 (registers simulate_pydantic factory)
import secretagent.learn.implement_learn  # noqa: F401 (registers learned factory)
from secretagent.dataset import Dataset
from secretagent.evaluate import ExactMatchEvaluator, Evaluator

#
# tools are the tools and interfaces
#

import ptools

#
# shared setup logic
#

_EXTRA_ARGS = {"allow_extra_args": True, "allow_interspersed_args": False}

def setup_and_load_dataset(dotlist: list[str]) -> Dataset:
    """Load config, dataset, and configure ptools.

    Returns dataset ready for evaluation.
    """
    config_file = Path(__file__).parent / 'conf' / 'conf.yaml'
    config.configure(yaml_file=config_file, dotlist=dotlist)
    config.set_root(Path(__file__).parent)

    split = config.require('dataset.split')
    dataset_json_file = Path(__file__).parent / 'data' / f'{split}.json'
    dataset = Dataset.model_validate_json(dataset_json_file.read_text())
    dataset.configure(
        shuffle_seed=config.get('dataset.shuffle_seed'),
        n=config.get('dataset.n') or None  # don't pass in 0
    )
    implement_via_config(ptools, config.require('ptools'))
    return dataset 

def run_experiment(
        top_level_interface: Interface, 
        dotlist: list[str] | None = None,
        evaluator: Evaluator | None = None
) -> pd.DataFrame:
    dataset = setup_and_load_dataset(dotlist or [])
    evaluator = evaluator or ExactMatchEvaluator()
    csv_path = evaluator.evaluate(dataset, top_level_interface)
    # print a summary
    df = pd.read_csv(csv_path)
    print(df)
    print()
    print(df.select_dtypes(include='number').mean())
    return df

#
# machinery to support using this file as a CLI
#

app = typer.Typer()

@app.command(context_settings=_EXTRA_ARGS)
def run(ctx: typer.Context):
    """Run sports understanding evaluation.

    Extra args are parsed as config overrides in dot notation, e.g.:
        uv run python expt.py run llm.model=gpt-4o cachier.enable_caching=false
    """
    run_experiment(
        top_level_interface=ptools.are_sports_in_sentence_consistent,
        dotlist=ctx.args)


@app.command(context_settings=_EXTRA_ARGS)
def quick_test(ctx: typer.Context):
    """Do a quick test of a configuration.

    Configures and loads data as in the run command, but just runs the
    top-level interface on a single example, tracing as much as
    possible.
    """
    dataset = setup(ctx.args)
    print('dataset is', dataset.summary())
    pprint.pprint(config.GLOBAL_CONFIG)

    input_args = dataset.cases[0].input_args
    print('input_args', input_args)
    with config.configuration(
            cachier={'enable_caching': False},
            echo={
                'service': True,
                'llm_input': True, 'llm_output': True, 
                'code_eval_input': True, 'code_eval_output': True}
    ):
        with record.recorder() as records:
            predicted_output = interface(*input_args)
    print('predicted output', predicted_output)
    pprint.pprint(records)


if __name__ == '__main__':
    app()
