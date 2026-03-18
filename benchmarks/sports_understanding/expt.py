"""Sports understanding benchmark experiment.

Example CLI commands:

    # run with defaults from conf/conf.yaml
    uv run python expt.py run

    # run first 6 examples only
    uv run python expt.py run dataset.n=6

    # override model and experiment name
    uv run python expt.py run llm.model=gpt-4o evaluate.expt_name=gpt4o_test

    # use a different config file
    uv run python expt.py run --config-file conf/ablation.yaml

    # record detailed results with rollouts
    uv run python expt.py run evaluate.record_details=true dataset.n=6
"""

import json
import pandas as pd
from pathlib import Path
import pprint
import re
from typing import Any

import typer

from secretagent import record, config
from secretagent.core import implement_via_config
import secretagent.implement_pydantic  # noqa: F401 (registers simulate_pydantic factory)
from secretagent.dataset import Dataset, Case
from secretagent.evaluate import Evaluator

#
# tools are the tools and interfaces
#

import ptools

#
# dataset-specific metrics
#

class SportsUnderstandingEvaluator(Evaluator):
    def compare_predictions(self, predicted_output, expected_output) -> dict[str, Any]:
        return dict(correct=float(predicted_output == expected_output))

#
# how to load the dataset
#

def load_dataset(split: str) -> Dataset:
    def example_as_case(index, split, example):
        return Case(
            name=f'{split}.{index:03d}',
            input_args=(re.search(r'"([^"]*)"', example['input']).group(1),),
            expected_output=(example['target']=="yes")
        )
    json_file = Path(__file__).parent / 'data' / f'{split}.json'
    with open(json_file) as fp:
        data = json.load(fp)
        examples = data['examples']
        return Dataset(
            name='sports_understanding',
            split=split,
            cases=[
                example_as_case(i, split, ex)
                for i, ex in enumerate(examples)
            ],
        )

#
# shared setup logic
#

_EXTRA_ARGS = {"allow_extra_args": True, "allow_interspersed_args": False}

def setup(ctx: typer.Context, config_file: Path | None = None):
    """Load config, dataset, and configure ptools.

    Returns (dataset, interface) ready for evaluation.
    """
    if config_file is None:
        config_file = Path(__file__).parent / 'conf' / 'conf.yaml'
    config.configure(yaml_file=config_file, dotlist=ctx.args)
    config.set_root(Path(__file__).parent)

    dataset = load_dataset(config.require('dataset.split')).configure(
        shuffle_seed=config.get('dataset.shuffle_seed'),
        n=config.get('dataset.n') or None  # don't pass in 0
        )
    print('dataset is', dataset.summary())

    implement_via_config(ptools, config.require('ptools'))
    return dataset, ptools.are_sports_in_sentence_consistent

#
# machinery to support using this file as a CLI
#

app = typer.Typer()

@app.callback()
def callback():
    """Sports understanding benchmark.

    This callback ensures typer treats the app as a multi-command CLI
    rather than collapsing a single subcommand to the top level.
    """

@app.command(context_settings=_EXTRA_ARGS)
def run(ctx: typer.Context,
        config_file: Path = typer.Option(None, help="Config YAML file")):
    """Run sports understanding evaluation.

    Extra args are parsed as config overrides in dot notation, e.g.:
        uv run python expt.py run llm.model=gpt-4o cachier.enable_caching=false
    """
    dataset, interface = setup(ctx, config_file)

    evaluator = SportsUnderstandingEvaluator()
    csv_path = evaluator.evaluate(dataset, interface)

    # print a summary
    df = pd.read_csv(csv_path)
    print(df)
    print()
    print(df.select_dtypes(include='number').mean())


@app.command(context_settings=_EXTRA_ARGS)
def quick_test(ctx: typer.Context,
               config_file: Path = typer.Option(None, help="Config YAML file")):
    """Do a quick test of a configuration.

    Configures and loads data as in the run command, but just runs the
    top-level interface on a single example, tracing as much as
    possible.
    """
    dataset, interface = setup(ctx, config_file)
    pprint.pprint(config.GLOBAL_CONFIG)

    input_args = dataset.cases[0].input_args
    print('input_args', input_args)
    with config.configuration(
            cachier={'enable_caching':False},
            echo={
                'service': True, 'llm_input': True, 'llm_output': True, 'code_eval_input': True, 'code_eval_output': True}):
        with record.recorder() as records:
            predicted_output = interface(*input_args)

    print('predicted output', predicted_output)
    pprint.pprint(records)



if __name__ == '__main__':
    app()
