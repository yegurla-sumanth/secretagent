"""MUSR benchmark experiment.

Example CLI commands:

    # download data first (from project root)
    uv run benchmarks/musr/data/download.py

    # run murder mystery workflow (default config)
    uv run python benchmarks/musr/expt.py run --config-file conf/murder.yaml

    # run first 10 examples
    uv run python benchmarks/musr/expt.py run --config-file conf/murder.yaml dataset.n=10

    # zero-shot (override workflow to simulate)
    uv run python benchmarks/musr/expt.py run --config-file conf/murder.yaml ptools.answer_question.method=simulate evaluate.expt_name=murder_zeroshot

    # agent loop (program of thought)
    uv run python benchmarks/musr/expt.py run --config-file conf/murder.yaml ptools.answer_question.method=program_of_thought evaluate.expt_name=murder_agent

    # object placement
    uv run python benchmarks/musr/expt.py run --config-file conf/object.yaml

    # team allocation
    uv run python benchmarks/musr/expt.py run --config-file conf/team.yaml
"""

import ast
import importlib
import json
import sys
import pandas as pd
from pathlib import Path
from typing import Any

# Allow running from any directory: add project src/ and this benchmark dir to path
_BENCHMARK_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BENCHMARK_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
sys.path.insert(0, str(_BENCHMARK_DIR))

import re
import typer

from secretagent import config, record
from secretagent.core import Interface, implement_via_config
from secretagent.dataset import Dataset, Case
from secretagent.evaluate import Evaluator
import secretagent.implement_pydantic  # noqa: F401 (registers simulate_pydantic factory)

SPLIT_TO_MODULE = {
    'murder_mysteries': 'ptools_murder',
    'object_placements': 'ptools_object',
    'team_allocation': 'ptools_team',
}


def _patched_extract_answer(return_type, text, answer_pattern):
    """Drop-in replacement for implement_core._extract_answer with fallback.

    First tries the standard <answer> pattern. If that fails, tries:
    1. Bare integer on the last non-empty line
    2. Last integer found anywhere in the text
    """
    # Try standard extraction first
    if answer_pattern is not None:
        m = re.search(answer_pattern, text, re.DOTALL | re.MULTILINE)
        if m:
            val = m.group(1).strip()
            if return_type in (int, str, float):
                return return_type(val)
            return ast.literal_eval(val)

    if answer_pattern is None and return_type is str:
        return text.strip()

    # Fallback: bare integer on last non-empty line
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if re.fullmatch(r'\d+', line):
            return int(line)

    # Fallback: last integer in the text
    ints = re.findall(r'\b(\d+)\b', text)
    if ints:
        return int(ints[-1])

    raise ValueError(f'cannot find answer in LLM output')


# Monkey-patch the framework's parsers with our fallback version
from secretagent import implement_core

# Patch SimulateFactory.parse_output (method, so lookup works)
def _patched_parse_output(self, return_type, text):
    return _patched_extract_answer(return_type, text, r'<answer>(.*)</answer>')
implement_core.SimulateFactory.parse_output = _patched_parse_output

# Patch PromptLLMFactory: must replace the module-level ref used in closures
# by patching build_fn itself
_orig_prompt_llm_build_fn = implement_core.PromptLLMFactory.build_fn
def _patched_prompt_llm_build_fn(self, interface, prompt_template_str=None,
                                  prompt_template_file=None, answer_pattern=None,
                                  **prompt_kw):
    from string import Template
    from textwrap import dedent
    import pathlib
    from secretagent import config, llm_util, record as rec_mod
    if (prompt_template_str is None) == (prompt_template_file is None):
        raise ValueError('Exactly one of prompt_template_str or prompt_template_file must be given')
    if prompt_template_file is not None:
        prompt_template_str = pathlib.Path(prompt_template_file).read_text()
    template = Template(dedent(prompt_template_str))
    def result_fn(*args, **kw):
        with config.configuration(**prompt_kw):
            arg_names = list(interface.annotations.keys())[:-1]
            arg_dict = dict(zip(arg_names, args))
            arg_dict.update(kw)
            prompt = template.substitute(arg_dict)
            llm_output, stats = llm_util.llm(prompt, config.require('llm.model'))
            return_type = interface.annotations.get('return', str)
            answer = _patched_extract_answer(return_type, llm_output, answer_pattern)
            rec_mod.record(func=interface.name, args=args, kw=kw, output=answer, stats=stats)
            return answer
    return result_fn
implement_core.PromptLLMFactory.build_fn = _patched_prompt_llm_build_fn


class MUSREvaluator(Evaluator):
    def compare_predictions(self, predicted_output, expected_output) -> dict[str, Any]:
        return dict(correct=(predicted_output == expected_output))

    def measure(self, example: Case, interface: Interface) -> dict[str, Any]:
        """Measure with error handling — parse failures count as incorrect."""
        try:
            return super().measure(example, interface)
        except Exception as e:
            print(f'  [error on {example.name}: {type(e).__name__}]')
            return dict(
                predicted_output=None,
                expected_output=example.expected_output,
                correct=False,
                input_tokens=0, output_tokens=0, latency=0, cost=0)


def load_dataset(split: str) -> Dataset:
    json_file = Path(__file__).parent / 'data' / f'{split}.json'
    with open(json_file) as f:
        data = json.load(f)

    cases = []
    for i, ex in enumerate(data['examples']):
        choices = ex['choices']
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)
        cases.append(Case(
            name=f'ex{i:03d}',
            input_args=(ex['narrative'], ex['question'], choices),
            expected_output=ex['answer_index'],
        ))

    return Dataset(name='musr', split=split, cases=cases)


app = typer.Typer()

@app.callback()
def callback():
    """MUSR benchmark."""

@app.command(context_settings={"allow_extra_args": True, "allow_interspersed_args": False})
def run(ctx: typer.Context,
        config_file: str = typer.Option(..., help="Config YAML file")):
    """Run MUSR evaluation. Extra args are config overrides in dot notation."""

    # Resolve config_file relative to benchmark dir if not absolute
    cfg_path = Path(config_file)
    if not cfg_path.is_absolute():
        cfg_path = _BENCHMARK_DIR / cfg_path
    config.configure(yaml_file=str(cfg_path), dotlist=ctx.args)
    config.set_root(_BENCHMARK_DIR)

    split = config.require('dataset.split')
    ptools = importlib.import_module(SPLIT_TO_MODULE[split])
    implement_via_config(ptools, config.require('ptools'))

    dataset = load_dataset(split).configure(
        shuffle_seed=config.get('dataset.shuffle_seed'),
        n=config.get('dataset.n'))
    print('dataset:', dataset.summary())

    entry_point = config.require('evaluate.entry_point')
    interface = getattr(ptools, entry_point)

    evaluator = MUSREvaluator()
    results = evaluator.evaluate(dataset, interface)
    df = pd.DataFrame(results)
    print(df)
    print(f'\nAccuracy: {df["correct"].mean():.1%}')


if __name__ == '__main__':
    app()
