"""CLI for analyzing experiment results saved by savefile.

Commands:
    list      Show available experiments and number of examples
    average   Report mean +/- stderr of metrics grouped by experiment.
    pair      Run paired t-tests on metrics across experiments.
    compare   Show configuration differences between experiments.

Experiment directories (or CSV files within them) are passed as extra
positional arguments after the subcommand.  Results are filtered through
savefile.filter_paths(), which supports ``--latest`` (keep k most recent
per tag) and ``--check`` (config-key constraints).

Example usage:

    # List all experiments under a directory
    uv run -m secretagent.cli.results list results/20260301.* --latest 0

    # List only the most recent experiment per tag (default)
    uv run -m secretagent.cli.results list results/*

    # Show averages with a config constraint
    uv run -m secretagent.cli.results average results/* --check llm.model=gpt-4o

    # Paired t-test between experiments
    uv run -m secretagent.cli.results pair results/* --latest 0

    # Compare configs across experiments
    uv run -m secretagent.cli.results compare results/* --latest 0
"""

from collections import Counter
import shutil
import typer
import pandas as pd
from itertools import combinations
from pathlib import Path
from typing import Optional

from scipy import stats as scipy_stats

from secretagent import config, savefile

app = typer.Typer()

_EXTRA_ARGS = {"allow_extra_args": True, "allow_interspersed_args": False}


def _get_dirs(ctx: typer.Context, latest: int = 1, check: Optional[list[str]] = None) -> list[Path]:
    """Resolve experiment directories from extra CLI args.

    Extra args in ctx.args are files or directories.
    files are mapped to their parent directory.
    Results are filtered through savefile.filter_paths.
    """
    if not ctx.args:
        raise ValueError('No paths provided.')
    dirs = savefile.filter_paths(ctx.args, latest=latest, dotlist=check or [])
    if not dirs:
        raise ValueError('No matching experiment directories found.')
    for d in dirs:
        if not (Path(d) / 'results.csv').exists():
            raise ValueError(f'No results.csv in {d}')
    return dirs



@app.command('list', context_settings=_EXTRA_ARGS)
def list_experiments(
    ctx: typer.Context,
    latest: int = typer.Option(1, help='Keep latest k dirs per tag; 0 for all'),
    check: Optional[list[str]] = typer.Option(None, help='Config constraint like key=value'),
):
    """Show available experiment directories and row counts.

    Extra args are CSV files or directories to inspect.
    """
    dirs = _get_dirs(ctx, latest=latest, check=check)
    for d in dirs:
        csv_path = d / 'results.csv'
        n = len(pd.read_csv(csv_path))
        print(f'{n:5d} examples in {d.name}')


@app.command(context_settings=_EXTRA_ARGS)
def average(
        ctx: typer.Context,
        latest: int = typer.Option(1, help='Keep latest k dirs per tag; 0 for all'),
        check: Optional[list[str]] = typer.Option(None, help='Config constraint like key=value'),
        metric: Optional[list[str]] = typer.Option(None, help='Metrics to average'),
):
    """Report mean +/- stderr of a metric and latency, grouped by experiment."""
    if metric is None:
        metric = ['cost']
    dirs = _get_dirs(ctx, latest=latest, check=check)
    dfs = [pd.read_csv(d / 'results.csv') for d in dirs]
    for path, df in zip(dirs, dfs):
        df['path'] = [str(path)] * len(df)
    result = pd.concat(dfs).groupby('path')[metric].agg(['mean', 'sem'])
    result = result.sort_values((metric[0], 'mean'), ascending=False)
    print(result)


@app.command(context_settings=_EXTRA_ARGS)
def pair(
    ctx: typer.Context,
    latest: int = typer.Option(1, help='Keep latest k dirs per tag; 0 for all'),
    check: Optional[list[str]] = typer.Option(None, help='Config constraint like key=value'),
    metric: Optional[list[str]] = typer.Option(None, help='Metrics to pair'),
):
    """Run paired t-tests across experiments."""
    if not metric:
        raise ValueError('At least one --metric is required for paired comparison')
    dirs = _get_dirs(ctx, latest=latest, check=check)
    dfs = [pd.read_csv(d / 'results.csv') for d in dirs]
    if len(dfs) < 2:
        raise ValueError('Need at least 2 experiments for paired comparison')
    # Sort by mean of the first metric so comparisons read in order
    paired = sorted(zip(dirs, dfs), key=lambda x: x[1][metric[0]].mean(), reverse=True)
    dirs, dfs = [list(t) for t in zip(*paired)]
    rows = []
    for (pa, da), (pb, db) in combinations(zip(dirs, dfs), 2):
        fua = savefile.file_under_part(pa)
        fub = savefile.file_under_part(pb)
        joined = da.join(db, lsuffix='_a', rsuffix='_b', how='inner')
        n = len(joined)
        if n == 0:
            print(f"\n{pa} vs {pb}: <2 shared example_ids, skipping")
            continue
        row = {
            '_comparison': f'{fua} vs {fub}',
            'n': n}
        for m in (metric or []):
            t, p = scipy_stats.ttest_rel(joined[f'{m}_a'], joined[f'{m}_b'])
            row[f'{m}_t'] = t
            row[f'{m}_p'] = p
        rows.append(row)
    df = pd.DataFrame(rows)
    p_cols = [c for c in df.columns if c.endswith('_p')]
    for c in p_cols:
        df[c] = df[c].map(lambda x: f'{x:.5f}')
    print(df)
            

@app.command(context_settings=_EXTRA_ARGS)
def compare_configs(
    ctx: typer.Context,
    latest: int = typer.Option(1, help='Keep latest k dirs per tag; 0 for all'),
    check: Optional[list[str]] = typer.Option(None, help='Config constraint like key=value'),
):
    """Show configuration differences between experiments."""
    dirs = _get_dirs(ctx, latest=latest, check=check)
    ctr: Counter[str] = Counter()
    all_pairs = {}
    for d in dirs:
        cfg = config.load_yaml_cfg(d / 'config.yaml')
        all_pairs[d] = config.to_dotlist(cfg)
        for pair in all_pairs[d]:
            ctr[pair] += 1
    for d, pairs in all_pairs.items():
        uncommon_pairs = [
            pair for pair in pairs
            if ctr[pair] < len(dirs)
        ]
        print(f'properties of {d}:')
        for p in uncommon_pairs:
            print(f'  {p}')


_DEFAULT_REQUIRED_FILES = ['config.yaml', 'results.csv']


@app.command(context_settings=_EXTRA_ARGS)
def validate(
    ctx: typer.Context,
    latest: int = typer.Option(1, help='Keep latest k dirs per tag; 0 for all'),
    check: Optional[list[str]] = typer.Option(None, help='Config constraint like key=value'),
    require: Optional[list[str]] = typer.Option(None, help='Additional required files'),
    norequire: Optional[list[str]] = typer.Option(None, help='Remove a default required file'),
    purge: bool = typer.Option(False, help='Delete directories that fail validation'),
):
    """Check that experiment directories contain required files."""
    required = set(_DEFAULT_REQUIRED_FILES)
    for f in (require or []):
        required.add(f)
    for f in (norequire or []):
        required.discard(f)
    dirs = savefile.filter_paths(ctx.args, latest=latest, dotlist=check or [])
    bad_dirs = []
    for d in dirs:
        missing = [f for f in sorted(required) if not (d / f).exists()]
        if missing:
            print(f'{d}: missing {", ".join(missing)}')
            bad_dirs.append(d)
    if not bad_dirs:
        print('All directories OK.')
        return
    if purge:
        answer = input(f'Delete {len(bad_dirs)} directories? [y/N] ')
        if answer.strip().lower() == 'y':
            for d in bad_dirs:
                shutil.rmtree(d)
                print(f'  deleted {d}')


@app.command(context_settings=_EXTRA_ARGS)
def delete_obsolete(
    ctx: typer.Context,
    latest: int = typer.Option(1, help='Keep latest k dirs per tag; 0 for all'),
    check: Optional[list[str]] = typer.Option(None, help='Config constraint like key=value'),
):
    """Delete experiment directories not retained by filter_paths."""
    all_dirs = {Path(p) for p in ctx.args if Path(p).is_dir()}
    keep = set(savefile.filter_paths(ctx.args, latest=latest, dotlist=check or []))
    to_delete = sorted(all_dirs - keep)
    if not to_delete:
        print('Nothing to delete.')
        return
    print('Keeping:')
    for d in sorted(keep):
        print(f'  {d}')
    print(f'\nWill delete {len(to_delete)} directories:')
    for d in to_delete:
        print(f'  {d}')
    answer = input('Proceed? [y/N] ')
    if answer.strip().lower() == 'y':
        for d in to_delete:
            shutil.rmtree(d)
            print(f'  deleted {d}')


@app.callback()
def main(
    config_file: Optional[str] = typer.Option(None, help='YAML config file to load'),
):
    """Analyze experiment results saved by savefile.
    """
    if config_file:
        config.configure(yaml_file=config_file)


if __name__ == '__main__':
    app()
