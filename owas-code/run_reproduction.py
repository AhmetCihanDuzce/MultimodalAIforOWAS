"""One-command offline numerical reproduction for the two-part public release."""
from importlib.metadata import version
from pathlib import Path
import json
import os
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'owas-data-results'


def main():
    if sys.version_info[:2] != (3, 12):
        raise SystemExit('Use Python 3.12 for the pinned reproduction environment.')
    if not (DATA / 'inputs' / 'ai_coding_input.csv').is_file():
        raise SystemExit('Extract both ZIP archives into the same parent directory: owas-code and owas-data-results must be siblings.')
    manifest_paths = [ROOT / 'MANIFEST.json', DATA / 'MANIFEST.json']
    if all(path.is_file() for path in manifest_paths):
        subprocess.run([sys.executable, str(ROOT / 'verify_checksums.py')], cwd=ROOT, check=True)
    else:
        raise SystemExit('Missing release manifests. Extract both complete release ZIPs into the same parent directory.')
    dependencies = {}
    for line in (ROOT / 'requirements.txt').read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        package, expected = line.split('==', 1)
        actual = version(package)
        dependencies[package] = actual
        if actual != expected:
            raise SystemExit(f'Version mismatch for {package}: installed {actual}, required {expected}. Install requirements.txt in a Python 3.12 virtual environment.')
    cache = ROOT / 'generated' / '.cache'
    cache.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['XDG_CACHE_HOME'] = str(cache)
    env['MPLCONFIGDIR'] = str(cache / 'matplotlib')
    (cache / 'fontconfig').mkdir(exist_ok=True)
    (cache / 'matplotlib').mkdir(exist_ok=True)
    env.update({'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'MPLBACKEND': 'Agg'})
    for script in ['analyse_study.py', 'independent_checks.py', 'audit_protocol.py', 'make_figures.py', 'verify_release.py']:
        print(f'Running {script}', flush=True)
        subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, env=env, check=True)
    figures = sorted((ROOT / 'generated' / 'figures').glob('*.png'))
    assert len(figures) == 4 and all(path.stat().st_size > 0 for path in figures)
    environment = {
        'python': platform.python_version(),
        'dependencies': dependencies,
        'thread_environment': {key: env[key] for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']},
        'figure_count': len(figures),
        'no_ai_api_calls': True,
        'statistical_inputs': 'Released deidentified coding and consensus CSVs.',
    }
    (ROOT / 'generated' / 'results' / 'runtime_environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    print('PASS: all released scientific CSV tables reproduced; four numerical figures regenerated.', flush=True)


if __name__ == '__main__':
    main()
