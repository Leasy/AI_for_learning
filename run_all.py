"""Reproduce the analysis, the five manuscript tables, and the in-text results."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))


def json_safe(value):
    """Keep machine metadata valid JSON, including unbounded confidence sets."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, 'item'):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else ('+infinity' if value > 0 else '-infinity')
    return value


def find_rscript(requested):
    """Locate Rscript."""
    if requested:
        candidate = shutil.which(requested)
        if candidate:
            return Path(candidate).resolve()
        candidate = Path(requested).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f'Rscript was not found: {requested}')
    candidate = shutil.which('Rscript')
    if candidate:
        return Path(candidate).resolve()
    if os.name == 'nt':
        installs = []
        for variable in ('ProgramFiles', 'ProgramFiles(x86)', 'LOCALAPPDATA'):
            base = os.environ.get(variable)
            if not base:
                continue
            for directory in (Path(base) / 'R', Path(base) / 'Programs' / 'R'):
                installs.extend(directory.glob('R-*/bin/Rscript.exe'))
        if installs:
            import re
            return max(installs, key=lambda p: tuple(
                int(number) for number in re.findall(r'\d+', p.parent.parent.name))).resolve()
    raise FileNotFoundError(
        'Rscript was not found. Install R, add its bin directory to PATH, or use '
        '--rscript "C:\\Program Files\\R\\R-4.5.2\\bin\\Rscript.exe". '
        'Base R is required.')


def publish_results(stage, output):
    """Publish completed results, preserving other files in an existing output directory."""
    if not output.exists():
        stage.rename(output)
        return

    # If a workbook is open in Excel on Windows, a replacement can fail. In that
    # case restore any earlier files and leave the previous complete run in place.
    previous = stage.parent / 'previous'
    installed = []
    files = sorted(path.relative_to(stage) for path in stage.rglob('*') if path.is_file())
    manifest = Path('run_info.json')
    files.remove(manifest)
    try:
        for relative in [*files, manifest]:
            target = output / relative
            backup = previous / relative
            if target.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
            target.parent.mkdir(parents=True, exist_ok=True)
            (stage / relative).replace(target)
            installed.append(relative)
    except OSError:
        for relative in reversed(installed):
            backup = previous / relative
            target = output / relative
            if backup.exists():
                backup.replace(target)
            else:
                target.unlink()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean-output', type=Path,
                        default=ROOT / 'data' / 'clean' / 'analysis_sessions.csv',
                        help='Complete regression CSV produced by Python.')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results',
                        help='Directory for all regression outputs and paper tables; defaults to results/.')
    parser.add_argument('--rscript', help='Rscript executable; otherwise found on PATH or in an existing Windows R installation.')
    args = parser.parse_args()
    rscript = find_rscript(args.rscript)
    from clean_data import clean_data

    from paper_results import TABLES, write_paper_results

    clean_output = args.clean_output.resolve()
    output = args.output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    print('Step 1/3: reconstructing the analysis data and saved dialogue scores...', flush=True)
    data_metadata = clean_data(ROOT, clean_output)
    with tempfile.TemporaryDirectory(prefix='.run_all_', dir=output.parent) as working:
        stage = Path(working) / 'results'
        print('Step 2/3: estimating the four specifications in R...', flush=True)
        subprocess.run([
            str(rscript), '--vanilla', str(ROOT / 'code' / 'run_regressions.R'),
            '--input', str(clean_output), '--output-dir', str(stage),
        ], check=True)
        r_info = stage / 'regression_info.json'
        regression_info = json.loads(r_info.read_text(encoding='utf-8-sig'))
        if regression_info.get('status') != 'passed':
            raise RuntimeError('R did not report successful regression checks.')
        regression_info['output_dir'] = output.as_posix()
        r_info.write_text(json.dumps(regression_info, indent=2, ensure_ascii=False) + '\n',
                          encoding='utf-8')

        print('Step 3/3: generating the numbered tables and in-text results...', flush=True)
        write_paper_results(clean_output, data_metadata, stage)
        info = {
            'started_utc': started,
            'completed_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'passed',
            'pipeline': 'Python cleaning -> R regressions -> manuscript tables and text results',
            'regression_input': str(clean_output),
            'data': data_metadata,
            'estimation': regression_info,
            'paper_tables': {number: [f'tables/{stem}.tex', f'tables/{stem}.csv']
                             for number, stem, _ in TABLES},
            'paper_text_results': 'paper_text_results.csv',
            'environment': {'python': platform.python_version(), 'Rscript': str(rscript)},
            'model_requests': 0,
        }
        (stage / 'run_info.json').write_text(
            json.dumps(json_safe(info), indent=2, ensure_ascii=False, allow_nan=False) + '\n',
            encoding='utf-8',
        )
        publish_results(stage, output)
    print(f'Paper results: {output / "README.md"}')
    print(f'Tables 1, 2, 3, A1 and A2: {output / "tables"}')
    print(f'Numbers quoted in the text: {output / "paper_text_results.csv"}')


if __name__ == '__main__':
    main()
