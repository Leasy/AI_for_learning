"""Write the LaTeX and CSV versions of a numbered paper table."""
import csv
from pathlib import Path
import re


TEMPLATES = Path(__file__).resolve().parent / 'table_templates'


def render_table(tables_dir, stem, rows_tex, csv_rows, *, fields=None):
    template = (TEMPLATES / f'{stem}.tex').read_text(encoding='utf-8')
    rows = rows_tex if isinstance(rows_tex, str) else '\n'.join(rows_tex)
    values = {'ROWS': rows, **(fields or {})}
    latex = re.sub(r'@@([A-Z][A-Z0-9_]*)@@', lambda m: str(values[m[1]]), template)
    (tables_dir / f'{stem}.tex').write_text(latex, encoding='utf-8', newline='\n')
    with (tables_dir / f'{stem}.csv').open('w', newline='', encoding='utf-8') as handle:
        csv.writer(handle, lineterminator='\n').writerows(csv_rows)
