"""Produce the numbered tables and the numerical results quoted in the manuscript."""
import csv
import shutil

import pandas as pd

from paper_descriptives import write_descriptive_tables
from paper_main_table import write_main_table
from paper_text_results import write_text_results


TABLES = (
    ('1', 'table_1_sample', 'Construction of the estimation sample'),
    ('2', 'table_2_summary', 'Summary statistics'),
    ('3', 'table_3_main', 'Safeguards, student dialogue, and the unassisted exam score'),
    ('A1', 'table_a1_rubric', 'Semantic scoring rubric'),
    ('A2', 'table_a2_scores', 'Distribution of first-message scores'),
)


def write_paper_results(clean_csv, metadata, output):
    panel = pd.read_csv(clean_csv, dtype={'class_id': str, 'teacher_id': str, 'grader_id': str})
    estimates = pd.read_csv(output / 'main_estimates.csv')
    reduced_forms = pd.read_csv(output / 'reduced_form_coefficients.csv')
    tables_dir = output / 'tables'
    tables_dir.mkdir(exist_ok=True)
    write_descriptive_tables(panel, metadata, tables_dir)
    write_main_table(panel, estimates, reduced_forms, tables_dir)
    write_text_results(panel, metadata, estimates, reduced_forms, output)

    # Keep the top-level main table in the same order and precision as Table 3.
    for extension in ('csv', 'tex'):
        shutil.copyfile(tables_dir / f'table_3_main.{extension}', output / f'main_table.{extension}')
    with (output / 'main_table.csv').open(encoding='utf-8', newline='') as handle:
        header, *rows = csv.reader(handle)
    markdown = ['# Table 3. Safeguards, student dialogue, and the unassisted exam score', '',
                '| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] * len(header)) + ' |']
    markdown.extend('| ' + ' | '.join(row) + ' |' for row in rows)
    (output / 'main_table.md').write_text('\n'.join(markdown) + '\n', encoding='utf-8', newline='\n')

    index = ['# Paper results', '', '| Table | Title | LaTeX | CSV |',
             '| --- | --- | --- | --- |']
    for number, stem, title in TABLES:
        tex = f'tables/{stem}.tex'
        csv_path = f'tables/{stem}.csv'
        index.append(f'| {number} | {title} | [LaTeX]({tex}) | [CSV]({csv_path}) |')
    index.extend(['', '[Numerical results quoted in the text](paper_text_results.csv) include '
                  'units, standardized effects, sample retention, and the direct GPA share '
                  'at which the residual is zero.',
                  '', '[Full-precision estimates](main_estimates.csv) and '
                  '[reduced-form coefficients](reduced_form_coefficients.csv) report '
                  'the underlying regressions. The [detailed regression output](regression_diagnostics.md) '
                  'includes additional inference diagnostics.', ''])
    (output / 'README.md').write_text('\n'.join(index), encoding='utf-8', newline='\n')
