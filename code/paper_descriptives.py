"""Generate the manuscript's sample, summary and semantic-scoring tables."""

from paper_table_io import render_table


SAMPLE_STAGES = {
    'released_student_sessions': 'Released records',
    'nonhonors': 'Non-honors records',
    'nonhonors_AI': 'GPT Base or GPT Tutor',
    'observed_user_chat': 'Recorded student dialogue',
    'common_complete_sessions': 'Complete model variables',
    'baseline_classifiable_sample': 'Classifiable dialogue (estimation sample)',
}

# Each row specifies the CSV label, the manuscript notation and the displayed precision.
SUMMARY_MEASURES = [
    ('Y', 'Unassisted exam score', 'Y', 3),
    ('G', 'Prior GPA', 'G', 3),
    ('D', 'Task difficulty', 'D', 3),
    ('E_baseline', 'Understanding share', 'E^{U}', 3),
    ('E_net_observed', 'Net understanding', 'E^{N}', 3),
    ('E_reasoning_observed', 'Reasoning coverage', 'E^{R}', 3),
    ('E_learning_observed', 'Semantic score', 'E^{L}', 3),
    ('O_count', 'Problems with a student message', 'O', 2),
]

# Manuscript labels for the frozen scoring scheme, written once for both appendix tables.
SCORE_CATEGORIES = [
    ('null', 'Null', 'Insufficient information',
     'Uninformative or off-topic text, the copied problem only, a generic help request, or an unsupported final guess without an identifiable operation.'),
    ('0', '0', 'Delegation of the answer or full solution',
     "A clear request to supply the answer or outsource the complete solution, without a specific learning request or a mathematical step of the student's own."),
    ('1', '1', 'Specific concept, step, or hint request',
     'A request about a specific mathematical concept, step, or hint, including an explicit request to withhold the answer.'),
    ('2', '2', 'Own operation or identifiable attempt',
     "An identifiable operation, relationship, strategy, or intermediate step of the student's own, or a request to check a specific step or misconception."),
    ('3', '3', 'Connected reasoning or error diagnosis',
     'Connected reasoning with linked steps, a reasoned justification, or an error diagnosis that goes beyond copying the question.'),
]


def tex_row(cells):
    return ' & '.join(cells) + r' \\'


def write_sample_table(metadata, tables_dir):
    csv_rows = [('Stage', 'All arms', 'Base', 'Tutor', 'Students', 'Classrooms')]
    tex_rows = []
    for stage in metadata['sample_flow']:
        label = SAMPLE_STAGES[stage['stage']]
        counts = [stage[k] for k in ('sessions', 'Base_sessions', 'Tutor_sessions', 'students', 'classrooms')]
        csv_rows.append((label, *counts))
        tex_rows.append(tex_row([label, *(f'{n:,}' for n in counts)]))
    render_table(tables_dir, 'table_1_sample', tex_rows, csv_rows)


def write_summary_table(panel, tables_dir):
    base = panel.loc[panel['T'].eq(0)]
    tutor = panel.loc[panel['T'].eq(1)]
    csv_rows = [('Variable', 'All mean', 'All SD', 'Base mean', 'Tutor mean')]
    tex_rows = []
    for column, label, symbol, digits in SUMMARY_MEASURES:
        if column == 'E_baseline':
            tex_rows.extend([r'\addlinespace', r'\multicolumn{5}{l}{\textit{Effort measures}} \\'])
        elif column == 'O_count':
            tex_rows.append(r'\addlinespace')
        values = [panel[column].mean(), panel[column].std(ddof=1), base[column].mean(), tutor[column].mean()]
        formatted = [f'{value:.{digits}f}' for value in values]
        csv_rows.append((label, *formatted))
        tex_label = (r'\quad ' if column.startswith('E_') else '') + f'{label}, ${symbol}$'
        tex_values = [f'${value}$' if value.startswith('-') else value for value in formatted]
        tex_rows.append(tex_row([tex_label, *tex_values]))

    tex_rows.append(r'\midrule')
    for label, key in [('Student-sessions', None), ('Students', 'student_id'), ('Classrooms', 'class_id')]:
        all_n, base_n, tutor_n = [len(frame) if key is None else frame[key].nunique() for frame in (panel, base, tutor)]
        csv_rows.append((label, all_n, '', base_n, tutor_n))
        tex_rows.append(tex_row([label, f'{all_n:,}', '', f'{base_n:,}', f'{tutor_n:,}']))
    render_table(tables_dir, 'table_2_summary', tex_rows, csv_rows)


def write_scoring_tables(metadata, tables_dir):
    rubric_csv = [('Score', "Evidence in the student's message")]
    rubric_tex = []
    distribution_csv = [('Score', 'Count', 'Percent', 'Recorded evidence')]
    distribution_tex = []
    counts = metadata['llm']['expanded_learning_score_counts']
    total = sum(counts.values())
    if set(counts).difference(key for key, *_ in SCORE_CATEGORIES) or total <= 0:
        raise ValueError('Unexpected score categories or empty first-message sample.')

    for key, label, short_text, rubric_text in SCORE_CATEGORIES:
        rubric_csv.append((label, rubric_text))
        if rubric_tex:
            rubric_tex.append(r'\addlinespace')
        rubric_tex.append(tex_row([label, rubric_text]))

        count = counts.get(key, 0)
        percent = f'{100 * count / total:.1f}'
        distribution_csv.append((label, count, percent, short_text))
        distribution_tex.append(tex_row([label, f'{count:,}', percent, short_text]))
    distribution_csv.append(('Total', total, '100.0', ''))
    distribution_tex.extend([r'\midrule', tex_row(['Total', f'{total:,}', '100.0', ''])])
    render_table(tables_dir, 'table_a1_rubric', rubric_tex, rubric_csv)
    render_table(tables_dir, 'table_a2_scores', distribution_tex, distribution_csv,
                 fields={'N_RECORDS': f'{total:,}'})


def write_descriptive_tables(panel, metadata, tables_dir):
    """Write the four descriptive manuscript tables as LaTeX and CSV."""
    write_sample_table(metadata, tables_dir)
    write_summary_table(panel, tables_dir)
    write_scoring_tables(metadata, tables_dir)
