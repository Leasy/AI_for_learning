"""Prepare released data and saved semantic scores for regression analysis."""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import csv
import io
import platform
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

from dialogue_rules import classify, norm
from offline_llm import deduplicate, read_json, read_lines, replay_scores, sha

METHODS = ['E_baseline', 'E_learning_observed', 'E_net_observed', 'E_reasoning_observed']
REASONING_RULES = frozenset({'specific_why_question', 'specific_how_step',
    'specific_step_explanation', 'articulated_reasoning', 'described_attempt',
    'error_diagnosis', 'reasoning_check', 'requests_help_without_answer'})
SALT = 'semantic-effort-v1-20261022'


def _semantic_text(value):
    value = re.sub(r'\\?u([0-9a-fA-F]{4})', lambda match:chr(int(match[1],16)), value)
    value = value.encode('utf-16','surrogatepass').decode('utf-16','replace')
    value = unicodedata.normalize('NFC', value).strip(' "')
    value = re.sub(r'\b(?:GPT\s*Base|GPT\s*Tutor|vanilla|augmented|ChatGPT|OpenAI)\b', '[AI tool]', value, flags=re.I)
    value = re.sub(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', '[email]', value)
    return re.sub(r'\b[0-9a-f]{8}-[0-9a-f-]{27,}\b', '[id]', value, flags=re.I)


def _extract_first_turns(raw_rows, questions, sample):
    groups, assistants_seen = {}, set()
    for row in raw_rows:
        if (int(row['username']),int(row['session_id'])) not in sample:
            continue
        conversation = row['conversation_id']
        if row['role'] == 'assistant':
            assistants_seen.add(conversation)
        if row['role'] != 'user':
            continue
        key = (row['username'],row['session_id'],row['grade'],row['problem_id'])
        groups.setdefault(key, (_semantic_text(row['message']),conversation in assistants_seen))
    inputs, linkage = [], {}
    for key, (text, after) in groups.items():
        student,session,grade,problem = key
        item_id = hashlib.sha256((SALT+'|'+'|'.join(key)).encode()).hexdigest()[:20]
        inputs.append({'item_id':item_id,
            'problem_text':_semantic_text(questions.get((grade,session,problem),'')),
            'student_text':text, 'after_assistant':after})
        linkage[item_id] = {'student_id':int(student),'session_id':int(session),
                            'grade':int(grade),'problem_id':int(problem)}
    return sorted(inputs,key=lambda x:x['item_id']), linkage


def first_turn_inputs(root: Path, analysis: pd.DataFrame):
    """Read-only first-message reconstruction for independent aggregation checks."""
    root = Path(root)
    questions_path = root/'data/raw/text_analysis/data/raw/question_list.csv'
    with questions_path.open(encoding='cp1252',newline='') as handle:
        questions = {(r['grade'],r['session'].lstrip('s'),r['problem_id']):r['question']
                     for r in csv.DictReader(handle)}
    sample = set(analysis[['student_id','session_id']].itertuples(index=False,name=None))
    with (root/'data/raw/text_analysis/data/raw/valid_student_data_w_time_stamp.csv').open(encoding='utf-8-sig',newline='') as handle:
        return _extract_first_turns(csv.DictReader(handle),questions,sample)


def _problem_evidence(raw):
    """Classify released messages in original conversation/source-row order."""
    grouped = defaultdict(lambda:{'u':False,'a':False,'reasoning':False,'messages':0})
    labels = Counter()
    for _, conversation in raw.groupby('conversation_id',sort=False):
        system = ' '.join(norm(text) for text in conversation.loc[conversation.role.eq('system'),'message'])
        previous = ''
        for row in conversation.itertuples(index=False):
            text = norm(row.message)
            if row.role == 'assistant':
                previous = text
            if row.role != 'user':
                continue
            label,rule = classify(text,system,previous)
            labels[label] += 1
            key = (int(row.username),int(row.session_id),int(row.grade),int(row.problem_id))
            entry = grouped[key]
            entry['u'] |= label == 'U'
            entry['a'] |= label == 'A'
            entry['reasoning'] |= label == 'U' and rule in REASONING_RULES
            entry['messages'] += 1
    problems = []
    for key,value in grouped.items():
        problems.append(dict(zip(['student_id','session_id','grade','problem_id'],key),**value))
    return pd.DataFrame(problems), dict(labels)


def build_data(root: Path):
    """Return analysis DataFrame and JSON-compatible metadata; write nothing."""
    root = Path(root).resolve()
    manifest = read_json(root/'data/input_manifest.json')
    for relative,digest in manifest['raw_files'].items():
        assert sha(root/relative) == digest, 'Released raw input changed: '+relative
    assert sha(root/'code/dialogue_rules.py') == manifest['dialogue_rules_sha256']
    run_dir = root/'data/llm/scoring_run'
    assert sha(run_dir/'manifest.json') == manifest['llm_manifest_sha256']
    assert sha(run_dir/'freeze.json') == manifest['llm_freeze_sha256']
    raw_root = root/'data/raw'
    def released(relative):
        return pd.read_csv(raw_root/relative,dtype='string',encoding='utf-8-sig')
    final = released('main_regressions/final_data.csv')
    practice = released('main_regressions/problem_part2.csv')
    test = released('main_regressions/problem_part3.csv')
    mapping = released('main_regressions/problem_mapping.csv')
    questions = pd.read_csv(raw_root/'text_analysis/data/raw/question_list.csv',dtype='string',encoding='latin-1')
    raw = pd.read_csv(raw_root/'text_analysis/data/raw/valid_student_data_w_time_stamp.csv',dtype=str,keep_default_na=False)
    assert not final.duplicated(['Student ID','Session']).any()
    assert not practice.duplicated(['Student ID','Problem']).any()
    assert not test.duplicated(['Student ID','Problem']).any()
    questions['practice_problem_id'] = questions['session']+'_'+questions.grade+'_'+questions.problem_id
    questions['gs'] = questions['session']+'_'+questions.grade
    assert len(questions) == questions.practice_problem_id.nunique() == 57
    assert set(questions.practice_problem_id) == set(practice.Problem) == set(mapping.part2)
    assignments = questions.groupby('gs').size().rename('P_assigned')
    panel = final.rename(columns={'Student ID':'student_id','Class':'class_id','Year':'grade',
        'Session':'session_id','Part3Tot':'Y','Part2Tot':'practice_session_score','gpa_prev':'G',
        'GPTBase':'B','GPTTutor':'T','Treatment arm':'arm','Honors':'honors',
        'teacher':'teacher_id','Grader':'grader_id'}).copy()
    panel['gs'] = 's'+panel.session_id+'_'+panel.grade
    panel = panel.merge(assignments,left_on='gs',right_index=True,how='left',validate='many_to_one')
    assert panel.P_assigned.notna().all()
    for name in ['Y','practice_session_score','G','B','T','honors']:
        panel[name] = pd.to_numeric(panel[name],errors='raise')
    assert panel.B.eq(panel.arm.eq('vanilla').astype(int)).all()
    assert panel['T'].eq(panel.arm.eq('augmented').astype(int)).all()
    reconciliation = {}
    for label,source,outcome in [('practice',practice,'practice_session_score'),('test',test,'Y')]:
        source['Score'] = pd.to_numeric(source.Score,errors='raise')
        means = source.groupby(['Student ID','Session']).Score.agg(['mean','count'])
        means.index.names = ['student_id','session_id']
        checked = panel.merge(means,left_on=['student_id','session_id'],right_index=True,how='left',validate='one_to_one')
        difference = (checked[outcome]-checked['mean']).abs()
        assert checked['count'].notna().all() and difference.dropna().lt(1e-12).all()
        assert checked[outcome].isna().eq(checked['mean'].isna()).all()
        assert (checked['count'].eq(checked.P_assigned) if label == 'practice' else checked['count'].eq(4)).all()
        reconciliation[label] = {'sessions_compared':len(checked),'max_absolute_difference':float(difference.max())}
    # Difficulty uses no-AI, nonhonors practice scores, equally weighting every assigned question.
    control = practice[practice['Treatment arm'].eq('control') & practice.Honors.eq('0')]
    difficulty = (1-control.groupby('Problem').Score.mean()).rename('difficulty')
    roster = questions[['gs','practice_problem_id']].merge(difficulty,left_on='practice_problem_id',right_index=True,how='left',validate='one_to_one')
    d = roster.groupby('gs').difficulty.agg(lambda x:float(x.mean()) if x.notna().all() else np.nan).rename('D')
    panel = panel.merge(d,left_on='gs',right_index=True,how='left',validate='many_to_one')
    users = raw[raw.role.eq('user')].copy()
    users['practice_problem_id'] = 's'+users.session_id+'_'+users.grade+'_'+users.problem_id
    assert users.practice_problem_id.isin(questions.practice_problem_id).all()
    logs = users.groupby(['username','session_id']).agg(total_user_messages=('role','size'),
        n_problems_with_chat=('practice_problem_id','nunique'),log_grade=('grade','first'),
        log_arm=('treatment','first'),n_log_grades=('grade','nunique'),n_log_arms=('treatment','nunique'))
    assert logs.n_log_grades.eq(1).all() and logs.n_log_arms.eq(1).all()
    logs.log_arm = logs.log_arm.map({'aug':'augmented','vanilla':'vanilla'})
    assert logs.log_arm.notna().all()
    panel = panel.merge(logs,left_on=['student_id','session_id'],right_index=True,how='left',validate='one_to_one')
    panel['has_chat'] = panel.total_user_messages.notna()
    assert panel.loc[panel.has_chat,'grade'].eq(panel.loc[panel.has_chat,'log_grade']).all()
    assert panel.loc[panel.has_chat,'arm'].eq(panel.loc[panel.has_chat,'log_arm']).all()
    panel['E_raw'] = (panel.total_user_messages/panel.P_assigned).where(panel.has_chat)
    flow = []
    def record(frame,stage):
        flow.append({'stage':stage,'sessions':len(frame),'students':int(frame.student_id.nunique()),
            'classrooms':int(frame.class_id.nunique()),'Base_sessions':int(frame.B.eq(1).sum()),
            'Tutor_sessions':int(frame['T'].eq(1).sum())})
    record(panel,'released_student_sessions')
    nonhonors = panel[panel.honors.eq(0)].copy(); record(nonhonors,'nonhonors')
    ai = nonhonors[(nonhonors.B+nonhonors['T']).eq(1)].copy(); record(ai,'nonhonors_AI')
    record(ai[ai.has_chat],'observed_user_chat')
    required = ['student_id','class_id','Y','G','E_raw','D','T','grade','session_id','teacher_id','grader_id']
    common = ai.dropna(subset=required).copy()
    for column in ['student_id','session_id','grade']:
        common[column] = common[column].astype(int)
    common = common.sort_values(['student_id','session_id'])
    record(common,'common_complete_sessions')
    assert len(common) == 1701 and common.student_id.nunique() == 508
    controls,category_coding = ['T','D','G'], {}
    for field in ['grade','session_id','teacher_id','grader_id']:
        levels = sorted(common[field].unique())
        category_coding[field] = {'reference':str(levels[0]),'levels':[str(x) for x in levels]}
        for value in levels[1:]:
            name = field+str(value); common[name] = common[field].eq(value).astype(int); controls.append(name)
    problems,message_counts = _problem_evidence(raw)
    assert sum(message_counts.values()) == 28666
    evidence = problems.merge(common[['student_id','session_id','grade']],on=['student_id','session_id','grade'],validate='many_to_one')
    assert len(evidence) == 6635 and int(evidence.messages.sum()) == 27560
    evidence['classified'] = evidence.u | evidence.a
    counts = evidence.groupby(['student_id','session_id','grade']).agg(
        U_count=('u','sum'),A_any_count=('a','sum'),classified_count=('classified','sum'),
        reasoning_count=('reasoning','sum'),observed_problem_count=('problem_id','size'))
    common = common.merge(counts,left_on=['student_id','session_id','grade'],right_index=True,validate='one_to_one')
    common['E_baseline'] = common.U_count/common.classified_count.replace(0,np.nan)
    common['E_net_observed'] = (common.U_count-common.A_any_count)/common.observed_problem_count
    common['E_reasoning_observed'] = common.reasoning_count/common.observed_problem_count
    analysis = common[common.classified_count.ge(1)].copy().sort_values(['student_id','session_id'])
    record(analysis,'baseline_classifiable_sample')
    assert len(analysis) == 1163 and analysis.student_id.nunique() == 468 and analysis.class_id.nunique() == 27
    assert not analysis.duplicated(['student_id','session_id']).any()
    assert analysis.groupby('student_id')[['class_id','grade','teacher_id','T','G']].nunique().le(1).all().all()
    with (raw_root/'text_analysis/data/raw/question_list.csv').open(encoding='cp1252',newline='') as handle:
        question_text = {(r['grade'],r['session'].lstrip('s'),r['problem_id']):r['question'] for r in csv.DictReader(handle)}
    sample = set(analysis[['student_id','session_id']].itertuples(index=False,name=None))
    inputs,linkage = _extract_first_turns(raw.to_dict('records'),question_text,sample)
    frozen_inputs = read_lines(run_dir/'full_inputs.jsonl')
    assert inputs == frozen_inputs and len(inputs) == 4756
    # Use CRLF for byte-exact JSONL reconstruction.
    input_bytes = ''.join(json.dumps(r,ensure_ascii=False)+'\r\n' for r in inputs).encode('utf-8')
    assert hashlib.sha256(input_bytes).hexdigest() == sha(run_dir/'full_inputs.jsonl')
    scores,llm_metadata = replay_scores(run_dir)
    _,duplicate_map = deduplicate(inputs)
    assert duplicate_map == read_json(run_dir/'duplicate_linkage.json')
    by_uid = {r['uid']:r for r in scores}
    session_scores,learning_distribution = defaultdict(list),Counter()
    for group in duplicate_map:
        score = by_uid[group['uid']]
        for item_id in group['item_ids']:
            info = linkage[item_id]
            session_scores[(info['student_id'],info['session_id'])].append(score['learning_score'])
            learning_distribution['null' if score['learning_score'] is None else str(score['learning_score'])] += 1
    llm_values = {key:sum(v for v in values if v is not None)/(3*len(values)) for key,values in session_scores.items()}
    analysis['E_learning_observed'] = [llm_values[key] for key in analysis[['student_id','session_id']].itertuples(index=False,name=None)]
    analysis['O_count'] = analysis.observed_problem_count.astype(int)
    analysis['R_count'] = analysis.reasoning_count.astype(int)
    assert len(llm_values) == len(analysis)
    assert np.isfinite(analysis[METHODS].to_numpy(float)).all()
    assert analysis[['E_baseline','E_learning_observed','E_reasoning_observed']].ge(0).all().all()
    assert analysis[['E_baseline','E_learning_observed','E_reasoning_observed']].le(1).all().all()
    assert analysis.E_net_observed.between(-1,1).all()
    columns = list(dict.fromkeys(['student_id','class_id','session_id','grade','teacher_id','grader_id',
        'Y','T','D','G',*controls,*METHODS,
        'O_count','U_count','A_any_count','R_count','classified_count']))
    output = analysis[columns].reset_index(drop=True)
    for column in ['Y','D','G',*METHODS]:
        output[column] = output[column].astype(float)
    output['T'] = output['T'].astype(int)
    metadata = {'methods':METHODS,'controls':controls,'sample_flow':flow,'category_coding':category_coding,
        'sessions':len(output),'students':int(output.student_id.nunique()),'classrooms':int(output.class_id.nunique()),
        'raw_input_sha256':manifest['raw_files'],'dialogue_rules_sha256':manifest['dialogue_rules_sha256'],
        'session_score_reconciliation':reconciliation,'raw_log_records':len(raw),'user_messages':len(users),
        'all_message_labels':message_counts,'common_observed_problems':len(evidence),
        'common_zero_classified_sessions':int(common.classified_count.eq(0).sum()),
        'llm':{'run_id':llm_metadata['manifest']['run_id'],'model_requested':llm_metadata['manifest']['model_requested'],
            'cli_version':llm_metadata['manifest']['cli_version'],'reasoning_effort':llm_metadata['manifest']['reasoning_effort'],
            'temperature':None,'seed':None,'backend_snapshot':None,
            'frozen_files_verified':llm_metadata['frozen_files_verified'],
            'manifest_sha256':manifest['llm_manifest_sha256'],'freeze_sha256':llm_metadata['freeze_sha256'],
            'full_input_sha256':llm_metadata['full_input_sha256'],
            'verification':llm_metadata['verification'],'expanded_learning_score_counts':dict(learning_distribution),
            'aggregation':'sum(non-null learning_score)/(3*all observed problems); null is no recorded positive evidence, not zero latent effort',
            'scoring_calls_during_reproduction':0},
        'measure_definitions':{'E_baseline':'U_any / count(problems with any U or A); U has priority',
            'E_learning_observed':'sum(non-null learning_score)/(3*observed problems)',
            'E_net_observed':'(count(any U)-count(any A))/observed problems; mixed and unclassified contribute zero',
            'E_reasoning_observed':'count(problems with any U matching frozen specific reasoning rules)/observed problems'},
        'reasoning_rule_ids':sorted(REASONING_RULES),
        'data_definitions':{'Y':'Released current-session normalized unassisted post-test mean (Part3Tot)',
            'D':'Mean across all assigned questions of 1 minus the nonhonors no-AI control practice-score mean',
            'G':'Released prior GPA fraction','T':'Tutor=1, Base=0',
            'unit':'One observed student-session; all four methods use the identical baseline sample'}}
    return output, metadata


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def near(actual, expected, label, *, atol=2e-12, rtol=1e-12):
    require(np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True),
            'Independent data check differs: ' + label)


def check_saved_llm(analysis, root: Path):
    """Re-aggregate actual saved model outputs, independently of the data builder."""
    root = Path(root).resolve()
    run = root / 'data/llm/scoring_run'
    frozen = json.loads((run / 'freeze.json').read_text(encoding='utf-8'))
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    for relative, digest in frozen['files'].items():
        path = (run / relative).resolve()
        require(path.is_relative_to(run.resolve()), 'Frozen LLM path escapes run directory')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                f'Frozen LLM artifact changed: {relative}')
    require(frozen['verification']['complete'], 'Incomplete frozen LLM run')
    require(manifest['transport'] == 'codex_cli', 'Unexpected LLM scoring transport')
    source_records, linkage = first_turn_inputs(root, analysis)
    frozen_records = [json.loads(line) for line in
                      (run / 'full_inputs.jsonl').read_text(encoding='utf-8').splitlines() if line]
    require(source_records == frozen_records, 'Raw first-turn input reconstruction differs')
    unique = {row['uid']: row for row in [json.loads(line) for line in
              (run / 'unique_inputs.jsonl').read_text(encoding='utf-8').splitlines() if line]}
    grouped = defaultdict(list)
    for item in source_records:
        require(set(item) == {'item_id', 'problem_text', 'student_text', 'after_assistant'},
                'Unexpected field in blinded model input')
        key = (item['problem_text'], item['student_text'], item['after_assistant'])
        grouped[key].append(item['item_id'])
    for key, ids in grouped.items():
        uid = hashlib.sha256(json.dumps(key, ensure_ascii=False).encode('utf-8')).hexdigest()[:24]
        require(unique[uid] == {'uid': uid, 'item_id': min(ids), 'problem_text': key[0],
                                'student_text': key[1], 'after_assistant': key[2]},
                'Exact-input deduplication/representative differs')
    require(len(unique) == len(grouped), 'Unique scoring input count differs')
    batches = {row['batch_id']: row['uids'] for row in
               json.loads((run / 'batches.json').read_text(encoding='utf-8'))}
    schema_fields = set(json.loads((run / 'scoring_schema.json').read_text(encoding='utf-8'))['required'])
    retained = {}
    for line in (run / 'attempts.jsonl').read_text(encoding='utf-8').splitlines():
        if not line:
            continue
        attempt = json.loads(line)
        if not attempt['valid'] or attempt['phase'] != 'primary':
            continue
        events = [json.loads(line) for line in
                  (run / attempt['stdout_file']).read_text(encoding='utf-8').splitlines() if line]
        require(sum(event.get('type') == 'turn.completed' for event in events) == 1,
                'Retained model attempt does not complete exactly one turn')
        require(not any(event.get('type') in {'error', 'turn.failed'} for event in events),
                'Retained model attempt contains a failure event')
        require(all(event.get('item', {}).get('type') in {'agent_message', 'reasoning'}
                    for event in events if event.get('type', '').startswith('item.')),
                'Retained blinded score used a tool')
        messages = [event['item']['text'] for event in events
                    if event.get('type') == 'item.completed' and
                    event.get('item', {}).get('type') == 'agent_message']
        require(messages, 'Saved valid attempt has no model message')
        final = json.loads((run / attempt['final_file']).read_text(encoding='utf-8'))
        require(json.loads(messages[-1]) == final, 'Raw model message differs from final JSON')
        require(attempt['return_code'] == 0 and not attempt['timed_out'], 'Invalid retained attempt')
        scores = final['scores']
        expected_uids = batches[attempt['batch_id']]
        require(len(scores) == len(expected_uids), 'Wrong batch score count')
        for uid, score in zip(expected_uids, scores):
            item = unique[uid]
            require(set(score) == schema_fields and score['item_id'] == item['item_id'],
                    'Wrong score fields/representative identity')
            require(score['after_assistant'] is item['after_assistant'], 'Wrong input flag')
            learning = score['learning_score']
            require(learning is None or (type(learning) is int and 0 <= learning <= 3),
                    'Invalid saved learning score')
            require(type(score['self_work_score']) is int and 0 <= score['self_work_score'] <= 3,
                    'Invalid saved self-work score')
            require(score['intent'] in {'answer_delegation', 'concept_or_step_help', 'own_attempt',
                                       'ambiguous', 'off_topic'} and
                    score['confidence'] in {'high', 'medium', 'low'}, 'Invalid semantic category')
            quote = score['evidence_quote']
            require(isinstance(quote, str) and len(quote) <= 160 and quote in item['student_text'],
                    'Evidence is not a literal source quotation')
            require(bool(quote) or not item['student_text'], 'Missing evidence quote')
            require(isinstance(score['reason'], str) and bool(score['reason'].strip()),
                    'Missing saved semantic reason')
        retained.setdefault(attempt['batch_id'], dict(zip(expected_uids, scores)))
    primary = {uid: score for scores in retained.values() for uid, score in scores.items()}
    require(set(primary) == set(unique), 'Incomplete primary score coverage')
    expanded = {}
    for row in json.loads((run / 'duplicate_linkage.json').read_text(encoding='utf-8')):
        for item_id in row['item_ids']:
            require(item_id not in expanded, 'Duplicate original scoring item')
            expanded[item_id] = primary[row['uid']]['learning_score']
    require(set(expanded) == {row['item_id'] for row in frozen_records}, 'LLM duplicate expansion differs')
    totals = defaultdict(lambda: [0, 0])
    for item_id, score in expanded.items():
        row = linkage[item_id]
        key = (str(row['student_id']), int(row['session_id']), int(row['grade']))
        totals[key][0] += 0 if score is None else score
        totals[key][1] += 1
    for row in analysis.itertuples(index=False):
        numerator, observed = totals[(str(row.student_id), int(row.session_id), int(row.grade))]
        require(observed > 0, 'Session has no observed semantic problem')
        near(row.E_learning_observed, numerator / (3 * observed), 'LLM session aggregation', atol=2e-12, rtol=0)
    return {'frozen_artifacts_verified': len(frozen['files']), 'original_items': len(expanded),
            'unique_primary_judgments': len(primary), 'primary_batches': len(retained),
            'all_session_aggregates_match': True}


def check_clean_data(analysis: pd.DataFrame, metadata: dict, root: Path):
    """Validate measurement definitions, units and design columns."""
    controls = list(metadata['controls'])
    require(len(controls) == len(set(controls)) == 33 and 'T' in controls,
            'Unexpected control-column definitions')
    require(len(analysis) == 1163 and analysis.student_id.nunique() == 468 and
            analysis.class_id.nunique() == 27, 'Unexpected fixed analysis sample')
    require(not analysis.duplicated(['student_id','session_id']).any(), 'Duplicate student-session row')
    require(set(analysis['T'].unique()) == {0,1}, 'Both treatment arms are required')
    require(analysis.groupby('class_id')['T'].nunique().eq(1).all(), 'Treatment changes within classroom')
    require(analysis.groupby('student_id')['class_id'].nunique().eq(1).all(), 'Students cross classroom clusters')
    numeric = ['Y',*controls,*METHODS,'O_count','U_count','A_any_count','R_count','classified_count']
    require(np.isfinite(analysis[numeric].to_numpy(dtype=float)).all(), 'Missing or nonfinite regression input')
    for measure in METHODS:
        require(analysis[measure].between(-1 if measure == 'E_net_observed' else 0,1).all(),
                'Invalid construction range: ' + measure)
    O = analysis.O_count.to_numpy(float)
    require((O > 0).all() and analysis.classified_count.gt(0).all(), 'Invalid measurement denominator')
    require((analysis[['O_count','U_count','A_any_count','R_count','classified_count']] % 1).eq(0).all().all(),
            'Problem counts are not integers')
    require(analysis.U_count.le(analysis.classified_count).all() and
            analysis.A_any_count.le(analysis.classified_count).all() and
            analysis.classified_count.le(analysis.O_count).all() and
            analysis.R_count.le(analysis.U_count).all(), 'Inconsistent problem counts')
    near(analysis.E_baseline, analysis.U_count/analysis.classified_count, 'Baseline U/C')
    near(analysis.E_net_observed, (analysis.U_count-analysis.A_any_count)/O, 'Net (U-A)/O')
    near(analysis.E_reasoning_observed, analysis.R_count/O, 'Reasoning R/O')
    matrix = np.column_stack([np.ones(len(analysis)), analysis[controls].to_numpy(float)])
    rank = int(np.linalg.matrix_rank(matrix))
    require(matrix.shape[1] == rank, 'Reduced-form design is not full rank')
    llm = check_saved_llm(analysis, root)
    return {'passed':True, 'sessions':len(analysis), 'students':int(analysis.student_id.nunique()),
        'classrooms':int(analysis.class_id.nunique()), 'regression_design_columns':matrix.shape[1],
        'regression_design_rank':rank, 'control_columns':controls,
        'checks':['released raw input hashes and unchanged bilingual rules',
                  'raw score reconciliation and common-sample reconstruction',
                  'units, classroom assignment, ranges and count-based definitions',
                  'complete standalone regression columns and full design rank',
                  'saved LLM artifact hashes, raw responses, evidence and independent aggregation'],
        'llm':llm, 'statistical_estimation_performed':False, 'model_calls':0}


def clean_data(root: Path, output: Path | None = None) -> dict:
    """Write the checked regression CSV and sibling cleaning_info.json; return metadata."""
    root = Path(root).resolve()
    output = (Path(output) if output is not None else root/'data/clean/analysis_sessions.csv').resolve()
    require(output.suffix.lower() == '.csv', 'The cleaning output must be a CSV file')
    require(not output.is_relative_to(root/'data/raw') and not output.is_relative_to(root/'data/llm'),
            'Cleaning output must not overwrite frozen raw inputs or LLM evidence')
    analysis, metadata = build_data(root)
    checks = check_clean_data(analysis, metadata, root)
    csv_text = analysis.to_csv(index=False, na_rep='', lineterminator='\n', float_format='%.17g')
    # Check standalone export before writing either artifact. Preserve joint grader labels.
    exported = pd.read_csv(io.StringIO(csv_text),
        dtype={'class_id':str,'teacher_id':str,'grader_id':str})
    require(list(exported.columns) == list(analysis.columns) and len(exported) == len(analysis),
            'Standalone CSV schema or row count changed during serialization')
    for column in analysis.columns:
        if column in {'class_id','teacher_id','grader_id'}:
            require(exported[column].astype(str).equals(analysis[column].astype(str)),
                    'Standalone CSV changed identifier: ' + column)
        else:
            near(exported[column], analysis[column], 'Standalone CSV round trip: ' + column,
                 atol=1e-14, rtol=1e-14)
    csv_bytes = csv_text.encode('utf-8')
    metadata.update({'clean_csv':str(output), 'clean_csv_sha256':hashlib.sha256(csv_bytes).hexdigest(),
        'clean_csv_rows':len(analysis), 'clean_csv_columns':list(analysis.columns),
        'data_checks':checks,
        'python_environment':{'python':sys.version, 'executable':sys.executable,
            'platform':platform.platform(), 'numpy':np.__version__, 'pandas':pd.__version__},
        'workflow':'Python data preparation -> analysis CSV -> R estimation'})
    info_bytes = (json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(csv_bytes)
    require(sha(output) == metadata['clean_csv_sha256'], 'Written regression CSV hash differs')
    (output.parent/'cleaning_info.json').write_bytes(info_bytes)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,
        help='Standalone regression CSV; default: data/clean/analysis_sessions.csv')
    args = parser.parse_args()
    metadata = clean_data(Path(__file__).resolve().parents[1],args.output)
    print(json.dumps({'passed':metadata['data_checks']['passed'],
        'clean_csv':metadata['clean_csv'], 'sha256':metadata['clean_csv_sha256'],
        'sessions':metadata['sessions'], 'students':metadata['students'],
        'classrooms':metadata['classrooms'], 'constructions':metadata['methods'],
        'model_calls':0},indent=2))


if __name__ == '__main__':
    main()


