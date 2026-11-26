"""Verify saved semantic-scoring artifacts and reconstruct problem scores."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

INPUT_FIELDS = {'item_id', 'problem_text', 'student_text', 'after_assistant'}
SCORE_FIELDS = {'item_id', 'learning_score', 'self_work_score', 'intent',
                'evidence_quote', 'reason', 'confidence', 'after_assistant'}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def checked_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Frozen artifact path escapes its scoring directory.')
    return path


def deduplicate(rows):
    grouped = defaultdict(list)
    for row in rows:
        assert set(row) == INPUT_FIELDS
        assert re.fullmatch('[0-9a-f]{20}', row['item_id'])
        assert type(row['after_assistant']) is bool
        grouped[(row['problem_text'], row['student_text'], row['after_assistant'])].append(row['item_id'])
    assert len(rows) == len({row['item_id'] for row in rows})
    unique, linkage = [], []
    for key, ids in grouped.items():
        uid = hashlib.sha256(json.dumps(key, ensure_ascii=False).encode('utf-8')).hexdigest()[:24]
        unique.append({'uid':uid, 'item_id':min(ids), 'problem_text':key[0],
                       'student_text':key[1], 'after_assistant':key[2]})
        linkage.append({'uid':uid, 'item_ids':sorted(ids)})
    return sorted(unique, key=lambda x:x['uid']), sorted(linkage, key=lambda x:x['uid'])


def validate_score(score, item):
    assert isinstance(score, dict) and set(score) == SCORE_FIELDS
    assert score['item_id'] == item['item_id']
    assert type(score['after_assistant']) is bool and score['after_assistant'] == item['after_assistant']
    for field in ['learning_score', 'self_work_score']:
        value = score[field]
        assert (field == 'learning_score' and value is None) or (type(value) is int and 0 <= value <= 3)
    assert score['intent'] in {'answer_delegation', 'concept_or_step_help', 'own_attempt', 'ambiguous', 'off_topic'}
    assert score['confidence'] in {'high', 'medium', 'low'}
    quote = score['evidence_quote']
    assert isinstance(quote, str) and len(quote) <= 160 and quote in item['student_text']
    assert quote or not item['student_text']
    assert isinstance(score['reason'], str) and score['reason'].strip()


def parse_output(stdout, final, expected):
    events = [json.loads(line) for line in stdout.decode('utf-8').splitlines() if line.strip()]
    assert events
    messages, threads, turns, tools = [], [], [], []
    for event in events:
        kind = event.get('type', '')
        assert kind not in {'error', 'turn.failed'}
        if kind == 'thread.started':
            threads.append(event.get('thread_id'))
        if kind == 'turn.completed':
            turns.append(event.get('usage'))
        if kind.startswith('item.'):
            item = event.get('item', {})
            if item.get('type') not in {'reasoning', 'agent_message'}:
                tools.append(event)
            if kind == 'item.completed' and item.get('type') == 'agent_message':
                messages.append(item.get('text', ''))
        elif any(token in kind.lower() for token in ['tool', 'command', 'mcp', 'web_search']):
            tools.append(event)
    assert not tools and len(turns) == 1 and messages
    output = json.loads(final.decode('utf-8'))
    assert json.loads(messages[-1]) == output
    assert isinstance(output, dict) and set(output) == {'scores'}
    assert isinstance(output['scores'], list) and len(output['scores']) == len(expected)
    scores = {}
    for score, item in zip(output['scores'], expected):
        validate_score(score, item)
        scores[item['uid']] = score
    assert len(scores) == len(expected)
    metadata = {'thread_ids':threads, 'turn_usage':turns, 'tool_events':tools,
                'event_count':len(events), 'returned_model':None,
                'backend_snapshot':None, 'system_fingerprint':None}
    for event in events:
        for field in ['model', 'model_name']:
            if isinstance(event.get(field), str):
                metadata['returned_model'] = event[field]
        for field in ['backend_snapshot', 'system_fingerprint']:
            if field in event:
                metadata[field] = event[field]
    return scores, metadata


def replay_scores(run_dir: Path):
    """Return primary UID+eight-field judgments and verified experimental metadata."""
    run_dir = Path(run_dir)
    manifest, frozen = read_json(run_dir / 'manifest.json'), read_json(run_dir / 'freeze.json')
    assert manifest['transport'] == 'codex_cli' and manifest['format_version'] == 2
    assert sha(run_dir / 'manifest.json') == frozen['manifest_sha256']
    for name, digest in frozen['files'].items():
        assert sha(checked_path(run_dir, name)) == digest, 'Frozen LLM artifact changed: ' + name
    for name, digest in manifest['files'].items():
        assert frozen['files'].get(name) == digest
    rows = read_lines(run_dir / 'full_inputs.jsonl')
    unique, linkage = deduplicate(rows)
    assert unique == read_lines(run_dir / 'unique_inputs.jsonl')
    assert linkage == read_json(run_dir / 'duplicate_linkage.json')
    size = manifest['batch_size']
    batches = [{'batch_id':f'batch_{i//size+1:04d}', 'uids':[r['uid'] for r in unique[i:i+size]]}
               for i in range(0, len(unique), size)]
    assert batches == read_json(run_dir / 'batches.json')
    eligible = [b['batch_id'] for b in batches if len(b['uids']) == size]
    repeat_ids = sorted(eligible, key=lambda x:hashlib.sha256(('repeat-batch:v1:'+x).encode()).hexdigest())[:manifest['repeat_batches_count']]
    assert repeat_ids == manifest['repeat_batch_ids']
    assert manifest['repeat_count'] == sum(len(b['uids']) for b in batches if b['batch_id'] in repeat_ids)
    items, by_batch = {r['uid']:r for r in unique}, {b['batch_id']:b for b in batches}
    prompt = (run_dir / 'scoring_prompt.txt').read_bytes().decode('utf-8')
    wrapper = (run_dir / 'batch_wrapper.txt').read_bytes().decode('utf-8')
    for batch in batches:
        inputs = [{k:items[uid][k] for k in sorted(INPUT_FIELDS)} for uid in batch['uids']]
        expected = (prompt+'\n\n'+wrapper+'\nINPUT CASES (JSON data):\n'+
                    json.dumps(inputs, ensure_ascii=False, sort_keys=True, indent=2)+'\n').encode('utf-8')
        assert (run_dir / f"requests/{batch['batch_id']}.txt").read_bytes() == expected
    attempts = read_lines(run_dir / 'attempts.jsonl')
    retained, seen = {}, set()
    for attempt in attempts:
        assert attempt['attempt_id'] not in seen
        seen.add(attempt['attempt_id'])
        batch_id, phase = attempt['batch_id'], attempt['phase']
        assert batch_id in by_batch and phase in {'primary', 'repeat'}
        assert phase != 'repeat' or batch_id in repeat_ids
        request = f'requests/{batch_id}.txt'
        assert attempt['request_file'] == request
        assert attempt['manifest_sha256'] == frozen['manifest_sha256']
        assert attempt['request_sha256'] == manifest['files'][request]
        for name, digest in attempt['artifacts'].items():
            assert frozen['files'].get(name) == digest
        if attempt.get('auth_check_file'):
            auth = read_json(checked_path(run_dir, attempt['auth_check_file']))
            assert auth['verified_chatgpt_subscription']
            assert auth['cli_executable_sha256'] == manifest['cli_executable_sha256']
        valid = False
        try:
            scores, output_metadata = parse_output(
                checked_path(run_dir, attempt['stdout_file']).read_bytes(),
                checked_path(run_dir, attempt['final_file']).read_bytes(),
                [items[uid] for uid in by_batch[batch_id]['uids']])
            valid = attempt['return_code'] == 0 and not attempt.get('timed_out', False)
        except (AssertionError, ValueError, KeyError, TypeError, OSError):
            pass
        assert bool(attempt['valid']) == valid
        if valid:
            assert output_metadata == attempt['output_metadata']
            retained.setdefault((phase, batch_id), scores)
    primary, repeats = {}, {}
    for (phase, _), scores in retained.items():
        (primary if phase == 'primary' else repeats).update(scores)
    assert len(primary) == len(unique) == manifest['unique_inputs'] == 3825
    assert len(rows) == manifest['input_records'] == 4756
    assert len(repeats) == manifest['repeat_count'] == 100
    agreement = {}
    fields = ['learning_score', 'self_work_score', 'intent', 'confidence']
    for field in fields:
        equal = sum(primary[uid][field] == score[field] for uid, score in repeats.items())
        agreement[field] = {'n':len(repeats), 'equal':equal, 'fraction':equal/len(repeats)}
    equal = sum(all(primary[uid][field] == score[field] for field in fields) for uid, score in repeats.items())
    agreement['all_rating_fields'] = {'n':len(repeats), 'equal':equal, 'fraction':equal/len(repeats)}
    verification = {'complete':True, 'primary_expected':len(unique), 'primary_valid':len(primary),
        'repeat_expected':manifest['repeat_count'], 'repeat_valid':len(repeats),
        'primary_batches_expected':len(batches), 'primary_batches_valid':sum(p == 'primary' for p,_ in retained),
        'repeat_batches_expected':len(repeat_ids), 'repeat_batches_valid':sum(p == 'repeat' for p,_ in retained),
        'attempt_count':len(attempts), 'valid_attempts':sum(bool(r['valid']) for r in attempts),
        'returned_models':sorted({r['output_metadata']['returned_model'] for r in attempts if r['valid'] and r['output_metadata']['returned_model']}),
        'returned_model_observed':any(r['valid'] and r['output_metadata']['returned_model'] for r in attempts),
        'manifest_sha256':frozen['manifest_sha256'], 'repeat_agreement':agreement}
    assert verification == frozen['verification']
    return [{'uid':uid, **primary[uid]} for uid in sorted(primary)], {
        'verification':verification, 'manifest':manifest, 'freeze_sha256':sha(run_dir / 'freeze.json'),
        'frozen_files_verified':len(frozen['files']), 'full_input_sha256':sha(run_dir / 'full_inputs.jsonl')}
