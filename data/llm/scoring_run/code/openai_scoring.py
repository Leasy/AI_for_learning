"""Frozen, blinded OpenAI Responses scoring with auditable local replay.

No credential is written to an artifact. prepare/verify/freeze never use the network.
Only execute sends the four documented input fields to the configured HTTPS API.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import threading
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import uuid

PACKAGE = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PACKAGE / 'data/semantic/api_runs'
INPUT_FIELDS = {'item_id', 'problem_text', 'student_text', 'after_assistant'}
SCORE_FIELDS = {'item_id', 'learning_score', 'self_work_score', 'intent',
                'evidence_quote', 'reason', 'confidence', 'after_assistant'}
INTENTS = {'answer_delegation', 'concept_or_step_help', 'own_attempt', 'ambiguous', 'off_topic'}
LOCK = threading.Lock()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')


def json_lines(rows: list[dict]) -> bytes:
    return ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True,
                              allow_nan=False) + '\n' for r in rows).encode('utf-8')


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def read_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def safe_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Artifact path escapes run directory')
    return path


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Immutable artifact differs: {path.name}')
        return
    with path.open('xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def deduplicate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple, list[str]] = {}
    seen: set[str] = set()
    for row in rows:
        if set(row) != INPUT_FIELDS or not re.fullmatch('[0-9a-f]{20}', row['item_id']):
            raise ValueError('Input has extra/invalid fields or item ID')
        if row['item_id'] in seen or type(row['after_assistant']) is not bool:
            raise ValueError('Duplicate item ID or invalid after_assistant')
        if not isinstance(row['problem_text'], str) or not isinstance(row['student_text'], str):
            raise ValueError('Input text must be a string')
        seen.add(row['item_id'])
        key = (row['problem_text'], row['student_text'], row['after_assistant'])
        grouped.setdefault(key, []).append(row['item_id'])
    unique, linkage = [], []
    for key, ids in grouped.items():
        uid = sha(json.dumps(key, ensure_ascii=False).encode('utf-8'))[:24]
        unique.append({'uid': uid, 'item_id': min(ids), 'problem_text': key[0],
                       'student_text': key[1], 'after_assistant': key[2]})
        linkage.append({'uid': uid, 'item_ids': sorted(ids)})
    unique.sort(key=lambda row: row['uid'])
    linkage.sort(key=lambda row: row['uid'])
    if len({row['uid'] for row in unique}) != len(unique):
        raise ValueError('UID collision')
    return unique, linkage


def select_repeats(unique: list[dict], count: int) -> list[str]:
    if count < 0:
        raise ValueError('repeat-count must be nonnegative')
    uids = [row['uid'] for row in unique]
    return sorted(uids, key=lambda uid: sha(('repeat:v1:' + uid).encode()))[:count]


def make_request(item: dict, prompt: str, schema: dict, config: dict) -> dict:
    # Whitelisting here prevents accidental leakage when upstream tables gain columns.
    api_input = {field: item[field] for field in sorted(INPUT_FIELDS)}
    strict_schema = {key: value for key, value in schema.items()
                     if key not in {'$schema', 'title'}}
    parameters = config['parameters']
    request = {'model': config['model'], 'instructions': prompt,
               'input': json.dumps(api_input, ensure_ascii=False, sort_keys=True),
               'text': {'format': {'type': 'json_schema', 'name': 'student_effort_evidence',
                                   'schema': strict_schema, 'strict': True}},
               'temperature': parameters['temperature'], 'store': False,
               'max_output_tokens': parameters['max_output_tokens']}
    if parameters['reasoning_effort'] is not None:
        request['reasoning'] = {'effort': parameters['reasoning_effort']}
    return request


def prepare(run_dir: Path, model: str, *, inputs: Path | None = None,
            prompt: Path | None = None, schema: Path | None = None,
            base_url: str = 'https://api.openai.com/v1', temperature: float = 0,
            reasoning_effort: str | None = 'none', repeat_count: int = 100,
            max_output_tokens: int = 1024) -> dict:
    if not re.search(r'-\d{4}-\d{2}-\d{2}$', model):
        raise ValueError('Use an explicit dated model snapshot, not a moving model alias')
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme != 'https' or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('base-url must be HTTPS without credentials/query/fragment')
    if not 0 <= temperature <= 2 or max_output_tokens < 1:
        raise ValueError('Invalid temperature or token limit')
    inputs = inputs or PACKAGE / 'data/semantic/full_first_turn_inputs.jsonl'
    prompt = prompt or PACKAGE / 'documentation/semantic/scoring_prompt.txt'
    schema = schema or PACKAGE / 'documentation/semantic/scoring_schema.json'
    config = {'model': model, 'base_url': base_url.rstrip('/'),
              'parameters': {'temperature': temperature, 'reasoning_effort': reasoning_effort,
                             'max_output_tokens': max_output_tokens, 'store': False}}
    manifest_path = run_dir / 'manifest.json'
    if manifest_path.exists():
        old = read_json(manifest_path)
        if any(old[key] != value for key, value in config.items()) or old['repeat_count'] != min(repeat_count, old['unique_inputs']):
            raise ValueError('Existing run has different immutable configuration; use a new run ID')
        verify_run(run_dir, require_complete=False)
        return old
    input_bytes, prompt_bytes, schema_bytes = inputs.read_bytes(), prompt.read_bytes(), schema.read_bytes()
    rows = [json.loads(line) for line in input_bytes.decode('utf-8').splitlines() if line]
    unique, linkage = deduplicate(rows)
    if not unique:
        raise ValueError('No scoring inputs')
    repeats = select_repeats(unique, repeat_count)
    run_dir.mkdir(parents=True, exist_ok=True)
    files = {'full_inputs.jsonl': input_bytes, 'scoring_prompt.txt': prompt_bytes,
             'scoring_schema.json': schema_bytes, 'unique_inputs.jsonl': json_lines(unique),
             'duplicate_linkage.json': json_bytes(linkage),
             'code/openai_scoring.py': Path(__file__).read_bytes()}
    for item in unique:
        body = make_request(item, prompt_bytes.decode('utf-8'), json.loads(schema_bytes), config)
        files[f"requests/{item['uid']}.json"] = json_bytes(body)
    for name, data in files.items():
        write_once(run_dir / name, data)
    manifest = {'format_version': 2, 'run_id': run_dir.name, 'prepared_utc': now(), **config,
                'input_records': len(rows), 'unique_inputs': len(unique),
                'repeat_count': len(repeats), 'repeat_uids': repeats,
                'repeat_selection': 'first N unique UIDs ordered by SHA256("repeat:v1:" + uid), selected before scoring',
                'input_fields': sorted(INPUT_FIELDS),
                'duplication_policy': 'Exact (problem_text, student_text, after_assistant) only; representative item_id=min(original IDs)',
                'primary_policy': 'First valid successful response per UID; repeats never replace or average primary ratings',
                'files': {name: sha(data) for name, data in sorted(files.items())},
                'source_files': {'inputs_sha256': sha(input_bytes), 'prompt_sha256': sha(prompt_bytes),
                                 'schema_sha256': sha(schema_bytes)},
                'provenance': {'python': sys.version, 'platform': platform.platform(),
                               'scorer_code_sha256': sha(Path(__file__).read_bytes()),
                               'client': 'Python stdlib urllib; OpenAI Responses API',
                               'api_endpoint': config['base_url'] + '/responses'}}
    write_once(manifest_path, json_bytes(manifest))
    return manifest


def validate_score(score: dict, item: dict) -> dict:
    if not isinstance(score, dict) or set(score) != SCORE_FIELDS:
        raise ValueError('Score must have exactly the original eight schema fields')
    if score['item_id'] != item['item_id'] or type(score['after_assistant']) is not bool or score['after_assistant'] != item['after_assistant']:
        raise ValueError('Response input ID/flag mismatch')
    for name in ('learning_score', 'self_work_score'):
        value = score[name]
        if name == 'learning_score' and value is None:
            continue
        if type(value) is not int or not 0 <= value <= 3:
            raise ValueError('Invalid score range/type')
    if score['intent'] not in INTENTS or score['confidence'] not in {'high', 'medium', 'low'}:
        raise ValueError('Invalid intent/confidence')
    quote = score['evidence_quote']
    if not isinstance(quote, str) or len(quote) > 160 or quote not in item['student_text']:
        raise ValueError('Evidence quote is not an exact student-text substring')
    if not quote and item['student_text']:
        raise ValueError('Nonempty text needs nonempty evidence quote')
    if not isinstance(score['reason'], str) or not score['reason'].strip():
        raise ValueError('Missing semantic reason')
    return score


def extract_score(raw: bytes, item: dict, model: str) -> tuple[dict, dict]:
    response = json.loads(raw.decode('utf-8'))
    if not isinstance(response, dict) or response.get('status') != 'completed' or response.get('error'):
        raise ValueError('Response not completed')
    if response.get('model') != model:
        raise ValueError('Returned model differs from frozen snapshot')
    texts = []
    for output in response.get('output', []):
        if output.get('type') != 'message':
            continue
        for content in output.get('content', []):
            if content.get('type') == 'refusal':
                raise ValueError('Model refused the scoring task')
            if content.get('type') == 'output_text':
                texts.append(content.get('text', ''))
    if len(texts) != 1:
        raise ValueError('Expected one structured output text')
    score = validate_score(json.loads(texts[0]), item)
    return score, response


def _context(run_dir: Path) -> tuple[dict, dict[str, dict]]:
    manifest = read_json(run_dir / 'manifest.json')
    if manifest.get('format_version') != 2:
        raise ValueError('Unknown manifest version')
    for name, digest in manifest['files'].items():
        path = safe_path(run_dir, name)
        if not path.is_file() or sha(path.read_bytes()) != digest:
            raise ValueError(f'Frozen artifact missing/changed: {name}')
    rows = read_lines(run_dir / 'full_inputs.jsonl')
    unique, linkage = deduplicate(rows)
    if unique != read_lines(run_dir / 'unique_inputs.jsonl') or linkage != read_json(run_dir / 'duplicate_linkage.json'):
        raise ValueError('Deduplication/linkage does not replay exactly')
    if len(rows) != manifest['input_records'] or len(unique) != manifest['unique_inputs']:
        raise ValueError('Input counts differ')
    if select_repeats(unique, manifest['repeat_count']) != manifest['repeat_uids']:
        raise ValueError('Deterministic repeat selection differs')
    prompt = (run_dir / 'scoring_prompt.txt').read_bytes().decode('utf-8')
    schema = read_json(run_dir / 'scoring_schema.json')
    for item in unique:
        expected = json_bytes(make_request(item, prompt, schema, manifest))
        if (run_dir / f"requests/{item['uid']}.json").read_bytes() != expected:
            raise ValueError('Saved request is inconsistent with frozen blind input/config')
    return manifest, {row['uid']: row for row in unique}


def _replay(run_dir: Path, manifest: dict, items: dict) -> tuple[dict, dict, list[dict]]:
    first: dict[tuple[str, str], dict] = {}
    seen_ids: set[str] = set()
    attempts = read_lines(run_dir / 'attempts.jsonl')
    manifest_digest = sha((run_dir / 'manifest.json').read_bytes())
    for attempt in attempts:
        attempt_id, uid, phase = attempt['attempt_id'], attempt['uid'], attempt['phase']
        if attempt_id in seen_ids or uid not in items or phase not in {'primary', 'repeat'}:
            raise ValueError('Invalid/duplicate attempt identity')
        if phase == 'repeat' and uid not in manifest['repeat_uids']:
            raise ValueError('Unregistered repeat UID')
        seen_ids.add(attempt_id)
        request_name = f'requests/{uid}.json'
        if attempt['request_file'] != request_name or attempt['request_sha256'] != manifest['files'][request_name] or attempt['manifest_sha256'] != manifest_digest:
            raise ValueError('Attempt request/manifest digest mismatch')
        executor_path = safe_path(run_dir, attempt['executor_code_file'])
        if not executor_path.exists() or sha(executor_path.read_bytes()) != attempt['executor_code_sha256']:
            raise ValueError('Execution code missing/changed')
        response = None
        valid = False
        if attempt.get('response_file'):
            response_path = safe_path(run_dir, attempt['response_file'])
            if not response_path.exists() or sha(response_path.read_bytes()) != attempt['response_sha256']:
                raise ValueError('Raw response missing/changed')
            try:
                score, response = extract_score(response_path.read_bytes(), items[uid], manifest['model'])
                valid = True
            except (ValueError, TypeError, KeyError):
                pass
        if bool(attempt['valid']) != valid:
            raise ValueError('Logged validation disagrees with raw response replay')
        if valid:
            if attempt.get('returned_model') != response['model'] or attempt.get('response_id') != response.get('id') or attempt.get('usage') != response.get('usage'):
                raise ValueError('Saved response metadata differs from raw response')
            first.setdefault((phase, uid), score)
    primary = {uid: score for (phase, uid), score in first.items() if phase == 'primary'}
    repeats = {uid: score for (phase, uid), score in first.items() if phase == 'repeat'}
    return primary, repeats, attempts


def verify_run(run_dir: Path, require_complete: bool = True) -> dict:
    run_dir = Path(run_dir)
    manifest, items = _context(run_dir)
    primary, repeats, attempts = _replay(run_dir, manifest, items)
    complete = len(primary) == len(items) and len(repeats) == manifest['repeat_count']
    paired = sorted(set(primary) & set(repeats))
    agreement = {}
    for field in ('learning_score', 'self_work_score', 'intent', 'confidence'):
        agreements = sum(primary[uid][field] == repeats[uid][field] for uid in paired)
        agreement[field] = {'n': len(paired), 'equal': agreements,
                            'fraction': agreements / len(paired) if paired else None}
    score_fields = ('learning_score', 'self_work_score', 'intent', 'confidence')
    agreement['all_rating_fields'] = {'n': len(paired), 'equal': sum(
        all(primary[uid][field] == repeats[uid][field] for field in score_fields) for uid in paired)}
    agreement['all_rating_fields']['fraction'] = agreement['all_rating_fields']['equal'] / len(paired) if paired else None
    result = {'complete': complete, 'primary_expected': len(items), 'primary_valid': len(primary),
              'repeat_expected': manifest['repeat_count'], 'repeat_valid': len(repeats),
              'attempt_count': len(attempts), 'valid_attempts': sum(bool(row['valid']) for row in attempts),
              'returned_models': sorted({row['returned_model'] for row in attempts if row['valid']}),
              'manifest_sha256': sha((run_dir / 'manifest.json').read_bytes()),
              'repeat_agreement': agreement}
    freeze_path = run_dir / 'freeze.json'
    if freeze_path.exists():
        freeze = read_json(freeze_path)
        if freeze['manifest_sha256'] != result['manifest_sha256']:
            raise ValueError('Frozen manifest changed')
        for name, digest in freeze['files'].items():
            path = safe_path(run_dir, name)
            if not path.exists() or sha(path.read_bytes()) != digest:
                raise ValueError(f'Frozen run artifact changed: {name}')
        if freeze['verification'] != result:
            raise ValueError('Frozen verification differs from replay')
    if require_complete and not complete:
        raise ValueError(f"Incomplete scoring: primary {len(primary)}/{len(items)}, repeats {len(repeats)}/{manifest['repeat_count']}")
    return result


def load_primary_scores(run_dir: Path) -> list[dict]:
    run_dir = Path(run_dir)
    verify_run(run_dir, require_complete=True)
    manifest, items = _context(run_dir)
    primary, _, _ = _replay(run_dir, manifest, items)
    return [{'uid': uid, **primary[uid]} for uid in sorted(primary)]


def freeze(run_dir: Path) -> dict:
    result = verify_run(run_dir, require_complete=True)
    path = run_dir / 'freeze.json'
    if path.exists():
        return read_json(path)
    manifest = read_json(run_dir / 'manifest.json')
    names = set(manifest['files']) | {'manifest.json'}
    if (run_dir / 'attempts.jsonl').exists():
        names.add('attempts.jsonl')
    for attempt in read_lines(run_dir / 'attempts.jsonl'):
        if attempt.get('response_file'):
            names.add(attempt['response_file'])
        names.add(attempt['executor_code_file'])
    frozen = {'format_version': 2, 'frozen_utc': now(),
              'manifest_sha256': result['manifest_sha256'],
              'files': {name: sha(safe_path(run_dir, name).read_bytes()) for name in sorted(names)},
              'verification': result}
    write_once(path, json_bytes(frozen))
    return frozen


def get_api_key() -> str:
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not key and os.name == 'nt':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as env:
                key = str(winreg.QueryValueEx(env, 'OPENAI_API_KEY')[0]).strip()
        except (OSError, ImportError):
            pass
    if not key:
        raise ValueError('OPENAI_API_KEY is not configured in process or Windows user environment')
    return key


class AuthenticationFailure(RuntimeError):
    pass


def append_attempt(path: Path, attempt: dict) -> None:
    with LOCK:
        with path.open('ab') as handle:
            handle.write(json_lines([attempt]))
            handle.flush()
            os.fsync(handle.fileno())


def execute(run_dir: Path, *, workers: int = 4, retries: int = 3,
            timeout: float = 120, max_units: int | None = None) -> dict:
    if workers < 1 or workers > 32 or retries < 0 or timeout <= 0:
        raise ValueError('Invalid execution limits')
    if (run_dir / 'freeze.json').exists():
        return verify_run(run_dir, require_complete=True)
    manifest, items = _context(run_dir)
    primary, repeats, _ = _replay(run_dir, manifest, items)
    key = get_api_key()
    lock_path = run_dir / 'execute.lock'
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(lock_fd, str(os.getpid()).encode())
    os.close(lock_fd)
    stop = threading.Event()
    jobs = [('primary', uid) for uid in sorted(items) if uid not in primary]
    jobs += [('repeat', uid) for uid in manifest['repeat_uids'] if uid not in repeats]
    if max_units is not None:
        jobs = jobs[:max_units]
    (run_dir / 'responses').mkdir(exist_ok=True)
    manifest_digest = sha((run_dir / 'manifest.json').read_bytes())
    executor_bytes = Path(__file__).read_bytes()
    executor_digest = sha(executor_bytes)
    executor_name = f'execution_code/{executor_digest}.py'
    write_once(run_dir / executor_name, executor_bytes)

    def work(phase: str, uid: str) -> None:
        for retry in range(retries + 1):
            if stop.is_set():
                return
            request_name = f'requests/{uid}.json'
            body = (run_dir / request_name).read_bytes()
            attempt_id = uuid.uuid4().hex
            entry = {'attempt_id': attempt_id, 'uid': uid, 'phase': phase,
                     'retry_index': retry, 'requested_utc': now(),
                     'manifest_sha256': manifest_digest, 'request_file': request_name,
                     'executor_code_file': executor_name, 'executor_code_sha256': executor_digest,
                     'request_sha256': sha(body), 'http_status': None, 'request_id': None,
                     'response_id': None, 'returned_model': None, 'usage': None,
                     'response_file': None, 'response_sha256': None, 'valid': False,
                     'error_kind': None}
            start = time.monotonic()
            retryable = True
            authentication_error = False
            try:
                request = urllib.request.Request(manifest['base_url'] + '/responses', data=body,
                                                headers={'Authorization': 'Bearer ' + key,
                                                         'Content-Type': 'application/json'}, method='POST')
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    raw = response.read()
                    entry['http_status'] = response.status
                    entry['request_id'] = response.headers.get('x-request-id')
                response_name = f'responses/{attempt_id}.json'
                write_once(run_dir / response_name, raw)
                entry['response_file'], entry['response_sha256'] = response_name, sha(raw)
                parsed_metadata = json.loads(raw.decode('utf-8'))
                if isinstance(parsed_metadata, dict):
                    entry.update(response_id=parsed_metadata.get('id'), returned_model=parsed_metadata.get('model'),
                                 usage=parsed_metadata.get('usage'))
                    entry['system_fingerprint'] = parsed_metadata.get('system_fingerprint')
                score, parsed = extract_score(raw, items[uid], manifest['model'])
                entry.update(valid=True, response_id=parsed.get('id'), returned_model=parsed.get('model'),
                             usage=parsed.get('usage'))
            except urllib.error.HTTPError as error:
                entry['http_status'] = error.code
                entry['request_id'] = error.headers.get('x-request-id') if error.headers else None
                # Error bodies can echo authentication fragments: store hash/count only.
                error_bytes = error.read()
                entry['http_error_body_sha256'] = sha(error_bytes)
                entry['http_error_body_bytes'] = len(error_bytes)
                entry['error_kind'] = 'http_' + str(error.code)
                retryable = error.code == 429 or 500 <= error.code <= 599
                authentication_error = error.code in {401, 403}
                if authentication_error:
                    stop.set()
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
                entry['error_kind'] = type(error).__name__
            except (ValueError, TypeError, KeyError) as error:
                entry['error_kind'] = 'invalid_structured_response:' + type(error).__name__
            finally:
                entry['finished_utc'] = now()
                entry['elapsed_seconds'] = round(time.monotonic() - start, 6)
                append_attempt(run_dir / 'attempts.jsonl', entry)
            if authentication_error:
                raise AuthenticationFailure('API authentication failed (401/403); attempts saved, configure credentials locally')
            if entry['valid'] or not retryable or retry == retries:
                return
            stop.wait(min(2 ** retry, 30))

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, phase, uid) for phase, uid in jobs]
            for future in as_completed(futures):
                try:
                    future.result()
                except AuthenticationFailure:
                    stop.set()
                    for queued in futures:
                        queued.cancel()
                    raise
    finally:
        lock_path.unlink(missing_ok=True)
    return verify_run(run_dir, require_complete=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'execute', 'verify', 'freeze'):
        command = commands.add_parser(name)
        command.add_argument('--run-id', default='openai_rescore_v2')
        command.add_argument('--run-root', type=Path, default=DEFAULT_ROOT)
        if name == 'prepare':
            command.add_argument('--model', required=True, help='Exact dated model snapshot')
            command.add_argument('--base-url', default='https://api.openai.com/v1')
            command.add_argument('--temperature', type=float, default=0)
            command.add_argument('--reasoning-effort', default='none')
            command.add_argument('--repeat-count', type=int, default=100)
            command.add_argument('--max-output-tokens', type=int, default=1024)
        elif name == 'execute':
            command.add_argument('--workers', type=int, default=4)
            command.add_argument('--retries', type=int, default=3)
            command.add_argument('--timeout', type=float, default=120)
            command.add_argument('--max-units', type=int)
        elif name == 'verify':
            command.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id) or args.run_id in {'.', '..'}:
        parser.error('Invalid run ID')
    run_dir = args.run_root / args.run_id
    try:
        if args.command == 'prepare':
            result = prepare(run_dir, args.model, base_url=args.base_url,
                             temperature=args.temperature, reasoning_effort=args.reasoning_effort,
                             repeat_count=args.repeat_count, max_output_tokens=args.max_output_tokens)
            output = {key: result[key] for key in ('run_id', 'model', 'input_records', 'unique_inputs', 'repeat_count')}
            output['manifest_sha256'] = sha((run_dir / 'manifest.json').read_bytes())
        elif args.command == 'execute':
            output = execute(run_dir, workers=args.workers, retries=args.retries,
                             timeout=args.timeout, max_units=args.max_units)
        elif args.command == 'verify':
            output = verify_run(run_dir, require_complete=not args.allow_incomplete)
        else:
            output = freeze(run_dir)['verification']
        print(json.dumps(output, ensure_ascii=False, indent=2))
    except (ValueError, OSError, AuthenticationFailure) as error:
        parser.exit(1, f'{type(error).__name__}: {error}\n')


if __name__ == '__main__':
    main()
