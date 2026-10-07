"""Usage: python -m benchmarks.audit.run --help. No API calls during freeze/test."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from .checker import check, validate_dataset

ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp,path)


def append(path, row):
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(path, repair=False):
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b'\n'):
        if not repair:
            raise ValueError(f'Partial final line in {path}')
        boundary = data.rfind(b'\n') + 1
        # Preserve incomplete bytes for audit before truncation.
        path.with_name(path.name+'.partial-'+uuid.uuid4().hex).write_bytes(data[boundary:])
        with path.open('r+b') as f:
            f.truncate(boundary)
            f.flush()
            os.fsync(f.fileno())
        data = data[:boundary]
    return [json.loads(line) for line in data.splitlines()]


def provenance():
    files = {}
    for base in ['src','templates','config','benchmarks/audit']:
        for p in sorted((ROOT/base).rglob('*')):
            if p.is_file() and p.suffix in {'.py','.json'} and '__pycache__' not in p.parts:
                files[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    versions = {}
    for name in ['numpy','scipy','z3-solver','pydantic','openai','python-dotenv']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'MISSING'
    return dict(files=files,packages=versions,python=sys.version)


def freeze(args):
    from .adapters import snapshot_catalog
    dataset = read(args.dataset)
    validate_dataset(dataset, development_fixture=args.development_fixture)
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Freeze requires an empty output directory')
    plan = []
    for q in dataset['queries']:
        configs = q.get('configs','ABCD')
        if not configs or set(configs)-set('ABCD') or len(set(configs)) != len(configs):
            raise ValueError('invalid configs')
        for trial in q['trials']:
            for config in configs:
                plan.append(dict(query_id=q['query_id'],config=config,trial_num=trial))
    random.Random(args.seed).shuffle(plan)
    plan.sort(key=lambda j: j["trial_num"])
    manifest = dict(development_fixture=args.development_fixture,created_at=now(),dataset=dataset,catalog=snapshot_catalog(),plan=plan,
        provenance=provenance(),model=args.model,max_tokens=args.max_tokens,
        temperature=args.temperature,timeout_seconds=args.timeout,base_seed=args.seed,
        c_definition='rule parser + existing templates; not bare symbolic solver',
        d_definition='local parse_query_to_contract + optimize_contract + FinOpsExplainer once',
        latency_definition='cold subprocess end-to-end, initialization and explanation included; grading excluded',
        a_policy='plain prose; blind human transcription, no regex guessing',
        retry_policy='none; interruptions and errors are terminal records; new experiment for reruns')
    manifest['experiment_hash'] = digest(manifest)
    atomic(out/'manifest.json',manifest)
    print(f'Frozen {len(dataset["queries"])} queries; {len(plan)} runs; hash {manifest["experiment_hash"]}')


def verify(out):
    m = read(out/'manifest.json')
    unsigned = {k:v for k,v in m.items() if k != 'experiment_hash'}
    if digest(unsigned) != m['experiment_hash']:
        raise ValueError('Frozen manifest modified')
    if provenance() != m['provenance']:
        raise ValueError('Code/environment changed; create a new experiment, do not resume this one')
    from .adapters import snapshot_catalog
    if snapshot_catalog() != m['catalog']:
        raise ValueError('Catalog changed; refusing mixed-version run')
    return m


def job_key(j):
    return f"{j['query_id']}-{j['config']}-trial{j['trial_num']}"


def make_row(j,q,run_id,latency,output,exception,m):
    if exception:
        grade = dict(passed=False,constraint_satisfied=None,cost_error_pct=None,reason='execution_error')
    else:
        grade = check(output,q['truth'],m['catalog'])
    # NLU accuracy cannot be inferred from schema validity; leave unmeasured.
    return dict(run_id=run_id,query_id=j['query_id'],config=j['config'],trial_num=j['trial_num'],
        timestamp=now(),query_category=q['query_category'],input={'raw_query_text':q['raw_query_text']},
        execution=dict(nlu_success=None,exception=exception,latency_ms=latency),
        raw_output=output if output is not None else dict(allocated_resources={},claimed_cost_usd=None,self_reported_status='pending_transcription'),
        check_result=grade)


def worker(args):
    from .adapters import execute_local,execute_llm
    job = read(args.job)
    try:
        m, j, q = job['manifest'],job['job'],job['query']
        if j['config'] in 'AB':
            output, raw = execute_llm(j['config'],q['raw_query_text'],m['catalog'],m['model'],
                                     m['timeout_seconds'],m['max_tokens'],m['temperature'])
        else:
            output, raw = execute_local(j['config'],q['raw_query_text'],m['base_seed']+j['trial_num'])
        atomic(args.result,dict(output=output,raw=raw,exception=None))
    except Exception as exc:
        # SDK errors may contain user text but credentials are never included deliberately.
        atomic(args.result,dict(output=None,raw={},exception=f'{type(exc).__name__}: {exc}'))


def run(args):
    out = Path(args.out).resolve()
    m = verify(out)
    if not args.configs or set(args.configs)-set('ABCD'):
        raise ValueError('configs must be a subset of ABCD')
    if set(args.configs)&set('AB') and not os.environ.get('OPENROUTER_API_KEY'):
        raise ValueError('Set OPENROUTER_API_KEY before selecting A/B. No fabricated fallback.')
    lock = out/'RUNNING.lock'
    try:
        fd = os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError:
        raise RuntimeError('RUNNING.lock exists. Verify no runner is alive before removing a stale lock.')
    os.write(fd,str(os.getpid()).encode())
    os.close(fd)
    try:
        records = load_jsonl(out/'results.jsonl',True)
        completed = {job_key(r) for r in records}
        events = load_jsonl(out/'events.jsonl',True)
        starts = {job_key(e):e for e in events if e['event']=='started'}
        api_used = sum(e['config'] in 'AB' for e in starts.values())
        queries = {q['query_id']:q for q in m['dataset']['queries']}
        (out/'artifacts').mkdir(exist_ok=True)
        executed = 0
        for j in m['plan']:
            key = job_key(j)
            if j['config'] not in args.configs or key in completed:
                continue
            q = queries[j['query_id']]
            artifact = out/'artifacts'/f'{key}.json'
            if artifact.exists():
                # Crash after durable artifact but before JSONL append: no second execution.
                bundle = read(artifact)
                if bundle['experiment_hash'] != m['experiment_hash']:
                    raise ValueError('Artifact hash mismatch')
                append(out/'results.jsonl',bundle['row'])
                completed.add(key)
                continue
            if key in starts:
                # A call might have completed remotely. Never silently spend quota again.
                row = make_row(j,q,starts[key]['run_id'],None,None,
                               'Interrupted execution; remote completion unknown; not retried',m)
                atomic(artifact,dict(experiment_hash=m['experiment_hash'],row=row,raw={}))
                append(out/'results.jsonl',row)
                continue
            if args.max_runs is not None and executed >= args.max_runs:
                break
            if j['config'] in 'AB' and api_used >= args.max_api_calls:
                continue
            run_id = key+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
            append(out/'events.jsonl',dict(event='started',run_id=run_id,timestamp=now(),**j))
            if j['config'] in 'AB':
                api_used += 1
            request = out/'artifacts'/f'{key}.request.json'
            result_path = out/'artifacts'/f'{key}.response.json'
            atomic(request,dict(manifest=m,job=j,query={'raw_query_text':q['raw_query_text']}))
            start = time.perf_counter()
            capture = {}
            try:
                proc = subprocess.run([sys.executable,'-m','benchmarks.audit.run','_worker',
                       '--job',str(request),'--result',str(result_path)],cwd=ROOT,
                       capture_output=True,text=True,encoding='utf-8',errors='replace',
                       timeout=m['timeout_seconds']+10)
                capture = read(result_path) if result_path.exists() else dict(output=None,raw={},exception=f'Worker exit {proc.returncode}')
                capture['stdout'] = proc.stdout
                capture['stderr'] = proc.stderr
            except subprocess.TimeoutExpired as exc:
                capture = dict(output=None,raw={},exception='TimeoutExpired: worker deadline exceeded')
            elapsed = (time.perf_counter()-start)*1000
            row = make_row(j,q,run_id,elapsed,capture.get('output'),capture.get('exception'),m)
            atomic(artifact,dict(experiment_hash=m['experiment_hash'],row=row,raw=capture))
            append(out/'results.jsonl',row)
            print(key, row['check_result']['reason'],f'{elapsed:.1f}ms',flush=True)
            executed += 1
    finally:
        lock.unlink(missing_ok=True)


def review(args):
    out = Path(args.out)
    records = load_jsonl(out/'results.jsonl')
    if args.action == 'export':
        review_file = Path(args.file)
        if review_file.exists():
            raise ValueError('Will not overwrite existing transcription')
        entries = []
        for row in records:
            if row['check_result']['reason'] != 'pending_blind_transcription':
                continue
            bundle = read(out/'artifacts'/f'{job_key(row)}.json')
            response = bundle['raw']['raw']['response']
            entries.append(dict(run_id=row['run_id'],raw_text=response['choices'][0]['message']['content'],
                                reviewer='',answer=None,notes='Transcribe only. Do not repair or infer missing numbers.'))
        atomic(review_file,entries)
        print('Transcription file exported without query truth or grade')
    else:
        m = verify(out)
        originals = {r['run_id']:r for r in records}
        queries = {q['query_id']:q for q in m['dataset']['queries']}
        annotations = load_jsonl(out/'annotations.jsonl')
        done = {r['run_id'] for r in annotations}
        for entry in read(args.file):
            if entry['run_id'] in done:
                raise ValueError('Already annotated; no grade overwrites')
            row = originals[entry['run_id']]
            if row['check_result']['reason'] != 'pending_blind_transcription':
                raise ValueError('Only pending prose is eligible for transcription')
            if not entry['reviewer'] or not isinstance(entry['answer'],dict):
                raise ValueError('Reviewer and verbatim answer transcription required')
            grade = check(entry['answer'],queries[row['query_id']]['truth'],m['catalog'])
            append(out/'annotations.jsonl',dict(run_id=entry['run_id'],timestamp=now(),
                reviewer=entry['reviewer'],answer=entry['answer'],check_result=grade,notes=entry.get('notes','')))
            done.add(entry['run_id'])
        print('Append-only annotations saved; original results untouched')


def wilson(k,n):
    if not n:
        return None
    z=1.96
    p=k/n
    den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z*((p*(1-p)/n+z*z/(4*n*n))**0.5)/den
    return [mid-half,mid+half]


def summarize(args):
    out = Path(args.out)
    m = read(out/'manifest.json')
    rows = load_jsonl(out/'results.jsonl')
    annotations = {r['run_id']:r for r in load_jsonl(out/'annotations.jsonl')}
    for r in rows:
        if r['run_id'] in annotations:
            r['check_result'] = annotations[r['run_id']]['check_result']
    stats = {}
    # Primary estimate: trial 1 only, identical query set planned for all four configs.
    common_ids = {q['query_id'] for q in m['dataset']['queries'] if set(q.get('configs','ABCD')) == set('ABCD') and 1 in q['trials']}
    for config in 'ABCD':
        all_rows = [r for r in rows if r['config']==config]
        primary = [r for r in all_rows if r['trial_num']==1 and r['query_id'] in common_ids]
        passed = sum(r['check_result']['passed'] is True for r in primary)
        pending = sum(r['check_result']['passed'] is None for r in primary)
        missing = len(common_ids)-len(primary)
        times = [r['execution']['latency_ms'] for r in primary if r['execution']['latency_ms'] is not None]
        repeat = {}
        for qid in sorted({r['query_id'] for r in all_rows}):
            qs = [r for r in all_rows if r['query_id']==qid]
            if len(qs)>1:
                outputs = [annotations[r['run_id']]['answer'] if r['run_id'] in annotations else r['raw_output'] for r in qs]
                repeat[qid] = dict(trials=len(qs),distinct_normalized_outputs=len({digest(x) for x in outputs}),
                                  passes=sum(r['check_result']['passed'] is True for r in qs),
                                  pending=sum(r['check_result']['passed'] is None for r in qs),
                                  latency_ms=[r['execution']['latency_ms'] for r in qs])
        stats[config] = dict(primary_queries_planned=len(common_ids),primary_completed=len(primary),
            primary_passes=passed,pending=pending,not_run=missing,
            accuracy=passed/len(primary) if primary and not pending and not missing else None,
            illustrative_wilson_95=wilson(passed,len(primary)) if primary and not pending and not missing else None,
            completed_all_runs=len(all_rows),
            primary_cost_error_pct=[r['check_result']['cost_error_pct'] for r in primary],
            primary_execution_errors=sum(r['execution']['exception'] is not None for r in primary),
            latency_ms_median=statistics.median(times) if times else None,
            latency_ms_range=[min(times),max(times)] if times else None,
            categories={cat:dict(completed=sum(r['query_category']==cat for r in primary),
                passed=sum(r['query_category']==cat and r['check_result']['passed'] is True for r in primary)) for cat in sorted({q['query_category'] for q in m['dataset']['queries']})},
            repetitions=repeat,
            failures=[dict(query_id=r['query_id'],trial_num=r['trial_num'],reason=r['check_result']['reason'],exception=r['execution']['exception']) for r in all_rows if r['check_result']['passed'] is False])
    output=dict(experiment_hash=m['experiment_hash'],note='Wilson intervals are descriptive for this curated set, not population confidence. Repeats are not independent queries. Cold latency includes API errors/timeouts; inspect individual records.',configs=stats)
    atomic(out/'summary.json',output)
    print(json.dumps(output,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    f=sub.add_parser('freeze'); f.add_argument('--dataset',required=True); f.add_argument('--out',required=True)
    f.add_argument('--model',required=True); f.add_argument('--max-tokens',type=int,default=4096)
    f.add_argument('--temperature',type=float,default=0.0); f.add_argument('--timeout',type=float,default=360)
    f.add_argument('--seed',type=int,default=20261006)
    f.add_argument('--development-fixture',action='store_true',help='Technical testing only; never report as user-approved evaluation')
    r=sub.add_parser('run'); r.add_argument('--out',required=True); r.add_argument('--configs',default='CD')
    r.add_argument('--max-api-calls',type=int,default=40); r.add_argument('--max-runs',type=int)
    s=sub.add_parser('summary'); s.add_argument('--out',required=True)
    v=sub.add_parser('review'); v.add_argument('action',choices=['export','import']); v.add_argument('--out',required=True); v.add_argument('--file',required=True)
    w=sub.add_parser('_worker'); w.add_argument('--job',required=True); w.add_argument('--result',required=True)
    a=p.parse_args()
    {'freeze':freeze,'run':run,'summary':summarize,'review':review,'_worker':worker}[a.cmd](a)


if __name__=='__main__':
    main()
