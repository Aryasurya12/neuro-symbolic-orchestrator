"""Independent checker: no imports from the system under test; no config argument."""
import math
import re


def finite(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError('expected finite number')
    return float(x)


def check(output, truth, catalog):
    """Grade normalized decisions against human truth and a frozen catalog.

    Allocation format: {vms:[{provider,instance_type,count}]},
    {regions:[id,id]}, or {bandwidth_mbps:number,replicas:integer}.
    Never use solver-reported feasibility, totals, latency, or SLA as evidence.
    """
    def verdict(passed, satisfied, error, reason):
        return dict(passed=passed, constraint_satisfied=satisfied,
                    cost_error_pct=error, reason=reason)
    if output is None:
        return verdict(None, None, None, 'pending_blind_transcription')
    expected = truth['expected_status']
    status = output.get('self_reported_status', '').lower()
    resources = output.get('allocated_resources', {})
    if expected != 'feasible':
        # Human-authored expected infeasibility/clarification/unsupported is the oracle.
        # A crash never counts as a correct abstention. A proposed allocation contradicts it.
        ok = status == expected and not resources
        return verdict(ok, None, None, 'correct_abstention' if ok else 'wrong_status_or_allocation')
    if status != 'feasible':
        return verdict(False, False, None, 'expected_feasible_allocation')
    c = truth['constraints']
    violations = []
    def require(ok, label):
        if not ok:
            violations.append(label)
    try:
        kind = truth['archetype']
        if kind == 'vm':
            if set(resources) != {'vms'}:
                raise ValueError('expected vms allocation only')
            items = resources['vms']
            if not isinstance(items, list) or not items:
                raise ValueError('empty allocation')
            by_key = {(s['provider'], s['name']): s for s in catalog['vms']}
            cost = cpu = ram = count_total = 0.0
            for item in items:
                n = finite(item['count'])
                if n <= 0 or not n.is_integer():
                    raise ValueError('VM count must be a positive integer')
                sku = by_key[(item['provider'], item['instance_type'])]
                require(sku['provider'] in c['providers'], 'provider')
                cost += sku['monthly_cost_usd'] * n
                cpu += sku['vcpus'] * n
                ram += sku['ram_gb'] * n
                count_total += n
            require(cpu >= c['min_vcpus'], 'vcpu')
            require(ram >= c['min_ram_gb'], 'ram')
            if 'exact_instances' in c:
                require(count_total == c['exact_instances'], 'instance_count')
            if 'min_vcpus_per_vm' in c:
                require(all(by_key[(i['provider'], i['instance_type'])]['vcpus'] >= c['min_vcpus_per_vm'] for i in items), 'vcpu_per_vm')
            if 'min_ram_gb_per_vm' in c:
                require(all(by_key[(i['provider'], i['instance_type'])]['ram_gb'] >= c['min_ram_gb_per_vm'] for i in items), 'ram_per_vm')
        elif kind == 'dr':
            if set(resources) != {'regions'} or len(resources['regions']) != 2:
                raise ValueError('expected exactly two region IDs')
            a, b = resources['regions']
            if a == b:
                raise ValueError('duplicate region')
            regions = {r['id']: r for r in catalog['regions']}
            x, y = regions[a], regions[b]
            require(x['provider'] in c['providers'] and y['provider'] in c['providers'], 'provider')
            if c['cross_provider']:
                require(x['provider'] != y['provider'], 'cross_provider')
            if c.get('distinct_geo'):
                require(x['geo'] != y['geo'], 'distinct_geo')
            latency = x['peer_latencies_ms'][b]
            # Declared benchmark model, not a real-world uptime guarantee.
            sla = 100 * (1 - (1-x['sla_pct']/100) * (1-y['sla_pct']/100))
            cost = x['base_cost_usd'] + y['base_cost_usd'] + latency * catalog['dr_latency_cost_per_ms']
            require(latency <= c['max_latency_ms'], 'latency')
            require(sla >= c['min_sla_pct'], 'sla')
        elif kind == 'scaling':
            if set(resources) != {'bandwidth_mbps', 'replicas'}:
                raise ValueError('expected bandwidth and replicas')
            bw, replicas = finite(resources['bandwidth_mbps']), finite(resources['replicas'])
            if replicas < 1 or not replicas.is_integer():
                raise ValueError('replicas must be a positive integer')
            require(c['min_bandwidth_mbps'] <= bw <= c['max_bandwidth_mbps'], 'bandwidth')
            require(replicas <= c['max_replicas'], 'replica_limit')
            model = catalog['scaling']
            cost = bw * model['usd_per_mbps'] + replicas * model['usd_per_replica']
            # Deliberately not clipped to 99%: clipping hides under-provisioning.
            cpu = bw / (replicas * model['capacity_mbps']) * 100
            require(cpu <= c['max_cpu_pct'], 'cpu')
        else:
            raise ValueError('unsupported grading archetype')
        require(cost <= c['budget_max_usd'] + 1e-8, 'budget')
        constraints_ok = not violations
        if 'optimal_cost_usd' in truth:
            require(abs(cost - truth['optimal_cost_usd']) <= truth.get('optimality_tolerance_usd', 0.02), 'optimality')
        claimed = finite(output['claimed_cost_usd'])
        error = abs(claimed-cost) / cost * 100 if cost else (0.0 if claimed == 0 else None)
        require(abs(claimed-cost) <= truth.get('cost_tolerance_usd', 0.02), 'cost_claim')
        return verdict(not violations, constraints_ok, error, ','.join(violations) or 'ok')
    except (ValueError, KeyError, TypeError, ZeroDivisionError, OverflowError) as exc:
        return verdict(False, False, None, 'invalid_allocation: ' + str(exc))


CATEGORIES = {'formal_english','colloquial_hinglish','missing_field','conflicting_constraints','out_of_archetype'}
FIELDS = {
    'vm': ({'providers','min_vcpus','min_ram_gb','budget_max_usd'}, {'exact_instances','min_vcpus_per_vm','min_ram_gb_per_vm'}),
    'dr': ({'providers','cross_provider','max_latency_ms','min_sla_pct','budget_max_usd'}, {'distinct_geo'}),
    'scaling': ({'min_bandwidth_mbps','max_bandwidth_mbps','max_replicas','max_cpu_pct','budget_max_usd'}, set()),
}


def validate_dataset(dataset, development_fixture=False):
    if not dataset.get('ground_truth_author') or (not development_fixture and not dataset.get('approved_by_user')):
        raise ValueError('Human ground truth and approval are required before freezing')
    seen = set()
    for q in dataset['queries']:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',q['query_id']) or q['query_id'] in seen or q['query_category'] not in CATEGORIES:
            raise ValueError('duplicate ID or invalid category')
        seen.add(q['query_id'])
        if not q['raw_query_text'].strip() or not q['trials'] or any(type(t) is not int or t < 1 for t in q['trials']) or len(q['trials']) != len(set(q['trials'])):
            raise ValueError('invalid query/trials')
        t = q['truth']
        if not isinstance(t, dict) or not t.get('rationale'):
            raise ValueError('Every query needs human-written truth and rationale')
        if t['expected_status'] not in {'feasible','infeasible','clarification','unsupported'}:
            raise ValueError('invalid expected_status')
        if t['expected_status'] == 'feasible':
            required, optional = FIELDS[t['archetype']]
            keys = set(t['constraints'])
            if not required <= keys or keys-required-optional:
                raise ValueError('Missing or unsupported truth constraint; do not silently ignore it')
