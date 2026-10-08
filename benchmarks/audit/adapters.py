"""Adapters to inspected repo interfaces. No legacy grades or synthetic fallback."""
import json
import re
from dataclasses import asdict
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class Answer(BaseModel):
    model_config = ConfigDict(extra='forbid')
    allocated_resources: dict[str, Any]
    claimed_cost_usd: float | None
    self_reported_status: Literal['feasible','infeasible','clarification','unsupported']


def snapshot_catalog():
    from src.symbolic.optimizers.domain_catalog import VM_CATALOG
    from src.symbolic.solvers.graph_model import InfrastructureGraph
    from config.settings import settings
    return dict(
        vms=[dict(name=s.name, provider=s.provider, vcpus=s.vcpus, ram_gb=s.ram_gb,
                  monthly_cost_usd=s.monthly_cost()) for s in VM_CATALOG],
        regions=[asdict(r) for r in InfrastructureGraph().get_all_nodes()],
        dr_latency_cost_per_ms=settings.LATENCY_COST_PER_MS,
        scaling=dict(usd_per_mbps=0.08, usd_per_replica=45.0, capacity_mbps=75.0),
        notes='Repository sample benchmark catalog; 730-hour month. DR SLA assumes independent failures. Scaling CPU is a synthetic model.'
    )


def normalize_solver(result):
    status = str(result.get('status', 'unknown')).lower()
    # Preserve timeout/unknown distinctly: not evidence of true infeasibility.
    if 'unknown' in str(result.get('error_message', '')).lower():
        status = 'unknown'
    elif 'infeasible' in status:
        status = 'infeasible'
    elif status in {'feasible','optimal','converged'}:
        status = 'feasible'
    allocation = {}
    if status == 'feasible':
        if 'allocated_vms' in result:
            allocation = {'vms': [dict(provider=v['provider'],instance_type=v['instance_type'],count=v['count']) for v in result['allocated_vms']]}
        elif 'primary_region' in result:
            def region_id(text):
                # Existing template emits 'Provider:id (geo)'; orchestrator emits id.
                match = re.fullmatch(r'(?:AWS|Azure|GCP):([^ ]+) \(.*\)', text)
                return match.group(1) if match else text
            allocation = {'regions': [region_id(result['primary_region']),region_id(result['secondary_region'])]}
        elif 'optimal_bandwidth_mbps' in result:
            allocation = {'bandwidth_mbps': result['optimal_bandwidth_mbps'], 'replicas':result['recommended_replicas']}
    return dict(allocated_resources=allocation,
                claimed_cost_usd=result.get('total_monthly_cost_usd', result.get('estimated_monthly_cost_usd')),
                self_reported_status=status)


def execute_local(config, query, seed):
    """C = existing heuristic parsing + traditional templates (NOT bare solver).
    D = local orchestrator with exactly one parse and solve, including explanation.
    Fresh instances + explicit per-trial seeds remove resume/order dependence.
    """
    from src.semantic.scope_parser import SCOPEParser
    if config == 'D':
        import numpy as np
        from src.orchestrator.service import NeuroSymbolicOrchestrator
        from src.semantic.explainer import FinOpsExplainer
        engine = NeuroSymbolicOrchestrator()
        engine.ga_solver.rng = np.random.default_rng(seed)
        engine.pso_solver.rng = np.random.default_rng(seed)
        contract, template, score = engine.parser.parse_query_to_contract(query)
        result = engine.optimize_contract(contract)
        report = FinOpsExplainer.generate_report(contract, result)
        return normalize_solver(result), dict(contract=contract.model_dump(), template=template,
                                              score=score, result=result, report=report, seed=seed)
    parser = SCOPEParser()
    params = parser.extract_parameters(query)
    constraints = parser.extract_constraints_from_text(query)
    template = parser.matcher.match_template(constraints)[0] if constraints else 'ILP_VM_Allocation'
    # Same defaults and calls as FourWayBenchmarker.run_mode_3_pure_symbolic.
    budget = params.get('budget_max_usd',500.0)
    providers = params.get('cloud_providers',['AWS'])
    if template == 'PSO_Continuous_Scaling':
        from templates.Continuous_PSO_Dynamic_Scaling import solve_pso_continuous_scaling
        result = solve_pso_continuous_scaling(bandwidth_min_mbps=100.,bandwidth_max_mbps=1000.,
                   target_cpu_pct=70.,budget_max_usd=budget,random_seed=seed)
    elif template == 'Z3_Graph_Disaster_Recovery':
        from templates.Graph_SMT_Z3_MultiRegion_Placement import solve_z3_graph_disaster_recovery
        result = solve_z3_graph_disaster_recovery(sla_pct=params.get('sla_availability_pct',99.9),
                   max_latency_ms=params.get('latency_max_ms',100.),budget_max_usd=budget,target_providers=providers)
    else:
        from templates.ILP_VM_Knapsack_Allocation import solve_ilp_vm_knapsack
        result = solve_ilp_vm_knapsack(required_vcpus=params.get('required_vcpus',4),
                   required_ram_gb=params.get('required_ram_gb',16.),budget_max_usd=budget,target_providers=providers)
    return normalize_solver(result), dict(parameters=params,template=template,result=result,seed=seed)


def llm_messages(config, query, catalog):
    common = (
        'Solve the cloud request using ONLY this benchmark catalog and its declared cost models. '
        'State the concrete allocation and total USD monthly cost, or explicitly ask for clarification '
        'when essential constraints are missing/ambiguous, state infeasible for contradictory constraints, '
        'or unsupported if this catalog cannot represent the request. Do not silently substitute a different task. '
        'VM allocations need provider, exact instance_type and positive integer count. '
        'DR allocations need two region IDs; scaling needs bandwidth_mbps and integer replicas. '
        'Minimize cost subject to all user constraints. Do not invent prices. '
        'Return your final answer, without a reasoning preamble.\nCATALOG:\n' + json.dumps(catalog)
    )
    if config == 'A':
        common += '\nRespond in plain natural language. No JSON or prescribed schema.'
    else:
        common += '\nReturn only JSON conforming to this schema:\n' + json.dumps(Answer.model_json_schema())
        common += '\nallocated_resources is {"vms":[{"provider":"AWS","instance_type":"t3.medium","count":1}]} '
        common += 'or {"regions":["region-id-1","region-id-2"]} or {"bandwidth_mbps":100,"replicas":2}. '
        common += 'For abstention use allocated_resources={} and claimed_cost_usd=null.'
    return [{'role':'system','content':common},{'role':'user','content':query}]


def execute_llm(config, query, catalog, model, timeout, tokens, temperature):
    import os
    from openai import OpenAI
    key = os.environ.get('GROQ_API_KEY') or os.environ.get('OPENROUTER_API_KEY')
    base_url = os.environ.get('GROQ_BASE_URL', 'https://api.groq.com/openai/v1')
    if not key:
        raise RuntimeError('GROQ_API_KEY (or OPENROUTER_API_KEY) absent; no synthetic fallback')
    # No hidden SDK retry, model fallback, automatic truncation retry or reasoning setting switch.
    client = OpenAI(base_url=base_url, api_key=key, timeout=timeout, max_retries=0)
    messages = llm_messages(config,query,catalog)
    response = client.chat.completions.create(model=model,messages=messages,temperature=temperature,max_tokens=tokens)
    raw = dict(messages=messages,response=response.model_dump(mode='json'))
    choice = response.choices[0]
    if choice.finish_reason != 'stop':
        return dict(allocated_resources={},claimed_cost_usd=None,self_reported_status='generation_'+str(choice.finish_reason)),raw
    if config == 'A':
        # Do not turn a fragile regex miss into an apparent reasoning failure.
        return None,raw
    try:
        answer = Answer.model_validate_json(choice.message.content or '').model_dump()
    except Exception as exc:
        raw['normalization_error'] = str(exc)
        answer = dict(allocated_resources={},claimed_cost_usd=None,self_reported_status='invalid_schema')
    return answer,raw
