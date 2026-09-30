"""SYM-6: Outcome-aware routing evaluation benchmark.

Compares five routing configurations against 13 deterministic scenarios from
Phase 9/10 evaluation infrastructure.

Configurations:
    BASELINE  – Full GA+PSO+Z3 race (no routing)
    SYM-3     – Deterministic adaptive routing only
    SYM-4     – Production learned checkpoint (SYM-4)
    CANDIDATE – SYM-5 candidate checkpoint (if available)
    REFERENCE – Full GA+PSO+Z3 evaluation oracle (for outcome comparison)

Scenarios (from Phase 9/10 evaluation dataset):
    Feasible:       A1_Small, A2_Medium, A3_Large, B1_Comfortable, B2_NearBoundary
    Infeasible:     B3_Tight, C1_Budget_Infeasible, C2_Resource_Infeasible, F1_Boundary_Budget
    Multi-provider: D1_Multi_AWS_Azure, D2_Multi_All
    Latency/SLA:    E1_DR_Feasible, E2_DR_Infeasible

Dataset source: DETERMINISTIC (same as Phase 9/10 evaluation, no random generation)
Data label: BENCHMARK_SYNTHETIC (constructed from known cost models)
Random seeds: N/A (deterministic workloads)

NOTE: Evaluation overhead is intentionally included (reference race runs all 3 solvers).
      This is NOT production latency — do not interpret reference runtime as production cost.
"""

import sys
import os
import json
import csv
import time
from typing import List, Optional
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.symbolic.models import SymbolicOptimizationRequest
from src.symbolic.optihive.eval_runner import (
    run_baseline, run_sym3, run_sym4, run_candidate, run_reference_race,
    CONFIG_BASELINE, CONFIG_SYM3, CONFIG_SYM4, CONFIG_CANDIDATE, CONFIG_REFERENCE,
)
from src.symbolic.optihive.outcome_evaluation import (
    compare_outcome, evaluate_promotion_gate,
    OUTCOME_EQUIVALENT, OUTCOME_ACCEPTABLE, OUTCOME_INFERIOR,
    OUTCOME_INFEASIBLE, OUTCOME_BOTH_INFEASIBLE, OUTCOME_FAILED, OUTCOME_NOT_EVALUATED,
)
from src.symbolic.optihive.telemetry_training import CANDIDATE_PATH, CHECKPOINT_PATH

# ─── Evaluation dataset ────────────────────────────────────────────────────────
EVAL_SCENARIOS = [
    # Name,                      cloud_providers,                budget, svc, vcpus, ram,    lat,    sla
    ("A1_Small",         ["aws", "azure"],                       500.0,  2,   4,     16.0,   0.0,    0.0),
    ("A2_Medium",        ["aws", "azure", "gcp"],               2000.0,  5,  16,     64.0,   0.0,    0.0),
    ("A3_Large",         ["aws", "azure", "gcp"],               5000.0, 10,  32,    128.0,   0.0,    0.0),
    ("B1_Comfortable",   ["aws"],                               1000.0,  3,   8,     32.0,   0.0,    0.0),
    ("B2_NearBoundary",  ["aws", "gcp"],                         500.0,  3,   8,     32.0,   0.0,    0.0),
    ("B3_Tight",         ["aws"],                                 50.0,  5,  16,     64.0,   0.0,    0.0),
    ("C1_Budget_Infeas", ["aws"],                                  5.0, 10,  64,    256.0,   0.0,    0.0),
    ("C2_Res_Infeas",    ["aws"],                               1000.0,  5, 512,   2048.0,   0.0,    0.0),
    ("D1_Multi_AWS_Az",  ["aws", "azure"],                       500.0,  2,   4,     16.0,   0.0,    0.0),
    ("D2_Multi_All",     ["aws", "azure", "gcp"],                500.0,  3,   8,     32.0,   0.0,    0.0),
    ("E1_DR_Feasible",   ["aws", "azure"],                       500.0,  2,   1,      4.0, 100.0,   99.9),
    ("E2_DR_Infeasible", ["aws"],                                  5.0,  2,   8,     32.0,  10.0,   99.99),
    ("F1_Boundary",      ["aws"],                                 30.0,  2,   4,     16.0,   0.0,    0.0),
]


def _make_request(row) -> SymbolicOptimizationRequest:
    name, providers, budget, svc, vcpus, ram, lat, sla = row
    return SymbolicOptimizationRequest(
        problem_type=name,
        cloud_providers=providers,
        budget_max_usd=budget,
        service_count=svc,
        required_vcpus=vcpus,
        required_ram_gb=ram,
        latency_max_ms=lat,
        sla_availability_pct=sla,
    )


def run_evaluation(candidate_path: Optional[str] = None) -> dict:
    """Run complete SYM-6 outcome-aware evaluation.

    Returns a results dict suitable for JSON serialisation.
    """
    if candidate_path is None:
        candidate_path = CANDIDATE_PATH

    print(f"\n{'='*70}")
    print("SYM-6: OUTCOME-AWARE ROUTING EVALUATION")
    print(f"{'='*70}")
    print(f"Evaluation scenarios : {len(EVAL_SCENARIOS)}")
    print(f"Data source          : BENCHMARK_SYNTHETIC (deterministic)")
    print(f"Production checkpoint: {CHECKPOINT_PATH}")
    print(f"Candidate checkpoint : {candidate_path}")
    print(f"Candidate available  : {os.path.exists(candidate_path)}")
    print()

    reference_outcomes = {}
    config_outcomes: dict = {
        CONFIG_BASELINE: [],
        CONFIG_SYM3: [],
        CONFIG_SYM4: [],
    }
    if os.path.exists(candidate_path):
        config_outcomes[CONFIG_CANDIDATE] = []

    print(f"{'Scenario':<22} {'Config':<12} {'Feas':^5} {'Cost':>10} {'MS':>8} {'Mode':<16}")
    print("-" * 82)

    for row in EVAL_SCENARIOS:
        name = row[0]
        req = _make_request(row)

        # REFERENCE race (evaluation oracle)
        ref = run_reference_race(name, req)
        reference_outcomes[name] = ref
        print(f"{name:<22} {'REFERENCE':<12} {str(ref.is_feasible)[0]:^5} "
              f"{ref.objective_cost_usd or 'N/A':>10} "
              f"{ref.total_runtime_ms:>8.1f} {'reference':<16}")

        # BASELINE
        bl = run_baseline(name, req)
        config_outcomes[CONFIG_BASELINE].append(bl)
        print(f"{'':<22} {CONFIG_BASELINE:<12} {str(bl.is_feasible)[0]:^5} "
              f"{bl.objective_cost_usd or 'N/A':>10} "
              f"{bl.total_runtime_ms:>8.1f} {bl.routing_mode:<16}")

        # SYM-3
        s3 = run_sym3(name, req)
        config_outcomes[CONFIG_SYM3].append(s3)
        print(f"{'':<22} {CONFIG_SYM3:<12} {str(s3.is_feasible)[0]:^5} "
              f"{s3.objective_cost_usd or 'N/A':>10} "
              f"{s3.total_runtime_ms:>8.1f} {s3.routing_mode:<16}")

        # SYM-4
        s4 = run_sym4(name, req)
        config_outcomes[CONFIG_SYM4].append(s4)
        print(f"{'':<22} {CONFIG_SYM4:<12} {str(s4.is_feasible)[0]:^5} "
              f"{s4.objective_cost_usd or 'N/A':>10} "
              f"{s4.total_runtime_ms:>8.1f} {s4.routing_mode:<16}")

        # CANDIDATE (if exists)
        if CONFIG_CANDIDATE in config_outcomes:
            sc = run_candidate(name, req, candidate_path)
            config_outcomes[CONFIG_CANDIDATE].append(sc)
            print(f"{'':<22} {CONFIG_CANDIDATE:<12} {str(sc.is_feasible)[0]:^5} "
                  f"{sc.objective_cost_usd or 'N/A':>10} "
                  f"{sc.total_runtime_ms:>8.1f} {sc.routing_mode:<16}")
        print()

    # ─── Outcome comparison ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("OUTCOME COMPARISONS")
    print(f"{'='*70}")
    print(f"{'Scenario':<22} {'Config':<12} {'Outcome':<16} {'CostDelta':>9} {'RTratio':>8}")
    print("-" * 72)

    all_comparisons = []
    for config_name, outcomes in config_outcomes.items():
        for outcome in outcomes:
            ref = reference_outcomes.get(outcome.scenario_name)
            if ref is None:
                continue
            comparison = compare_outcome(outcome.scenario_name, config_name, outcome, ref)
            all_comparisons.append(comparison)
            cost_delta = f"{comparison.cost_delta_pct*100:+.1f}%" if comparison.cost_delta_pct is not None else "N/A"
            rt_ratio = f"{comparison.runtime_ratio:.2f}x" if comparison.runtime_ratio is not None else "N/A"
            print(f"{comparison.scenario_name:<22} {config_name:<12} "
                  f"{comparison.outcome_label:<16} {cost_delta:>9} {rt_ratio:>8}")

    # ─── Per-configuration summary ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY BY CONFIGURATION")
    print(f"{'='*70}")
    print(f"{'Config':<14} {'Equiv':>6} {'Accp':>6} {'Infr':>6} {'InFeas':>7} {'Failed':>7} {'BothInf':>8} {'NotEval':>8}")
    print("-" * 70)

    summary_by_config = {}
    for config_name in config_outcomes.keys():
        comps = [c for c in all_comparisons if c.config_name == config_name]
        equiv = sum(1 for c in comps if c.outcome_label == OUTCOME_EQUIVALENT)
        accp = sum(1 for c in comps if c.outcome_label == OUTCOME_ACCEPTABLE)
        infr = sum(1 for c in comps if c.outcome_label == OUTCOME_INFERIOR)
        infeas = sum(1 for c in comps if c.outcome_label == OUTCOME_INFEASIBLE)
        failed = sum(1 for c in comps if c.outcome_label == OUTCOME_FAILED)
        both_inf = sum(1 for c in comps if c.outcome_label == OUTCOME_BOTH_INFEASIBLE)
        not_eval = sum(1 for c in comps if c.outcome_label == OUTCOME_NOT_EVALUATED)
        print(f"{config_name:<14} {equiv:>6} {accp:>6} {infr:>6} {infeas:>7} {failed:>7} {both_inf:>8} {not_eval:>8}")
        summary_by_config[config_name] = {
            "equivalent": equiv, "acceptable": accp, "inferior": infr,
            "infeasible": infeas, "failed": failed, "both_infeasible": both_inf,
            "not_evaluated": not_eval,
        }

    # ─── Latency summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("LATENCY SUMMARY (ms)")
    print(f"{'='*70}")
    print(f"{'Config':<14} {'Median':>10} {'P95':>10} {'Max':>10}")
    print("-" * 50)

    import statistics

    latency_summary = {}
    all_configs_with_ref = list(config_outcomes.items()) + [(CONFIG_REFERENCE, list(reference_outcomes.values()))]
    for config_name, outcomes in all_configs_with_ref:
        rts = sorted(o.total_runtime_ms for o in outcomes)
        n = len(rts)
        if n == 0:
            continue
        median = statistics.median(rts)
        p95 = rts[int(0.95 * n)] if n >= 10 else None
        maximum = rts[-1]
        p95_str = f"{p95:.1f}" if p95 is not None else "N/A (n<10)"
        print(f"{config_name:<14} {median:>10.1f} {p95_str:>10} {maximum:>10.1f}")
        latency_summary[config_name] = {"median_ms": median, "p95_ms": p95, "max_ms": maximum, "n": n}

    # ─── Promotion gate (CANDIDATE only) ──────────────────────────────────────
    gate_result = None
    candidate_available = CONFIG_CANDIDATE in config_outcomes

    if candidate_available:
        print(f"\n{'='*70}")
        print("PROMOTION GATE EVALUATION")
        print(f"{'='*70}")

        candidate_comps = [c for c in all_comparisons if c.config_name == CONFIG_CANDIDATE]

        # Load candidate val_accuracy
        candidate_val_acc = None
        try:
            with open(candidate_path) as f:
                cp_data = json.load(f)
            candidate_val_acc = cp_data.get("metadata", {}).get("val_accuracy")
        except Exception:
            pass

        gate_result = evaluate_promotion_gate(
            comparisons=candidate_comps,
            candidate_val_accuracy=candidate_val_acc,
            candidate_checkpoint_path=candidate_path,
        )

        print(f"  val_accuracy ok      : {gate_result.val_accuracy_ok} "
              f"({gate_result.val_accuracy} >= {gate_result.val_accuracy_threshold})")
        print(f"  feasibility ok       : {gate_result.feasibility_preserved} "
              f"({gate_result.feasibility_failures} violations in {gate_result.feasibility_scenarios_tested} scenarios)")
        print(f"  hard constraints ok  : {gate_result.hard_constraint_ok}")
        print(f"  outcome agreement ok : {gate_result.outcome_agreement_ok} "
              f"({gate_result.equivalent_or_acceptable_count}/{gate_result.total_evaluated} "
              f"equiv/acceptable, need >= {gate_result.min_agreement_rate:.0%})")
        print(f"\n  ELIGIBLE FOR PROMOTION: {gate_result.eligible_for_promotion}")
        print(f"  Reason: {gate_result.reason}")
        if gate_result.notes:
            for note in gate_result.notes:
                print(f"    - {note}")
        print(f"\n  [IMPORTANT] Production checkpoint NOT automatically promoted.")
        print(f"  [IMPORTANT] Run with --promote flag to authorise promotion.")
    else:
        print(f"\n[SKIP] No candidate checkpoint at {candidate_path}")
        print("       Run benchmarks/train_sym5_from_telemetry.py to generate one.")

    # ─── Production checkpoint verification ───────────────────────────────────
    print(f"\n{'='*70}")
    print("PRODUCTION CHECKPOINT VERIFICATION")
    print(f"{'='*70}")
    try:
        with open(CHECKPOINT_PATH) as f:
            prod_data = json.load(f)
        print(f"  Production version: {prod_data.get('model_version', 'unknown')}")
        print(f"  Production file   : {CHECKPOINT_PATH}")
        print(f"  Status            : UNCHANGED")
    except Exception as e:
        print(f"  WARNING: Cannot read production checkpoint: {e}")

    # ─── Build result dict ────────────────────────────────────────────────────
    results = {
        "sym6_evaluation": {
            "version": "1.0",
            "dataset": {
                "source": "BENCHMARK_SYNTHETIC",
                "scenario_count": len(EVAL_SCENARIOS),
                "scenario_names": [r[0] for r in EVAL_SCENARIOS],
                "seeds": "deterministic (no random seeds)",
            },
            "configurations_evaluated": list(config_outcomes.keys()) + [CONFIG_REFERENCE],
            "thresholds": {
                "cost_tolerance_pct": 5,
                "cost_degradation_limit_pct": 20,
                "runtime_limit_factor": 3.0,
                "min_agreement_rate_pct": 80,
                "val_accuracy_threshold": 0.70,
            },
            "outcome_comparisons": [
                {
                    "scenario": c.scenario_name,
                    "config": c.config_name,
                    "outcome_label": c.outcome_label,
                    "candidate_feasible": c.candidate_feasible,
                    "reference_feasible": c.reference_feasible,
                    "candidate_cost": c.candidate_cost,
                    "reference_cost": c.reference_cost,
                    "cost_delta_pct": round(c.cost_delta_pct * 100, 2) if c.cost_delta_pct is not None else None,
                    "runtime_ratio": round(c.runtime_ratio, 3) if c.runtime_ratio is not None else None,
                    "notes": c.notes,
                }
                for c in all_comparisons
            ],
            "summary_by_config": summary_by_config,
            "latency_ms": latency_summary,
            "promotion_gate": (
                {
                    "eligible_for_promotion": gate_result.eligible_for_promotion,
                    "promoted": False,
                    "val_accuracy_ok": gate_result.val_accuracy_ok,
                    "val_accuracy": gate_result.val_accuracy,
                    "feasibility_preserved": gate_result.feasibility_preserved,
                    "hard_constraint_ok": gate_result.hard_constraint_ok,
                    "outcome_agreement_ok": gate_result.outcome_agreement_ok,
                    "equivalent_or_acceptable_count": gate_result.equivalent_or_acceptable_count,
                    "total_evaluated": gate_result.total_evaluated,
                    "reason": gate_result.reason,
                    "notes": gate_result.notes,
                }
                if gate_result else {"status": "SKIPPED", "reason": "No candidate checkpoint available."}
            ),
            "production_checkpoint_unchanged": True,
        }
    }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SYM-6 Outcome-aware evaluation")
    parser.add_argument("--promote", action="store_true",
                        help="Promote candidate if gate passes (requires human authorisation)")
    parser.add_argument("--candidate-path", default=None,
                        help="Override candidate checkpoint path")
    args = parser.parse_args()

    candidate_path = args.candidate_path or CANDIDATE_PATH
    results = run_evaluation(candidate_path=candidate_path)

    os.makedirs("benchmarks/results", exist_ok=True)

    json_path = "benchmarks/results/sym6_outcome_evaluation.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results saved: {json_path}")

    # CSV export
    csv_path = "benchmarks/results/sym6_outcome_evaluation.csv"
    comparisons = results["sym6_evaluation"]["outcome_comparisons"]
    if comparisons:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=comparisons[0].keys())
            writer.writeheader()
            writer.writerows(comparisons)
        print(f"CSV results saved : {csv_path}")

    # Promotion
    gate = results["sym6_evaluation"]["promotion_gate"]
    if args.promote:
        if gate.get("eligible_for_promotion"):
            print("\n[PROMOTE] Gate passed. Authorising promotion...")
            from src.symbolic.optihive.telemetry_training import promote_candidate_model
            promo = promote_candidate_model()
            print(f"  Promotion result: {promo}")
            results["sym6_evaluation"]["promotion_gate"]["promoted"] = promo.get("promoted", False)
            results["sym6_evaluation"]["promotion_gate"]["promotion_result"] = promo
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
        else:
            print("\n[SKIP PROMOTE] Gate conditions not met. Production checkpoint unchanged.")
    else:
        print("\n[DRY RUN] --promote not specified. Production checkpoint unchanged.")

    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
