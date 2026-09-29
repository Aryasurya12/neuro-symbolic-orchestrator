"""Test script to verify qualitative intent parsing and precedence in SCOPEParser and Orchestrator."""

import sys
from src.semantic.scope_parser import SCOPEParser
from src.schemas.contract import CloudOptimizationContract
from src.orchestrator.service import NeuroSymbolicOrchestrator

def run_tests():
    parser = SCOPEParser()
    orchestrator = NeuroSymbolicOrchestrator()

    test_cases = [
        # 1. High Compute / Scaling
        ("Deploy a high compute workload in AWS with $400 budget", 4, 16.0, "High Compute Scaling"),
        ("Need heavy workload instances in GCP with $500 budget", 4, 16.0, "High Compute Scaling"),
        ("Optimize high compute scaling service on Azure with $600 budget", 4, 16.0, "High Compute Scaling"),
        ("Run intensive batch processing on AWS with $450 budget", 4, 16.0, "High Compute Scaling"),
        ("Deploy compute-intensive nodes on AWS under $700", 4, 16.0, "High Compute Scaling"),

        # 2. High Memory
        ("Deploy high memory service on AWS with $500 budget", 4, 32.0, "High Memory"),
        ("We need a memory intensive caching layer on GCP under $600", 4, 32.0, "High Memory"),
        ("Allocate large ram nodes on Azure with $800 budget", 4, 32.0, "High Memory"),

        # 3. Heavy Database / Enterprise Cluster
        ("Deploy heavy database on AWS with $1000 budget", 8, 32.0, "Heavy Database / Enterprise Cluster"),
        ("Provision database tier on GCP under $900", 8, 32.0, "Heavy Database / Enterprise Cluster"),
        ("Setup enterprise cluster on Azure with $1200 budget", 8, 32.0, "Heavy Database / Enterprise Cluster"),
        ("Allocate large cluster across AWS with $1500 budget", 8, 32.0, "Heavy Database / Enterprise Cluster"),

        # 4. Precedence: Explicit numbers override qualitative keywords
        ("Deploy heavy database with 16 vCPUs and 64GB RAM on AWS with $2000 budget", 16, 64.0, "Heavy Database / Enterprise Cluster"),
        ("Deploy high compute workload with 2 vCPUs on AWS with $300 budget", 2, 16.0, "High Compute Scaling"),
        ("Deploy high memory workload with 64GB RAM on GCP with $800 budget", 4, 64.0, "High Memory"),

        # 5. Fallback: No explicit numbers and no qualitative keywords
        ("Deploy a simple service on AWS with $300 budget", 1, 1.0, None),
    ]

    all_passed = True
    print("\n--- RUNNING SCOPE PARSER QUALITATIVE INTENT TESTS ---\n")
    for query, exp_vcpus, exp_ram, exp_intent in test_cases:
        contract, template, score = parser.parse_query_to_contract(query)
        act_intent = contract.metadata.get("qualitative_intent") if contract.metadata else None

        vcpus_ok = contract.required_vcpus == exp_vcpus
        ram_ok = contract.required_ram_gb == exp_ram
        intent_ok = act_intent == exp_intent

        passed = vcpus_ok and ram_ok and intent_ok
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_passed = False

        print(f"{status} | Query: '{query}'")
        print(f"      Expected: vCPUs={exp_vcpus}, RAM={exp_ram}GB, Intent='{exp_intent}'")
        print(f"      Actual:   vCPUs={contract.required_vcpus}, RAM={contract.required_ram_gb}GB, Intent='{act_intent}'")
        print(f"      Metadata: {contract.metadata}")

    print("\n--- RUNNING ORCHESTRATOR END-TO-END TEST ---\n")
    report = orchestrator.process_query("Deploy a high compute scaling workload on AWS with $500 budget")
    print(f"Orchestrator output preview (first 200 chars):\n{report[:200]}...")

    if all_passed:
        print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
    else:
        print("\n❌ SOME TESTS FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
