"""SYM-6: Outcome-aware evaluation CLI (alias for benchmark script).

Usage:
    python benchmarks/evaluate_sym6_candidate.py [--promote] [--candidate-path PATH]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.benchmark_sym6_outcome_evaluation import main
if __name__ == "__main__":
    main()
