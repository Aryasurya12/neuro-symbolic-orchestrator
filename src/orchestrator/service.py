from src.semantic.scope_parser import SCOPEParser
from src.semantic.explainer import FinOpsExplainer
from src.semantic.schemas import CloudOptimizationContract
from src.symbolic.adapters import from_contract, to_explainer_dict
from src.symbolic.optimizers.genetic_algorithm import GeneticAlgorithm
from src.symbolic.optimizers.particle_swarm import ParticleSwarmOptimization
from src.symbolic.solvers.z3_solver import GraphSteeredZ3Solver
from src.symbolic.optihive.solver_selection import OptiHiveSelector
import time

class NeuroSymbolicOrchestrator:
    """
    Central Orchestrator integrating the Semantic (Part A) and Symbolic (Part B) layers.
    Routes unstructured natural language down to constraint satisfaction, then explains 
    the result back in natural language.
    """
    def __init__(self):
        self.parser = SCOPEParser()
        self.selector = OptiHiveSelector(seed=42)

    def process_query(self, user_query: str) -> str:
        """
        Executes the end-to-end flow from natural language to final explanation report.
        """
        # 1. Semantic Layer: Parse Query -> Structured Contract
        contract, matched_template, score = self.parser.parse_query_to_contract(user_query)
        
        # 2. Optimization Layer
        result_dict = self.optimize_contract(contract)

        # 3. Explanation Layer: Result -> Natural Language Report
        report = FinOpsExplainer.generate_report(contract, result_dict)
        return report

    def optimize_contract(self, contract: CloudOptimizationContract) -> dict:
        """
        Executes the symbolic engines based on problem type and returns
        the result as a dictionary ready for the explainer.
        """
        sym_req = from_contract(contract)
        candidates = []

        if contract.problem_type == "ILP_VM_Allocation":
            ga_solver = GeneticAlgorithm(random_seed=42)
            ga_res = ga_solver.solve(sym_req)
            candidates.append(ga_res)
        
        elif contract.problem_type == "PSO_Continuous_Scaling":
            pso_solver = ParticleSwarmOptimization(random_seed=42)
            pso_res = pso_solver.solve(sym_req)
            candidates.append(pso_res)

        elif contract.problem_type == "Z3_Graph_Disaster_Recovery":
            z3_solver = GraphSteeredZ3Solver()
            z3_res = z3_solver.solve(sym_req)
            candidates.append(z3_res)

        else:
            # Fallback or hybrid
            ga_solver = GeneticAlgorithm(random_seed=42)
            pso_solver = ParticleSwarmOptimization(random_seed=42)
            z3_solver = GraphSteeredZ3Solver()
            
            candidates.append(ga_solver.solve(sym_req))
            candidates.append(pso_solver.solve(sym_req))
            candidates.append(z3_solver.solve(sym_req))

        # 3. OptiHive-Inspired Selection
        if len(candidates) == 1:
            final_result = candidates[0]
        else:
            final_result = self.selector.select(sym_req, candidates)

        if not final_result:
            return {
                "status": "Infeasible",
                "solver": "OptiHive",
                "error_message": "No feasible candidate found by any solver."
            }

        # 4. Adapter Mapping
        result_dict = to_explainer_dict(final_result, contract)
        return result_dict
