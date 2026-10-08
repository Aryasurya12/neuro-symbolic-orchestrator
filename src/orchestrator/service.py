from src.semantic.scope_parser import SCOPEParser
from src.semantic.explainer import FinOpsExplainer
from src.semantic.schemas import CloudOptimizationContract
from src.symbolic.adapters import from_contract, to_explainer_dict
from src.symbolic.optimizers.genetic_algorithm import GeneticAlgorithm
from src.symbolic.optimizers.particle_swarm import ParticleSwarmOptimization
from src.symbolic.solvers.z3_solver import GraphSteeredZ3Solver
from src.symbolic.solvers.graph_model import InfrastructureGraph
from src.symbolic.solvers.graph_steering import GraphSteeringLayer
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
        
        # Hoist graph and GraphSteeringLayer model weights to __init__ stage
        # so weights are loaded ONCE at startup rather than reloaded from disk on every query.
        self.graph = InfrastructureGraph()
        self.graph_steering_layer = GraphSteeringLayer(self.graph)
        
        self.ga_solver = GeneticAlgorithm(random_seed=42)
        self.pso_solver = ParticleSwarmOptimization(random_seed=42)
        self.z3_solver = GraphSteeredZ3Solver(
            steering_layer=self.graph_steering_layer,
            graph=self.graph,
        )

    def process_query(self, user_query: str, mode: int = 4, offline: bool = True, enable_llm_explainer: bool = False) -> str:
        """
        Executes the end-to-end flow from natural language to final explanation report.
        Mode 4 uses NVIDIA neural requirement interpretation.
        Mode 3 uses local rule-based SCOPE parsing.
        """
        if mode == 4:
            from src.semantic.nvidia_extractor import NVIDIAExtractor
            extractor = NVIDIAExtractor()
            ext_res = extractor.extract_contract_from_query(user_query, offline=offline)
            if not ext_res.is_executable:
                if ext_res.clarification_questions:
                    questions_formatted = "\n- " + "\n- ".join(ext_res.clarification_questions)
                    return f"[NVIDIA Requirement Interpretation - Clarification Needed]\nStatus: {ext_res.outcome}\nQuestions required to proceed:{questions_formatted}"
                return f"[NVIDIA Requirement Interpretation - {ext_res.outcome.upper()}]\nReason: {ext_res.error_message or 'Unsupported or conflicting requirements'}"
            contract = ext_res.contract
        else:
            # 1. Semantic Layer: Parse Query -> Structured Contract (Mode 3 Rule-based)
            contract, matched_template, score = self.parser.parse_query_to_contract(user_query)
            
            # Stage 1 Logging: Indicate why baseline sizing was selected if qualitative intent was detected
            if contract.metadata and "qualitative_intent" in contract.metadata:
                intent = contract.metadata["qualitative_intent"]
                print(f"🎯 [Stage 1] Qualitative Intent Detected: '{intent}' -> Baseline set to {contract.required_vcpus} vCPUs / {contract.required_ram_gb:.0f}GB RAM")

        # 2. Optimization Layer
        result_dict = self.optimize_contract(contract)

        # 3. Explanation Layer: Result -> Natural Language Report
        report = FinOpsExplainer.generate_report(
            contract, result_dict, enable_llm_explainer=enable_llm_explainer, offline=offline
        )
        return report

    def optimize_contract(self, contract: CloudOptimizationContract) -> dict:
        """
        Executes the symbolic engines and evaluates the solver candidates
        using the statistical latent-class EM quality model in OptiHiveSelector.
        """
        sym_req = from_contract(contract)
        candidates = []

        # Stage 4: Solver execution / Parallel Race
        if contract.problem_type == "ILP_VM_Allocation":
            candidates.append(self.ga_solver.solve(sym_req))
        elif contract.problem_type == "PSO_Continuous_Scaling":
            candidates.append(self.pso_solver.solve(sym_req))
        elif contract.problem_type == "Z3_Graph_Disaster_Recovery":
            candidates.append(self.z3_solver.solve(sym_req))
        else:
            candidates.append(self.ga_solver.solve(sym_req))
            candidates.append(self.pso_solver.solve(sym_req))
            candidates.append(self.z3_solver.solve(sym_req))

        # Stage 5: OptiHive Latent-Class EM Solver Selection
        final_result = self.selector.select(sym_req, candidates)

        if not final_result or not final_result.is_feasible:
            # Check if source candidates have specific constraint_status / error_message
            primary_candidate = candidates[0] if candidates else None
            if primary_candidate and not primary_candidate.is_feasible:
                return to_explainer_dict(primary_candidate, contract)
            
            if final_result:
                return to_explainer_dict(final_result, contract)

            return {
                "status": "Infeasible",
                "solver": "OptiHive",
                "error_message": "No feasible candidate found by any solver."
            }

        # Stage 6: Adapter Mapping
        result_dict = to_explainer_dict(final_result, contract)
        return result_dict

