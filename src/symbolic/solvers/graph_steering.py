"""SYM-2: Learned Graph-Steering (Soft Preference) Layer with Deterministic Fallback.

This layer produces soft heuristic preference scores based on graph topology.
It incorporates a lightweight RGCN/GraphSAGE-inspired NumPy model for learned steering.
It has absolutely zero authority to declare a candidate feasible or infeasible.
"""

import os
from typing import Dict, Tuple
from .graph_model import InfrastructureGraph
from .graph_gnn import NumPyGraphModel

class GraphSteeringLayer:
    """Produces soft preferences for candidate region topologies."""

    def __init__(self, graph: InfrastructureGraph, model_path: str = "src/symbolic/solvers/gnn_checkpoint.json"):
        self.graph = graph
        self.model = None
        self._learned_embeddings = None
        
        # Attempt to load the learned model
        if os.path.exists(model_path):
            try:
                self.model = NumPyGraphModel()
                self.model.load(model_path)
                self._learned_embeddings = self.model.forward(self.graph)
                print(f"[GraphSteering] Loaded learned RGCN/GraphSAGE-inspired steering model from {model_path}.")
            except Exception as e:
                print(f"[GraphSteering] Error loading learned model: {e}. Falling back to deterministic heuristic.")
                self.model = None
        else:
            print(f"[GraphSteering] Learned model checkpoint '{model_path}' not found. Using deterministic fallback.")

    def score_pair(self, id_a: str, id_b: str) -> float:
        """
        Calculates a soft preference score for a pair of regions.
        Higher score = worse preference (cost penalty).
        
        Uses learned model if available, else deterministic fallback.
        """
        if self.model and self._learned_embeddings is not None:
            # Learned steering
            try:
                return self.model.score_pair(self._learned_embeddings, id_a, id_b)
            except Exception as e:
                print(f"[GraphSteering] Learned scoring failed during inference: {e}. Using fallback.")
                
        # Deterministic fallback
        return self._deterministic_score_pair(id_a, id_b)

    def _deterministic_score_pair(self, id_a: str, id_b: str) -> float:
        """Original deterministic fallback heuristic."""
        node_a = self.graph.get_node(id_a)
        node_b = self.graph.get_node(id_b)
        
        latency = self.graph.get_latency(id_a, id_b)
        
        # Base latency penalty
        penalty = latency * 1.0
        
        # Geo-diversity bonus (encourages disaster recovery separation)
        if node_a.geo != node_b.geo:
            penalty -= 50.0
            
        return penalty

    def get_all_pair_scores(self) -> Dict[Tuple[str, str], float]:
        """Returns all pairwise heuristic scores for Z3 soft objective optimization."""
        scores = {}
        nodes = self.graph.get_all_nodes()
        
        for a in nodes:
            for b in nodes:
                scores[(a.id, b.id)] = self.score_pair(a.id, b.id)
                
        return scores
