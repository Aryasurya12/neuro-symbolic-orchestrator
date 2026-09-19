"""SYM-2: Deterministic Graph-Steering (Soft Preference) Layer.

This layer produces soft heuristic preference scores based on graph topology.
It serves as an abstraction for future SAGE-GNN/RGCN models.
It has absolutely zero authority to declare a candidate feasible or infeasible.
"""

from typing import Dict, Tuple
from .graph_model import InfrastructureGraph, RegionNode

class GraphSteeringLayer:
    """Produces soft preferences for candidate region topologies."""

    def __init__(self, graph: InfrastructureGraph):
        self.graph = graph

    def score_pair(self, id_a: str, id_b: str) -> float:
        """
        Calculates a deterministic soft preference score for a pair of regions.
        Higher score = worse preference (we treat it as a cost penalty).
        
        Rules:
        - Latency penalty: 1.0 unit per ms
        - Geo-diversity: 50.0 bonus (negative penalty) if different geo
        """
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
