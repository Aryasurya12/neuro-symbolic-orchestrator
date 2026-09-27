"""SYM-2: Lightweight NumPy Graph Steering Model (RGCN/GraphSAGE-inspired)."""

import numpy as np
import json
import os
from typing import Dict, List, Tuple
from .graph_model import InfrastructureGraph, RegionNode

class NumPyGraphModel:
    """
    A lightweight, deterministic graph neural steering model.
    It implements a simplified RGCN/GraphSAGE message passing step using NumPy.
    
    Why NumPy? The infrastructure graph contains only 5 nodes. Introducing a massive
    PyTorch dependency to demonstrate topological graph steering on a 5-node graph 
    is unnecessary and violates dependency discipline. This implementation captures 
    the mathematically relevant relation-aware neighborhood aggregation (GraphSAGE) 
    and relational transformations (RGCN) purely via NumPy array ops.
    """
    def __init__(self, in_features: int = 3, hidden_dim: int = 8, out_features: int = 4):
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        
        # We define a few deterministic relations: 0: Same-Provider, 1: Cross-Provider, 2: Fast-Link (<50ms)
        self.num_relations = 3
        
        # Initialize deterministic weights
        rng = np.random.default_rng(42)
        
        # Relational weights for message passing: W_r for each relation
        self.W_r = rng.normal(0, 0.1, (self.num_relations, in_features, hidden_dim))
        
        # Self-loop weight
        self.W_self = rng.normal(0, 0.1, (in_features, hidden_dim))
        
        # Pair scoring mechanism (concatenated embeddings -> scalar score)
        self.W_score = rng.normal(0, 0.1, (hidden_dim * 2, 1))
        self.b_score = rng.normal(0, 0.1, (1,))

    def _extract_features(self, graph: InfrastructureGraph) -> Tuple[np.ndarray, Dict[str, int]]:
        nodes = graph.get_all_nodes()
        node_idx = {n.id: i for i, n in enumerate(nodes)}
        
        X = np.zeros((len(nodes), self.in_features))
        for i, n in enumerate(nodes):
            # Feature 0: Base cost (normalized roughly by 100)
            X[i, 0] = n.base_cost_usd / 100.0
            # Feature 1: SLA
            X[i, 1] = n.sla_pct / 100.0
            # Feature 2: Provider encoding (AWS=0.1, Azure=0.5, GCP=0.9)
            if n.provider == "AWS":
                X[i, 2] = 0.1
            elif n.provider == "Azure":
                X[i, 2] = 0.5
            elif n.provider == "GCP":
                X[i, 2] = 0.9
        
        return X, node_idx

    def _extract_edges(self, graph: InfrastructureGraph, node_idx: Dict[str, int]) -> Dict[int, List[Tuple[int, int]]]:
        nodes = graph.get_all_nodes()
        edges_by_rel = {r: [] for r in range(self.num_relations)}
        
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if i == j: continue
                
                lat = graph.get_latency(a.id, b.id)
                if a.provider == b.provider:
                    edges_by_rel[0].append((i, j))
                else:
                    edges_by_rel[1].append((i, j))
                
                if lat < 50.0:
                    edges_by_rel[2].append((i, j))
                    
        return edges_by_rel

    def forward(self, graph: InfrastructureGraph) -> np.ndarray:
        """
        Forward pass mimicking an RGCN layer.
        Returns node embeddings of shape (N, hidden_dim).
        """
        X, node_idx = self._extract_features(graph)
        edges_by_rel = self._extract_edges(graph, node_idx)
        
        N = X.shape[0]
        H = np.zeros((N, self.hidden_dim))
        
        # 1. Relational neighborhood aggregation
        for r in range(self.num_relations):
            if not edges_by_rel[r]: continue
            
            # Simple aggregation matrix A_r
            A_r = np.zeros((N, N))
            for src, dst in edges_by_rel[r]:
                A_r[dst, src] = 1.0  # Message flows src -> dst
            
            # Normalize neighborhood (mean aggregation - GraphSAGE style)
            row_sums = A_r.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            A_r_norm = A_r / row_sums
            
            # H += A_r_norm * X * W_r
            msg = np.dot(X, self.W_r[r])
            H += np.dot(A_r_norm, msg)
            
        # 2. Add self-loop (skip connection)
        H += np.dot(X, self.W_self)
        
        # 3. Activation
        H = np.maximum(0, H) # ReLU
        
        self._node_idx = node_idx
        return H

    def score_pair(self, H: np.ndarray, id_a: str, id_b: str) -> float:
        """
        Scores a pair of nodes using the generated embeddings.
        Output is a scalar "penalty" score (higher = worse) for Z3.
        """
        idx_a = self._node_idx[id_a]
        idx_b = self._node_idx[id_b]
        
        emb_a = H[idx_a]
        emb_b = H[idx_b]
        
        # Concatenate embeddings
        pair_emb = np.concatenate([emb_a, emb_b])
        
        score = np.dot(pair_emb, self.W_score) + self.b_score
        return float(score[0])

    def save(self, filepath: str):
        data = {
            "W_r": self.W_r.tolist(),
            "W_self": self.W_self.tolist(),
            "W_score": self.W_score.tolist(),
            "b_score": self.b_score.tolist()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load(self, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.W_r = np.array(data["W_r"])
        self.W_self = np.array(data["W_self"])
        self.W_score = np.array(data["W_score"])
        self.b_score = np.array(data["b_score"])

