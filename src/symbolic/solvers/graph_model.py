"""SYM-2: Deterministic Graph Model representing cloud infrastructure topology."""

from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass(frozen=True)
class RegionNode:
    """Represents a cloud region in the infrastructure graph."""
    id: str
    provider: str
    geo: str
    base_cost_usd: float
    sla_pct: float
    peer_latencies_ms: Dict[str, float]

class InfrastructureGraph:
    """Deterministic graph representation of the cloud placement domain."""
    def __init__(self):
        self._nodes: Dict[str, RegionNode] = {}
        self._build_graph()

    def _build_graph(self):
        # We reuse the region definitions from the original prototype as the deterministic graph
        raw_regions = [
            {
                "id": "us-east-1", "provider": "AWS", "geo": "US_East",
                "base_cost_usd": 120.0, "sla_pct": 99.95,
                "peer_latencies_ms": {"us-west-2": 65.0, "eu-west-1": 85.0, "eastus": 12.0, "us-central1": 32.0},
            },
            {
                "id": "us-west-2", "provider": "AWS", "geo": "US_West",
                "base_cost_usd": 130.0, "sla_pct": 99.95,
                "peer_latencies_ms": {"us-east-1": 65.0, "eu-west-1": 135.0, "eastus": 70.0, "us-central1": 42.0},
            },
            {
                "id": "eu-west-1", "provider": "AWS", "geo": "Europe",
                "base_cost_usd": 140.0, "sla_pct": 99.95,
                "peer_latencies_ms": {"us-east-1": 85.0, "us-west-2": 135.0, "eastus": 90.0, "us-central1": 105.0},
            },
            {
                "id": "eastus", "provider": "Azure", "geo": "US_East",
                "base_cost_usd": 125.0, "sla_pct": 99.95,
                "peer_latencies_ms": {"us-east-1": 12.0, "us-west-2": 70.0, "eu-west-1": 90.0, "us-central1": 28.0},
            },
            {
                "id": "us-central1", "provider": "GCP", "geo": "US_Central",
                "base_cost_usd": 115.0, "sla_pct": 99.95,
                "peer_latencies_ms": {"us-east-1": 32.0, "us-west-2": 42.0, "eu-west-1": 105.0, "eastus": 28.0},
            },
        ]
        
        for r in raw_regions:
            self._nodes[r["id"]] = RegionNode(
                id=r["id"],
                provider=r["provider"],
                geo=r["geo"],
                base_cost_usd=r["base_cost_usd"],
                sla_pct=r["sla_pct"],
                peer_latencies_ms=r["peer_latencies_ms"]
            )

    def get_node(self, node_id: str) -> RegionNode:
        """Fetch a node deterministically."""
        return self._nodes[node_id]

    def get_all_nodes(self) -> List[RegionNode]:
        """Fetch all nodes."""
        return list(self._nodes.values())

    def get_latency(self, id_a: str, id_b: str) -> float:
        """Get latency between two nodes. Returns high latency if no direct edge."""
        if id_a == id_b:
            return 0.0
        
        try:
            node_a = self.get_node(id_a)
            if id_b in node_a.peer_latencies_ms:
                return node_a.peer_latencies_ms[id_b]
                
            node_b = self.get_node(id_b)
            if id_a in node_b.peer_latencies_ms:
                return node_b.peer_latencies_ms[id_a]
        except KeyError:
            pass
            
        return 999.0
