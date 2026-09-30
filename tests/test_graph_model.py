import pytest
from src.symbolic.solvers.graph_model import InfrastructureGraph

def test_graph_initialization():
    graph = InfrastructureGraph()
    nodes = graph.get_all_nodes()
    assert len(nodes) == 5
    
    node = graph.get_node("us-east-1")
    assert node.provider == "AWS"
    assert node.geo == "US_East"

def test_graph_latency_lookup():
    graph = InfrastructureGraph()
    lat = graph.get_latency("us-east-1", "us-west-2")
    assert lat == 65.0
    
    # Missing direct link should return high penalty
    lat2 = graph.get_latency("non-existent", "non-existent-2")
    assert lat2 == 999.0
