import pytest
from src.symbolic.solvers.graph_model import InfrastructureGraph
from src.symbolic.solvers.graph_steering import GraphSteeringLayer

def test_graph_steering_preferences():
    graph = InfrastructureGraph()
    steering = GraphSteeringLayer(graph, model_path="dummy")
    
    # us-east-1 and eastus are both US_East geo, latency 12.0
    # Penalty: 12.0
    score_same_geo = steering.score_pair("us-east-1", "eastus")
    assert score_same_geo == 12.0
    
    # us-east-1 (US_East) and eu-west-1 (Europe), latency 85.0
    # Penalty: 85.0 - 50.0 = 35.0
    score_diff_geo = steering.score_pair("us-east-1", "eu-west-1")
    assert score_diff_geo == 35.0

def test_graph_steering_all_pairs():
    graph = InfrastructureGraph()
    steering = GraphSteeringLayer(graph, model_path="dummy")
    scores = steering.get_all_pair_scores()
    
    assert len(scores) == 25
    assert ("us-east-1", "eastus") in scores
