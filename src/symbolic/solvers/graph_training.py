"""SYM-2: Training Pipeline for Lightweight Graph Steering Model."""

import numpy as np
import os
from .graph_model import InfrastructureGraph
from .graph_gnn import NumPyGraphModel

def generate_synthetic_data(graph: InfrastructureGraph):
    """
    Generate synthetic deterministic training data representing Z3 optimization outcomes.
    Input: pair of nodes (id_a, id_b)
    Target: 1.0 if they are a highly preferred pair (low latency, different geo)
            0.0 if they are a terrible pair (high latency, same geo)
            0.5 for moderate pairs
    """
    nodes = graph.get_all_nodes()
    dataset = []
    
    for a in nodes:
        for b in nodes:
            if a.id == b.id: continue
            
            lat = graph.get_latency(a.id, b.id)
            
            # Oracle heuristic: preferred pairs are diverse geo + low latency
            target = 0.5
            if a.geo != b.geo and lat < 80.0:
                target = 1.0 # Optimal DR
            elif a.geo == b.geo and lat < 20.0:
                target = 0.8 # Good local pair
            elif lat > 100.0:
                target = 0.0 # Terrible latency
                
            dataset.append((a.id, b.id, target))
            
    # Deterministic shuffle
    rng = np.random.default_rng(42)
    rng.shuffle(dataset)
    
    # Split: Train (60%), Val (20%), Test (20%)
    n = len(dataset)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    
    return dataset[:train_end], dataset[train_end:val_end], dataset[val_end:]

def pack_params(model: NumPyGraphModel) -> np.ndarray:
    return np.concatenate([
        model.W_r.flatten(),
        model.W_self.flatten(),
        model.W_score.flatten(),
        model.b_score.flatten()
    ])

def unpack_params(model: NumPyGraphModel, params: np.ndarray):
    idx = 0
    size = model.W_r.size
    model.W_r = params[idx:idx+size].reshape(model.W_r.shape)
    idx += size
    
    size = model.W_self.size
    model.W_self = params[idx:idx+size].reshape(model.W_self.shape)
    idx += size
    
    size = model.W_score.size
    model.W_score = params[idx:idx+size].reshape(model.W_score.shape)
    idx += size
    
    size = model.b_score.size
    model.b_score = params[idx:idx+size].reshape(model.b_score.shape)

def compute_loss(model: NumPyGraphModel, graph: InfrastructureGraph, data: list) -> float:
    # Forward pass
    H = model.forward(graph)
    loss = 0.0
    for id_a, id_b, target in data:
        # Our model outputs a "penalty" score. 
        # Target = 1.0 (preferred) -> penalty should be low (e.g. 0.0)
        # Target = 0.0 (terrible) -> penalty should be high (e.g. 100.0)
        expected_penalty = (1.0 - target) * 100.0
        
        pred = model.score_pair(H, id_a, id_b)
        loss += (pred - expected_penalty) ** 2
    return loss / len(data)

def train_model():
    print("--- Training Lightweight RGCN/GraphSAGE-inspired Model ---")
    graph = InfrastructureGraph()
    train_data, val_data, test_data = generate_synthetic_data(graph)
    
    print(f"Dataset Split: {len(train_data)} Train, {len(val_data)} Val, {len(test_data)} Test")
    
    model = NumPyGraphModel()
    
    # Simple Finite Difference Gradient Descent (since parameters < 150)
    epochs = 150
    lr = 0.005
    epsilon = 1e-4
    
    best_val_loss = float('inf')
    best_params = None
    
    for epoch in range(epochs):
        params = pack_params(model)
        grad = np.zeros_like(params)
        
        base_loss = compute_loss(model, graph, train_data)
        
        for i in range(len(params)):
            params_eps = params.copy()
            params_eps[i] += epsilon
            unpack_params(model, params_eps)
            loss_eps = compute_loss(model, graph, train_data)
            grad[i] = (loss_eps - base_loss) / epsilon
            
        # Update
        params -= lr * grad
        unpack_params(model, params)
        
        val_loss = compute_loss(model, graph, val_data)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params.copy()
            
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Train Loss: {base_loss:7.2f} | Val Loss: {val_loss:7.2f}")
            
    print("\n--- Training Complete ---")
    unpack_params(model, best_params)
    test_loss = compute_loss(model, graph, test_data)
    print(f"Final Test Loss: {test_loss:7.2f}")
    
    # Save checkpoint
    checkpoint_path = os.path.join(os.path.dirname(__file__), "gnn_checkpoint.json")
    model.save(checkpoint_path)
    print(f"Saved model checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    train_model()
