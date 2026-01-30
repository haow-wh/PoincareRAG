import json
import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Define HyperbolicMapper class (copy from train_hyperbolic.py)
class HyperbolicMapper(nn.Module):
    """欧几里得到双曲空间的映射模型"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        h = self.net(x)
        norm = torch.norm(h, p=2, dim=1, keepdim=True)
        scale = torch.minimum(
            torch.tensor(1.0, device=h.device) - 1e-5,
            1.0 / (norm + 1e-5)
        )
        return scale * h

def generate_hyperbolic_embeddings(vdb_path, model_path, output_path):
    # Check for restored file first
    restored_path = vdb_path.replace('.json', '_restored.json')
    if os.path.exists(restored_path):
        print(f"Found restored file: {restored_path}. Using it instead.")
        vdb_path = restored_path
    
    print(f"Loading Euclidean vectors from {vdb_path}...")
    
    if not os.path.exists(vdb_path):
        print(f"Error: {vdb_path} not found.")
        return

    with open(vdb_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Extract vectors
    if isinstance(data, dict) and 'data' in data:
        items = data['data']
    elif isinstance(data, list):
        items = data
    else:
        print("Error: Unknown vdb format")
        return

    vectors = []
    keys = []
    
    for item in items:
        # Try different key names
        vec = item.get('__vector__') or item.get('embedding')
        key = item.get('entity_name') or item.get('__id__') or item.get('id')
        
        if vec and key:
            vectors.append(vec)
            keys.append(key)
            
    if not vectors:
        print("No vectors found!")
        print("-" * 50)
        print("ERROR: The input file contains no vector data.")
        print("This usually means the vector database dump is incomplete.")
        print(f"Please run the restoration script to regenerate vectors:")
        print(f"python MLP/restore_vectors.py")
        print("-" * 50)
        return
        
    print(f"Found {len(vectors)} vectors.")
    
    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Constants from train_hyperbolic.py
    INPUT_DIM = 1536
    HIDDEN_DIM = 512
    OUTPUT_DIM = 128
    
    model = HyperbolicMapper(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model {model_path} not found! Using random weights (Just for testing flow).")
        
    model.eval()
    
    # Inference
    batch_size = 128
    hyp_embeddings = {}
    
    with torch.no_grad():
        for i in range(0, len(vectors), batch_size):
            batch_vecs = vectors[i:i+batch_size]
            batch_keys = keys[i:i+batch_size]
            
            tensor_vecs = torch.tensor(batch_vecs, dtype=torch.float32).to(device)
            hyp_out = model(tensor_vecs).cpu().numpy()
            
            for key, hyp_vec in zip(batch_keys, hyp_out):
                hyp_embeddings[key] = hyp_vec.tolist()
                
    # Save
    print(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hyp_embeddings, f, ensure_ascii=False, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    # 使用绝对路径以确保正确
    vdb_path = r"e:\HyperRAG\PoincaréRAG\working_dir\vdb_entities.json"
    model_path = r"e:\HyperRAG\PoincaréRAG\MLP\hyperbolic_mapper.pth"
    output_path = r"e:\HyperRAG\PoincaréRAG\working_dir\hyperbolic_embeddings.json"
    
    generate_hyperbolic_embeddings(vdb_path, model_path, output_path)
