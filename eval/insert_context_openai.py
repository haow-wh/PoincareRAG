import os
import sys
import json
import time
sys.path.append("../")
from hirag import PoincareRAG, QueryParam

os.environ["OPENAI_API_KEY"] = "sk-778tOs0CHyHTDRzC5876B2751b9c4332Ba21C70f0296862f"

DATASET = "legal"
file_path = f"./eval/datasets/{DATASET}/{DATASET}_unique_contexts.json"

graph_func = PoincareRAG(
    working_dir=f"./eval/datasets/{DATASET}/work_dir_hi_Hicluster", 
    enable_hierachical_mode=True, 
    embedding_func_max_async=4,
    enable_naive_rag=True)

with open(file_path, mode="r") as f:
    unique_contexts = json.load(f)
    graph_func.insert(unique_contexts)
