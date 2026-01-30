import numpy as np
import yaml
import os
from hirag import PoincareRAG, QueryParam
from openai import AsyncOpenAI
from dataclasses import dataclass
from hirag.base import BaseKVStorage
from hirag._utils import compute_args_hash
import asyncio
import json
import httpx
from hirag._utils import logger
from datetime import datetime
import hirag._cluster_utils as cu
from MLP.train_hyperbolic import HyperbolicMapperWrapper, OUTPUT_DIM

# Windows async fix
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load configuration
with open('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract configurations
OPENAI_EMBEDDING_MODEL = config['openai']['embedding_model']
OPENAI_MODEL = config['openai']['model']
OPENAI_API_KEY = config['openai']['api_key']
OPENAI_URL = config['openai']['base_url']

# Global storage for interception
global_euclidean_storage = []
global_hyperbolic_storage = []

@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)

def wrap_embedding_func_with_attrs(**kwargs):
    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func
    return final_decro

@wrap_embedding_func_with_attrs(embedding_dim=OUTPUT_DIM, max_token_size=config['model_params']['max_token_size'])
async def OPENAI_embedding(texts: list[str]) -> np.ndarray:
    openai_async_client = AsyncOpenAI(base_url=OPENAI_URL, api_key=OPENAI_API_KEY)
    
    try:
        response = await openai_async_client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL, input=texts, encoding_format="float"
        )
    except Exception as e:
        actual_sleep_time = 60
        await asyncio.sleep(actual_sleep_time)
        try:
            response = await openai_async_client.embeddings.create(
                model=OPENAI_EMBEDDING_MODEL, input=texts, encoding_format="float"
            )
        except Exception as e:
            logger.error(f"An unexpected non-OpenAI Embedding error occurred: {e}")
            raise
    
    if not hasattr(response, 'data') or not isinstance(response.data, list):
        logger.error("response.data is not a list")
        return np.array([])
    
    # 1. Get Euclidean Embeddings
    euclidean_embeddings = np.array([dp.embedding for dp in response.data])
    
    # 2. Intercept and store Euclidean
    # Note: We append the batch. Later we will vstack.
    global_euclidean_storage.append(euclidean_embeddings)
    
    # 3. Convert to Hyperbolic
    hyperbolic_embeddings = await cu.to_hyperbolic_embedding(euclidean_embeddings)
    
    # FIX: Ensure 2D array to prevent "squeeze" issue when batch size is 1
    if len(hyperbolic_embeddings.shape) == 1:
        hyperbolic_embeddings = hyperbolic_embeddings[np.newaxis, :]
    
    # 4. Intercept and store Hyperbolic
    global_hyperbolic_storage.append(hyperbolic_embeddings)
    
    # 5. Return Hyperbolic for PoincareRAG workflow
    return hyperbolic_embeddings

async def OPENAI_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # Simplified version or copied from original to satisfy PoincareRAG requirements
    openai_async_client = AsyncOpenAI(
        api_key=OPENAI_API_KEY, base_url=OPENAI_URL
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    for m in history_messages:
        if m.get("content"):
            messages.append(m)
    
    if prompt:
        messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        args_hash = compute_args_hash(OPENAI_MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    try:
        response = await openai_async_client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages, **kwargs
        )
    except Exception as e:
        # Simple retry logic as in original
        await asyncio.sleep(60)
        try:
            response = await openai_async_client.chat.completions.create(
                model=OPENAI_MODEL, messages=messages, **kwargs
            )
        except Exception as e:
             # Just fail or return empty if critical, but let's try one more time as in original
             await asyncio.sleep(60)
             response = await openai_async_client.chat.completions.create(
                model=OPENAI_MODEL, messages=messages, timeout=httpx.Timeout(300.0), **kwargs
            )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": OPENAI_MODEL}}
        )
    return response.choices[0].message.content

def run_extraction():
    DATASETS = ["mix", "legal", "cs", "agriculture"]
    
    # Ensure output directory exists
    os.makedirs("../embeddings", exist_ok=True)
    
    for dataset in DATASETS:
        print(f"Processing dataset: {dataset}...")
        
        # Reset globals
        global_euclidean_storage.clear()
        global_hyperbolic_storage.clear()
        
        # Init Hyperbolic Mapper
        cu.init_hyperbolic_mapper("MLP/hyperbolic_mapper.pth")
        
        # Paths
        file_path = f"./eval/datasets/{dataset}/{dataset}_unique_contexts.json"

        work_dir = f"./eval/datasets/{dataset}/work_dir_extraction_tsne"
        
        # Clean up old vector DB files in this work_dir
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        
        try:
            for fname in ["vdb_chunks.json", "vdb_entities.json"]:
                fpath = os.path.join(work_dir, fname)
                if os.path.exists(fpath):
                    os.remove(fpath)
        except Exception as e:
            logger.error(f"Failed to clean old vector DB: {e}")
            
        # Init PoincareRAG
        graph_func = PoincareRAG(
            working_dir=work_dir, 
            enable_llm_cache=config['hirag']['enable_llm_cache'],
            embedding_func=OPENAI_embedding,
            best_model_func=OPENAI_model_if_cache,
            cheap_model_func=OPENAI_model_if_cache,
            embedding_batch_num=config['hirag']['embedding_batch_num'],
            enable_hierachical_mode=True, 
            embedding_func_max_async=4,
            enable_naive_rag=False
        )
        
        # Load data
        if not os.path.exists(file_path):
            print(f"Warning: Data file {file_path} not found. Skipping {dataset}.")
            continue
            
        with open(file_path, mode="r", encoding="utf-8") as f:
            unique_contexts = json.load(f)
            print(f"Processing {len(unique_contexts)} items.")
            graph_func.insert(unique_contexts)
            
        # Save intercepted embeddings
        if global_euclidean_storage:
            all_euclidean = np.vstack(global_euclidean_storage)
            save_path_euc = f"./embeddings/{dataset}_euclidean.npy"
            np.save(save_path_euc, all_euclidean)
            print(f"Saved {all_euclidean.shape[0]} Euclidean embeddings to {save_path_euc}")
        else:
            print(f"No Euclidean embeddings collected for {dataset}")

        if global_hyperbolic_storage:
            all_hyperbolic = np.vstack(global_hyperbolic_storage)
            save_path_hyp = f"./embeddings/{dataset}_hyperbolic.npy"
            np.save(save_path_hyp, all_hyperbolic)
            print(f"Saved {all_hyperbolic.shape[0]} Hyperbolic embeddings to {save_path_hyp}")
        else:
            print(f"No Hyperbolic embeddings collected for {dataset}")
            
        print(f"Finished {dataset}.\n")

if __name__ == "__main__":
    run_extraction()
