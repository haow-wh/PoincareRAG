import os
import json
import argparse
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import os
import logging
import numpy as np
import yaml
from hirag import PoincareRAG, QueryParam
from openai import AsyncOpenAI, OpenAI
from dataclasses import dataclass
from hirag.base import BaseKVStorage
from hirag._utils import compute_args_hash
import asyncio
import random
import json
os.environ["OPENAI_API_KEY"] = "***"

from MLP.train_hyperbolic import HyperbolicMapperWrapper, OUTPUT_DIM

# 全局初始化一次（避免重复加载模型）
HYPERBOLIC_MAPPER = None

def init_hyperbolic_mapper(model_path):
    """初始化双曲映射器"""
    global HYPERBOLIC_MAPPER
    if HYPERBOLIC_MAPPER is None:
        HYPERBOLIC_MAPPER = HyperbolicMapperWrapper(model_path)
    return HYPERBOLIC_MAPPER


# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract configurations
OPENAI_EMBEDDING_MODEL = config['openai']['embedding_model']
# OPENAI_MODEL = config['openai']['model']
OPENAI_MODEL = "deepseek-r1"
OPENAI_API_KEY = config['openai']['api_key']
OPENAI_URL = config['openai']['base_url']
# WORKING_DIR = config['hirag']['working_dir']
TOTAL_TOKEN_COST = 0
TOTAL_API_CALL_COST = 0

@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)

def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro

@wrap_embedding_func_with_attrs(embedding_dim=config['model_params']['openai_embedding_dim'], max_token_size=config['model_params']['max_token_size'])
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
            print(f"An unexpected non-OpenAI Embedding error occurred (Attempt : {e}")
            raise # 对于非OpenAI的未知错误，可能不应重试，直接抛出
    
    # 获取欧几里得嵌入
    euclidean_embeddings = np.array([dp.embedding for dp in response.data])
    
    # -HiE 消融实验：直接返回欧几里得嵌入
    return euclidean_embeddings

async def OPENAI_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=OPENAI_API_KEY, base_url=OPENAI_URL
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(OPENAI_MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    try:
        response = await openai_async_client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages, **kwargs
        )
    except Exception as e:
        actual_sleep_time = 60
        await asyncio.sleep(actual_sleep_time)
        try:
            response = await openai_async_client.chat.completions.create(
                model=OPENAI_MODEL, messages=messages, **kwargs
            )
        except Exception as e:
            print(f"An unexpected non-OpenAI error occurred (Attempt : {e}")
            raise # 对于非OpenAI的未知错误，可能不应重试，直接抛出


    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": OPENAI_MODEL}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="mix")
    parser.add_argument("-m", "--mode", type=str, default="hi", help="hi / naive / hi_global / hi_local / hi_bridge / hi_nobridge")
    args = parser.parse_args()
    
    if args.mode == "naive":
        mode = True
    elif args.mode == "global" or "local":
        mode = False

    MAX_QUERIES = 100

    print(f"mode is {args.mode}")
    DATASET = args.dataset
    input_path = f"./eval/datasets/{DATASET}/{DATASET}_query.jsonl"
    output_path = f"./eval/datasets/{DATASET}/{DATASET}_Hicl_hiem-HiE_ds.jsonl"
    graph_func = PoincareRAG(
        working_dir = f"eval/datasets/{DATASET}/work_dir_Hicluster_5_hiem-HiE", 
        enable_llm_cache=False,
        embedding_func=OPENAI_embedding,
        best_model_func=OPENAI_model_if_cache,
        cheap_model_func=OPENAI_model_if_cache,
        enable_hierachical_mode=True, 
        embedding_func_max_async=8,
        enable_naive_rag=True)

    query_list = []
    with open(input_path, encoding="utf-8", mode="r") as f:      # get context
        lines = f.readlines()
        for item in lines:
            item_dict = json.loads(item)
            if DATASET == "mix":
                query_list.append(item_dict["input"])
            else:
                query_list.append(item_dict["query"])
    query_list = query_list[:MAX_QUERIES]
    answer_list = []

    print(f"Perform {args.mode} search:")
    for query in tqdm(query_list):
        tqdm.write(f"Q: {query}")
        answer = graph_func.query(query=query, param=QueryParam(mode=args.mode))
        tqdm.write(f"A: {answer} \n ################################################################################################")
        answer_list.append(answer)
    
    result_to_write = []
    for query, answer in zip(query_list, answer_list):
        result_to_write.append({"query": query, "answer": answer})
    with open(output_path, "w") as f:
        for item in result_to_write:
            f.write(json.dumps(item) + "\n")
        
