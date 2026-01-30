import numpy as np
import yaml
import os
from hirag import PoincareRAG, QueryParam
from openai import AsyncOpenAI, OpenAI
from dataclasses import dataclass
from hirag.base import BaseKVStorage
from hirag._utils import compute_args_hash
import asyncio
import json
import httpx
from hirag._utils import logger
from datetime import datetime
from sklearn.cluster import KMeans
import hirag._cluster_utils as cu

# Windows下解决asyncio Loop关闭报错问题
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())



# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract configurations
OPENAI_EMBEDDING_MODEL = config['openai']['embedding_model']
OPENAI_MODEL = config['openai']['model']
OPENAI_API_KEY = config['openai']['api_key']
OPENAI_URL = config['openai']['base_url']
GLM_API_KEY = config['glm']['api_key']
GLM_MODEL = config['glm']['model']
GLM_URL = config['glm']['base_url']

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

# 在异步函数所在的文件中
from MLP.train_hyperbolic import HyperbolicMapperWrapper, OUTPUT_DIM  # 导入包装器类与输出维度

# 全局初始化一次（避免重复加载模型）

@wrap_embedding_func_with_attrs(embedding_dim=OUTPUT_DIM, max_token_size=config['model_params']['max_token_size'])  # 使用双曲嵌入维度
async def OPENAI_embedding(texts: list[str]) -> np.ndarray:
    openai_async_client = AsyncOpenAI(base_url=OPENAI_URL, api_key=OPENAI_API_KEY)
    # logger.warning("embedding get")
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
            logger.error(f"An unexpected non-OpenAI Embedding error occurred (Attempt : {e}")
            raise
    
    if not hasattr(response, 'data') or not isinstance(response.data, list):
        logger.error("response.data is not a list")
        return np.array([])
    
    # 获取欧几里得嵌入
    euclidean_embeddings = np.array([dp.embedding for dp in response.data])
    # logger.warning("欧几里得embedding正常")
    
    # -HiE 消融实验结束：恢复双曲转换
    return await cu.to_hyperbolic_embedding(euclidean_embeddings)




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
    for m in history_messages:
        if m.get("content"):
            messages.append(m)
        else:
            print("None in history_messages", flush=True)
            print(history_messages)

    if prompt:
        messages.append({"role": "user", "content": prompt})
    else:
        print("prompt is None", flush=True)

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
            actual_sleep_time = 60
            await asyncio.sleep(actual_sleep_time)
            try:
                formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.error("访问时间:", formatted_time)
                response = await openai_async_client.chat.completions.create(
                    model=OPENAI_MODEL, 
                    messages=messages,
                    timeout=httpx.Timeout(300.0),  # ⭐️ 设置超时时间（单位：秒） 
                    **kwargs
                )
                formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.error("访问结束时间:", formatted_time)
            except Exception as e:
                formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.error("访问超时时间:", formatted_time)
                logger.error(f"An unexpected non-OpenAI error occurred (Attempt : {e}")
                logger.error(messages)
                raise # 对于非OpenAI的未知错误，可能不应重试，直接抛出


    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": OPENAI_MODEL}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content

def example():
    graph_func = PoincareRAG(working_dir=config['hirag']['working_dir'],
                        enable_llm_cache=config['hirag']['enable_llm_cache'],
                        embedding_func=OPENAI_embedding,
                        best_model_func=OPENAI_model_if_cache,
                        cheap_model_func=OPENAI_model_if_cache,
                        enable_hierachical_mode=config['hirag']['enable_hierachical_mode'], 
                        embedding_batch_num=config['hirag']['embedding_batch_num'],
                        embedding_func_max_async=config['hirag']['embedding_func_max_async'],
                        enable_naive_rag=config['hirag']['enable_naive_rag'])

    # comment this if the working directory has already been indexed
    # with open("./working_dir/追风筝的人.txt",encoding='gbk') as f:
    #     graph_func.insert(f.read())

    with open("./working_dir2/book.txt") as f:
        graph_func.insert(f.read())

    print("Perform hi search:")
    print(graph_func.query("What are the top themes in this story?", param=QueryParam(mode="hi")))

# insert_context_openai
def sample_insert():
    cu.init_hyperbolic_mapper("MLP/hyperbolic_mapper.pth") # -HiE experiment: enabled
    # DATASET = "mix"
    # DATASET = "legal"
    DATASET = "cs"
    # DATASET = "agriculture"
    import logging
    # 配置日志
    logging.basicConfig(
        filename='{DATASET}.log',  # 设置日志文件名
        filemode='a'  # 设置文件模式，'a' 表示追加模式
    )
    file_path = f"./eval/datasets/{DATASET}/{DATASET}_unique_contexts.json"
    work_dir = f"./eval/datasets/{DATASET}/work_dir_Hicluster_5_hiem-HiE"
    try:
        for fname in ["vdb_chunks.json", "vdb_entities.json"]:
            fpath = os.path.join(work_dir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
    except Exception as e:
        logger.error(f"清理旧向量库失败: {e}")
    graph_func = PoincareRAG(
        working_dir=work_dir, 
        enable_llm_cache=config['hirag']['enable_llm_cache'],
        embedding_func=OPENAI_embedding,
        best_model_func=OPENAI_model_if_cache,
        cheap_model_func=OPENAI_model_if_cache,
        # to_hyperbolic_embedding=to_hyperbolic_embedding,
        embedding_batch_num=config['hirag']['embedding_batch_num'],
        enable_hierachical_mode=True, 
        embedding_func_max_async=4,
        enable_naive_rag=True)

    with open(file_path, mode="r", encoding="utf-8") as f:
        unique_contexts = json.load(f)
        graph_func.insert(unique_contexts)

sample_insert()
