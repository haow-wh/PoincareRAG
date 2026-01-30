import json
import random
import tiktoken
import time
import requests
from collections import defaultdict
import numpy as np
import logging
import argparse
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMClient:
    """通用大模型API客户端"""
    def __init__(self, api_url, api_key, model_name):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        logger.info(f"初始化LLM客户端: {api_url}, 模型: {model_name}")
    
    def chat_completion(self, messages, max_tokens=2000, temperature=0.7):
        """通用聊天补全API调用"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API请求失败: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"API请求异常: {str(e)}")
            return None

def split_text(text, max_tokens=2000):
    """
    将长文本分割成较小的段落，确保不超过最大token限制
    
    参数:
        text: 要分割的文本
        max_tokens: 每个段落的最大token数
        
    返回:
        分割后的段落列表
    """
    try:
        # 使用tiktoken计算token
        encoder = tiktoken.encoding_for_model("gpt-4")
        tokens = encoder.encode(text)
        
        segments = []
        for i in range(0, len(tokens), max_tokens):
            segment_tokens = tokens[i:i+max_tokens]
            segments.append(encoder.decode(segment_tokens))
        
        logger.info(f"文本分割为 {len(segments)} 个段落")
        return segments
    except Exception as e:
        logger.error(f"文本分割失败: {str(e)}")
        return [text]

def generate_triplets_from_segment(llm_client, segment, num_triplets):
    """
    从单个文本段落生成三元组
    
    参数:
        llm_client: LLM客户端实例
        segment: 文本段落
        num_triplets: 需要生成的三元组数量
        
    返回:
        三元组列表 [(child, parent, negative), ...]
    """
    triplets = []
    
    # 英文提示词
    prompt = f"""
    Based on the following text content, generate {num_triplets} semantic triplets (child concept, parent concept, unrelated concept):
    
    Text content:
    {segment}
    
    Requirements:
    1. The child concept must be a hyponym/subclass of the parent concept
    2. The unrelated concept must be completely irrelevant to both child and parent
    3. Each triplet should represent a clear hierarchical relationship
    4. Output format must be JSON list: [["child1", "parent1", "unrelated1"], ...]
    5. All concepts must be in English and concise noun phrases
    
    Examples:
    [["Siberian Husky", "Dog Breeds", "Circuit Board"], ["Roses", "Flowering Plants", "Semiconductor"]]
    """
    
    try:
        logger.info(f"向大模型发送请求，生成 {num_triplets} 个三元组...")
        
        # 创建消息
        messages = [{"role": "user", "content": prompt}]
        
        # 调用API
        response = llm_client.chat_completion(messages)
        
        if not response:
            logger.error("API调用失败，未获取到响应")
            return []
        
        # 解析响应
        result = response["choices"][0]["message"]["content"].strip()
        
        # 处理不同的响应格式
        if result.startswith("```json"):
            result = result[7:-3].strip()  # 去除代码块标记
        elif result.startswith("["):
            pass  # 已经是JSON格式
        else:
            # 尝试提取JSON部分
            start_idx = result.find("[")
            end_idx = result.rfind("]")
            if start_idx != -1 and end_idx != -1:
                result = result[start_idx:end_idx+1]
        
        batch_triplets = json.loads(result)
        triplets.extend(batch_triplets)
        logger.info(f"成功生成 {len(batch_triplets)} 个三元组")
        
    except json.JSONDecodeError:
        logger.error("无法解析响应为JSON")
        logger.debug(f"原始响应: {result}")
    except KeyError:
        logger.error("API响应格式异常")
        logger.debug(f"完整响应: {response}")
    except Exception as e:
        logger.error(f"生成三元组时出错: {str(e)}")
    
    return triplets

def generate_triplets(json_file, api_url, api_key, model_name, total_triplets=5000, max_per_segment=10):
    """
    从JSON文件中的多个文档生成三元组
    
    参数:
        json_file: 包含文档列表的JSON文件路径
        api_url: 大模型API地址
        api_key: API密钥
        model_name: 模型名称
        total_triplets: 需要生成的总三元组数量
        max_per_segment: 每个文本段落最多生成的三元组数量
        
    返回:
        三元组列表 [(child, parent, negative), ...]
    """
    # 创建LLM客户端
    llm_client = LLMClient(api_url, api_key, model_name)
    
    try:
        # 读取JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        logger.info(f"成功加载 {len(documents)} 个文档")
    except Exception as e:
        logger.error(f"加载文档失败: {str(e)}")
        return []
    
    all_segments = []
    segment_doc_map = []  # 记录每个段落的来源文档索引
    
    # 第一步: 分割所有文档为段落
    for doc_idx, doc in enumerate(documents):
        segments = split_text(doc)
        all_segments.extend(segments)
        segment_doc_map.extend([doc_idx] * len(segments))
    
    # 创建段落的索引列表并完全随机化
    segment_indices = list(range(len(all_segments)))
    np.random.shuffle(segment_indices)
    
    triplets = []
    concept_counter = defaultdict(int)
    processed_segments = set()
    triplet_count = 0
    start_time = time.time()
    
    # 第二步: 随机处理段落
    for i, seg_idx in enumerate(segment_indices):
        if triplet_count >= total_triplets:
            break
            
        # 跳过已处理的段落
        if seg_idx in processed_segments:
            continue
            
        segment = all_segments[seg_idx]
        doc_idx = segment_doc_map[seg_idx]
        processed_segments.add(seg_idx)
        
        # 计算本段需要生成的三元组数量
        needed = min(max_per_segment, total_triplets - triplet_count)
        
        # 生成三元组
        segment_triplets = generate_triplets_from_segment(llm_client, segment, needed)
        
        # 过滤重复概念
        valid_triplets = []
        for triplet in segment_triplets:
            try:
                child, parent, negative = triplet
                
                # 简单清洗
                child = child.strip()
                parent = parent.strip()
                negative = negative.strip()
                
                # 跳过无效三元组
                if not child or not parent or not negative:
                    continue
                if child == parent or child == negative or parent == negative:
                    continue
                    
                # 检查概念是否过于频繁出现
                if concept_counter[child] > 5 or concept_counter[parent] > 5 or concept_counter[negative] > 5:
                    continue
                    
                # 更新计数器
                concept_counter[child] += 1
                concept_counter[parent] += 1
                concept_counter[negative] += 1
                
                valid_triplets.append((child, parent, negative))
            except Exception as e:
                logger.error(f"处理三元组时出错: {str(e)} | 三元组: {triplet}")
        
        # 添加到结果集
        triplets.extend(valid_triplets)
        triplet_count += len(valid_triplets)
        
        elapsed = time.time() - start_time
        segments_per_hour = (i + 1) / elapsed * 3600 if elapsed > 0 else 0
        logger.info(
            f"进度: {len(processed_segments)}/{len(all_segments)} 段落 | "
            f"三元组: {triplet_count}/{total_triplets} | "
            f"速度: {segments_per_hour:.1f} 段落/小时"
        )
        
        # 避免API速率限制
        sleep_time = 1 + random.random()
        time.sleep(sleep_time)
    
    # 如果随机处理后仍未达到目标数量，尝试补充
    if triplet_count < total_triplets:
        logger.warning(f"随机处理后只生成 {triplet_count} 个三元组，尝试补充...")
        
        # 从未处理的段落中补充
        remaining_segments = [idx for idx in segment_indices if idx not in processed_segments]
        np.random.shuffle(remaining_segments)
        
        for seg_idx in remaining_segments:
            if triplet_count >= total_triplets:
                break
                
            segment = all_segments[seg_idx]
            doc_idx = segment_doc_map[seg_idx]
            processed_segments.add(seg_idx)
            
            needed = min(max_per_segment, total_triplets - triplet_count)
            segment_triplets = generate_triplets_from_segment(llm_client, segment, needed)
            
            # 过滤和添加
            for triplet in segment_triplets:
                try:
                    child, parent, negative = triplet
                    child = child.strip()
                    parent = parent.strip()
                    negative = negative.strip()
                    
                    if not child or not parent or not negative:
                        continue
                    if child == parent or child == negative or parent == negative:
                        continue
                        
                    if concept_counter[child] > 5 or concept_counter[parent] > 5 or concept_counter[negative] > 5:
                        continue
                        
                    concept_counter[child] += 1
                    concept_counter[parent] += 1
                    concept_counter[negative] += 1
                    
                    triplets.append((child, parent, negative))
                    triplet_count += 1
                except Exception as e:
                    logger.error(f"补充处理三元组时出错: {str(e)}")
            
            logger.info(f"补充处理: 文档 {doc_idx+1} | 三元组: {triplet_count}/{total_triplets}")
            time.sleep(1 + random.random())
    
    # 最终分析
    concepts = set()
    for child, parent, negative in triplets:
        concepts.add(child)
        concepts.add(parent)
        concepts.add(negative)
    
    logger.info(f"成功生成 {len(triplets)} 个三元组，涉及 {len(concepts)} 个独特概念")
    logger.info(f"总耗时: {time.time() - start_time:.1f} 秒")
    
    return triplets[:total_triplets]

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='使用大模型API生成三元组')
    parser.add_argument('--input', type=str, default='mix_unique_contexts.json', help='输入JSON文件路径')
    parser.add_argument('--output', type=str, default='triplets.json', help='输出JSON文件路径')
    
    args = parser.parse_args()

    num = 5000
    max_per_segment = 8
    api_url = "https://api.v3.cm/v1/chat/completions"
    api_key = "sk-hYjGPKrARvs6YmGSB0438804DaCd46D2Ac5e4812A38dFdE3"
    model_name = "gpt-4o-mini"
    
    # 设置随机种子
    np.random.seed(42)
    random.seed(42)
    
    # 生成三元组
    triplets = generate_triplets(
        json_file=args.input,
        api_url=api_url,
        api_key=api_key,
        model_name=model_name,
        total_triplets=num,
        max_per_segment=max_per_segment
    )
    
    # 保存生成的三元组
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(triplets, f, ensure_ascii=False, indent=2)
        logger.info(f"三元组已保存至 {args.output}")
        
        # 打印示例
        logger.info("三元组示例:")
        for i in range(min(5, len(triplets))):
            logger.info(f"{triplets[i]}")
    except Exception as e:
        logger.error(f"保存结果失败: {str(e)}")