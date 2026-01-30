import json
import requests
import time
import os
import logging
import argparse
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingClient:
    """通用嵌入API客户端"""
    def __init__(self, api_url, api_key, model_name):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本嵌入"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "input": texts
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                # Ensure correct order based on index
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in sorted_data]
            else:
                logger.error(f"Embedding API failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Embedding API exception: {str(e)}")
            return None

def restore_vectors(input_path, output_path, api_url, api_key, model_name):
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Loading entities from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Handle NanoVectorDB format
    if isinstance(raw_data, dict) and 'data' in raw_data:
        items = raw_data['data']
        meta = {k: v for k, v in raw_data.items() if k != 'data'}
    elif isinstance(raw_data, list):
        items = raw_data
        meta = {"embedding_dim": 1536} # Default guess
    else:
        logger.error("Unknown file format")
        return

    # Identify items needing vectors
    to_process = []
    for item in items:
        if '__vector__' not in item and 'embedding' not in item:
            # We need text to embed. Usually entity_name or content
            text = item.get('entity_name') or item.get('content') or item.get('__id__')
            if text:
                to_process.append((item, text))
    
    if not to_process:
        logger.info("All items already have vectors. Nothing to do.")
        return

    logger.info(f"Found {len(to_process)} items missing vectors. Starting restoration...")
    
    client = EmbeddingClient(api_url, api_key, model_name)
    
    batch_size = 20
    total = len(to_process)
    
    for i in range(0, total, batch_size):
        batch = to_process[i:i+batch_size]
        batch_items = [b[0] for b in batch]
        batch_texts = [b[1] for b in batch]
        
        embeddings = client.get_embeddings(batch_texts)
        
        if embeddings:
            for item, emb in zip(batch_items, embeddings):
                item['__vector__'] = emb
            logger.info(f"Processed {min(i+batch_size, total)}/{total}")
        else:
            logger.error("Failed to get embeddings for batch. Stopping.")
            break
            
        time.sleep(0.5) # Rate limit protection

    # Save result
    output_data = meta.copy()
    output_data['data'] = items
    
    logger.info(f"Saving restored data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Restore missing vectors in vdb_entities.json')
    parser.add_argument('--input', type=str, default=r'e:\HyperRAG\PoincaréRAG\working_dir\vdb_entities.json')
    parser.add_argument('--output', type=str, default=r'e:\HyperRAG\PoincaréRAG\working_dir\vdb_entities_restored.json')
    
    # Defaults from train_hyperbolic.py
    # Try to load from config.yaml
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
    
    default_api_key = "sk-hYjGPKrARvs6YmGSB0438804DaCd46D2Ac5e4812A38dFdE3"
    default_base_url = "https://api.v3.cm/v1/embeddings"
    default_model = "text-embedding-ada-002"
    
    if os.path.exists(config_path):
        logger.info(f"Loading config from {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Simple parsing to avoid dependency on PyYAML
            import re
            
            # Find openai section
            openai_match = re.search(r'openai:(.*?)(?:^\w|\Z)', content, re.DOTALL | re.MULTILINE)
            if openai_match:
                openai_section = openai_match.group(1)
                
                key_match = re.search(r'api_key:\s*["\']?([^"\s\n]+)["\']?', openai_section)
                if key_match:
                    default_api_key = key_match.group(1)
                    
                url_match = re.search(r'base_url:\s*["\']?([^"\s\n]+)["\']?', openai_section)
                if url_match:
                    base = url_match.group(1).rstrip('/')
                    if not base.endswith('/embeddings'):
                        default_base_url = f"{base}/embeddings"
                    else:
                        default_base_url = base
                        
                model_match = re.search(r'embedding_model:\s*["\']?([^"\s\n]+)["\']?', openai_section)
                if model_match:
                    default_model = model_match.group(1)
                    
            logger.info(f"Loaded config: API URL={default_base_url}, Model={default_model}")
            
        except Exception as e:
            logger.error(f"Failed to parse config.yaml: {e}")
    
    parser.add_argument('--api_url', type=str, default=default_base_url)
    parser.add_argument('--api_key', type=str, default=default_api_key)
    parser.add_argument('--model', type=str, default=default_model)
    
    args = parser.parse_args()
    
    restore_vectors(args.input, args.output, args.api_url, args.api_key, args.model)
