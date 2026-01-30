import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging
import argparse
import os
import requests

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置参数
INPUT_DIM = 1536  # OpenAI嵌入维度
HIDDEN_DIM = 512
OUTPUT_DIM = 128  # 双曲嵌入维度
ALPHA = 0.2  # 聚类损失边界
BETA = 0.1   # 向心损失边界

class EmbeddingClient:
    """通用嵌入API客户端"""
    def __init__(self, api_url, api_key, model_name):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        logger.info(f"初始化嵌入客户端: {api_url}, 模型: {model_name}")
    
    def get_embeddings(self, texts):
        """获取文本嵌入向量"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
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
                return [item["embedding"] for item in data["data"]]
            else:
                logger.error(f"嵌入API请求失败: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"嵌入API请求异常: {str(e)}")
            return None

class TripletDataset(Dataset):
    """加载三元组数据集"""
    def __init__(self, triplets_file, api_url, api_key, model_name):
        # 创建嵌入客户端
        self.embedding_client = EmbeddingClient(api_url, api_key, model_name)
        
        try:
            with open(triplets_file, 'r', encoding='utf-8') as f:
                self.triplets = json.load(f)
            
            logger.info(f"成功加载 {len(self.triplets)} 个三元组")
        except Exception as e:
            logger.error(f"加载三元组失败: {str(e)}")
            self.triplets = []
        
        # 获取所有唯一概念
        self.concepts = set()
        for child, parent, negative in self.triplets:
            self.concepts.update([child, parent, negative])
        
        self.cache_file = "embedding_cache_data.json"
        
        # 获取文本嵌入
        self.embeddings = self.get_text_embeddings()
    
    def get_text_embeddings(self):
        """批量获取文本嵌入"""

        # 检查缓存是否存在且有效
        if os.path.exists(self.cache_file):
            logger.info(f"发现嵌入缓存文件: {self.cache_file}")
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                # 验证缓存是否包含所有概念
                cached_concepts = set(cache_data.keys())
                if self.concepts.issubset(cached_concepts):
                    logger.info(f"缓存有效，包含所有 {len(self.concepts)} 个概念")
                    return cache_data
                else:
                    missing = self.concepts - cached_concepts
                    logger.warning(f"缓存缺失 {len(missing)} 个概念，将更新缓存")
            except Exception as e:
                logger.error(f"加载缓存失败: {str(e)}")

        embeddings = {}
        batch_size = 32  # 批量限制
        
        concepts = list(self.concepts)
        logger.info(f"开始获取 {len(concepts)} 个概念的嵌入...")
        
        for i in range(0, len(concepts), batch_size):
            batch = concepts[i:i+batch_size]
            
            try:
                batch_embeddings = self.embedding_client.get_embeddings(batch)
                
                if batch_embeddings and len(batch_embeddings) == len(batch):
                    for j, text in enumerate(batch):
                        embeddings[text] = batch_embeddings[j]
                    
                    logger.info(f"已获取 {min(i+batch_size, len(concepts))}/{len(concepts)} 个嵌入")
                else:
                    logger.error(f"获取嵌入失败，跳过本批次: {batch}")
            except Exception as e:
                logger.error(f"获取嵌入失败: {str(e)}")
        
        # 保存到缓存
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=2)
            logger.info(f"嵌入已保存到缓存: {self.cache_file}")
        except Exception as e:
            logger.error(f"保存嵌入缓存失败: {str(e)}")

        return embeddings
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        child, parent, negative = self.triplets[idx]
        return (
            np.array(self.embeddings[child]),
            np.array(self.embeddings[parent]),
            np.array(self.embeddings[negative])
        )

class HyperbolicMapper(nn.Module):
    """欧几里得到双曲空间的映射模型"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 初始化权重
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        # 映射到双曲空间
        h = self.net(x)
        
        # 投影到Poincaré球 (||h|| < 1)
        norm = torch.norm(h, p=2, dim=1, keepdim=True)
        scale = torch.minimum(
            torch.tensor(1.0, device=h.device) - 1e-5,
            1.0 / (norm + 1e-5)
        )
        return scale * h

def poincare_distance(u, v, eps=1e-5):
    """计算Poincaré球中的双曲距离"""
    # 欧几里得范数平方
    u_norm_sq = torch.sum(u**2, dim=1, keepdim=True)
    v_norm_sq = torch.sum(v**2, dim=1, keepdim=True)
    
    # 差异范数平方
    diff = u - v
    diff_norm_sq = torch.sum(diff**2, dim=1, keepdim=True)
    
    # Poincaré距离公式
    denominator = (1 - u_norm_sq) * (1 - v_norm_sq)
    inside = 1 + 2 * diff_norm_sq / (denominator + eps)
    
    # 数值稳定性处理
    inside = torch.clamp(inside, min=1.0 + eps)
    return torch.arccosh(inside)

def hyperbolic_norm(x):
    """计算双曲范数 (到原点的距离)"""
    x_norm_sq = torch.sum(x**2, dim=1)
    # 添加小量防止除零
    return torch.arccosh(1 + 2 * x_norm_sq / (1 - x_norm_sq + 1e-5))

def hit_loss(e, e_plus, e_minus, alpha, beta):
    """计算HIT损失 (聚类损失 + 向心损失)"""
    # 聚类损失
    dist_pos = poincare_distance(e, e_plus)
    dist_neg = poincare_distance(e, e_minus)
    cluster_loss = torch.mean(torch.clamp(dist_pos - dist_neg + alpha, min=0))
    
    # 向心损失 (仅使用正样本)
    norm_e = hyperbolic_norm(e)
    norm_e_plus = hyperbolic_norm(e_plus)
    centri_loss = torch.mean(torch.clamp(norm_e_plus - norm_e + beta, min=0))
    
    return cluster_loss + centri_loss, cluster_loss, centri_loss

def train_model(args):
    """主训练函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载数据集
    dataset = TripletDataset(
        args.triplets_file, 
        args.embed_api_url, 
        args.api_key, 
        args.embed_model_name
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=min(4, os.cpu_count())
    )
    
    # 初始化模型和优化器
    model = HyperbolicMapper(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(args.epochs):
        total_loss = 0.0
        cluster_loss_sum = 0.0
        centri_loss_sum = 0.0
        
        model.train()
        for batch_idx, (child_embs, parent_embs, neg_embs) in enumerate(dataloader):
            # 转移到设备
            child_embs = child_embs.float().to(device)
            parent_embs = parent_embs.float().to(device)
            neg_embs = neg_embs.float().to(device)
            
            # 前向传播
            e = model(child_embs)
            e_plus = model(parent_embs)
            e_minus = model(neg_embs)
            
            # 计算损失
            loss, cluster_loss, centri_loss = hit_loss(
                e, e_plus, e_minus, ALPHA, BETA)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # 记录统计信息
            total_loss += loss.item()
            cluster_loss_sum += cluster_loss.item()
            centri_loss_sum += centri_loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f} (Cluster: {cluster_loss.item():.4f}, "
                    f"Centri: {centri_loss.item():.4f})"
                )
        
        # 每轮统计
        avg_loss = total_loss / len(dataloader)
        avg_cluster = cluster_loss_sum / len(dataloader)
        avg_centri = centri_loss_sum / len(dataloader)
        
        logger.info(f"\nEpoch {epoch+1} Summary: "
              f"Avg Loss: {avg_loss:.10f} | "
              f"Cluster Loss: {avg_cluster:.4f} | "
              f"Centri Loss: {avg_centri:.4f}\n")
        
        # 学习率调整
        scheduler.step(avg_loss)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.model_path)
            logger.info(f"保存最佳模型到 {args.model_path} (loss: {best_loss:.4f})")
    
    logger.info("训练完成!")

def save_hyperbolic_embeddings(args, output_file="hyperbolic_embeddings.json"):
    """保存双曲化后的嵌入向量到文件"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载数据集（只用于获取概念和嵌入）
    dataset = TripletDataset(
        args.triplets_file, 
        args.embed_api_url, 
        args.api_key, 
        args.embed_model_name
    )
    
    # 初始化模型并加载训练好的权重
    model = HyperbolicMapper(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    logger.info(f"加载模型权重: {args.model_path}")
    
    # 收集所有概念和对应的欧几里得嵌入
    concepts = list(dataset.embeddings.keys())
    euclidean_embeddings = [dataset.embeddings[concept] for concept in concepts]
    
    # 批量处理嵌入
    batch_size = 128
    hyperbolic_embeddings = {}
    logger.info(f"开始处理 {len(concepts)} 个概念的双曲嵌入...")
    
    with torch.no_grad():
        for i in range(0, len(concepts), batch_size):
            batch_concepts = concepts[i:i+batch_size]
            batch_embs = euclidean_embeddings[i:i+batch_size]
            
            # 转换为tensor并转移到设备
            tensor_embs = torch.tensor(batch_embs, dtype=torch.float32).to(device)
            
            # 双曲化
            hyperbolic_batch = model(tensor_embs)
            
            # 转换为numpy数组并保存
            hyperbolic_np = hyperbolic_batch.cpu().numpy()
            
            for j, concept in enumerate(batch_concepts):
                # 转换为Python列表格式
                hyperbolic_embeddings[concept] = hyperbolic_np[j].tolist()
            
            logger.info(f"已处理 {min(i+batch_size, len(concepts))}/{len(concepts)} 个概念")
    
    # 保存到文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(hyperbolic_embeddings, f, ensure_ascii=False, indent=2)
        logger.info(f"成功保存 {len(hyperbolic_embeddings)} 个双曲嵌入到 {output_file}")
    except Exception as e:
        logger.error(f"保存双曲嵌入失败: {str(e)}")
    
    return hyperbolic_embeddings


class HyperbolicMapperWrapper:
    """双曲映射模型包装器，用于方便调用"""
    def __init__(self, model_path, device=None):
        """
        初始化包装器
        :param model_path: 模型文件路径
        :param device: 指定设备 (None表示自动选择)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.model = HyperbolicMapper(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        logger.info(f"加载双曲映射模型: {model_path}, 设备: {self.device}")
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        将欧几里得嵌入转换为双曲嵌入
        :param embeddings: 欧几里得嵌入向量 (单个向量或批量向量)
        :return: 双曲嵌入向量
        """
        # 确保输入是二维数组
        if len(embeddings.shape) == 1:
            embeddings = embeddings[np.newaxis, :]
        
        with torch.no_grad():
            tensor_embs = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
            hyperbolic_embs = self.model(tensor_embs).cpu().numpy()
        
        return hyperbolic_embs.squeeze()  # 移除多余的维度

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练双曲映射模型')
    parser.add_argument('--triplets_file', type=str, default='triplets.json', help='三元组数据文件')
    parser.add_argument('--api_key', type=str, default="sk-hYjGPKrARvs6YmGSB0438804DaCd46D2Ac5e4812A38dFdE3", help='API密钥')
    parser.add_argument('--embed_api_url', type=str, default="https://api.v3.cm/v1/embeddings", help='嵌入API地址')
    parser.add_argument('--embed_model_name', type=str, default="text-embedding-ada-002", help='嵌入模型名称')
    parser.add_argument('--model_path', type=str, default='hyperbolic_mapper.pth', help='模型保存路径')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--cpu', action='store_true', help='强制使用CPU')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    train_model(args)
    
    args = parser.parse_args()
    
    save_hyperbolic_embeddings(args, "my_hyperbolic_embeddings.json")