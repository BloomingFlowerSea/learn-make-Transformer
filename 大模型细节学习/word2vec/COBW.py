import torch
import torch.nn as nn
import torch.nn.functional as F




class CBOW_NegativeSampling(nn.Module):
    def __init__(self, vocab_size, embed_size, window_size=2):
        super(CBOW_NegativeSampling, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.window_size = window_size
        
        # 优化点 1：双 Embedding 表架构
        # in_embed 替代原先的 embedding，用于提取上下文特征
        self.in_embed = nn.Embedding(vocab_size, embed_size)
        
        # out_embed 替代了原先低效的 nn.Linear
        # 它的本质是一个权重矩阵，但通过 Gather 操作按需拉取，极大地节省了显存带宽
        self.out_embed = nn.Embedding(vocab_size, embed_size)
        
        # 权重初始化
        nn.init.xavier_uniform_(self.in_embed.weight)
        # 输出层通常初始化为 0 或者很小的值
        nn.init.constant_(self.out_embed.weight, 0)

    def forward(self, context_indices, target_indices, neg_indices):
        """
        前向传播直接计算并返回 Loss
        :param context_indices: [batch_size, 2*window_size] 上下文
        :param target_indices:  [batch_size] 真实的中心词 (正样本)
        :param neg_indices:     [batch_size, K] 采样出的负样本
        :return: scalar loss
        """
        batch_size = context_indices.shape[0]

        # 1. 计算上下文向量 (求平均)
        # [B, 2W, D] -> [B, D]
        context_vecs = self.in_embed(context_indices)
        context_avg = context_vecs.mean(dim=1) 

        # 2. 处理正样本 (真实的中心词)
        # [B, D]
        pos_vecs = self.out_embed(target_indices)
        # 算子优化：按元素相乘后在特征维度求和，等价于逐样本做点积
        # pos_scores: [B]
        pos_scores = torch.sum(context_avg * pos_vecs, dim=1)
        
        # 3. 处理负样本
        # [B, K, D]
        neg_vecs = self.out_embed(neg_indices)
        # 算子优化：使用 bmm (Batch Matrix Multiplication) 批量计算负样本点积
        # context_avg.unsqueeze(2) 形状为 [B, D, 1]
        # bmm([B, K, D], [B, D, 1]) -> [B, K, 1] -> squeeze得到 [B, K]
        neg_scores = torch.bmm(neg_vecs, context_avg.unsqueeze(2)).squeeze(2)

        # 4. 计算 Negative Sampling Loss (使用 F.logsigmoid 保证数值稳定性)
        # 目标：最大化正样本被预测为 1 的概率，最大化负样本被预测为 0 的概率
        # 数学公式：- log(sigmoid(pos_scores)) - sum(log(sigmoid(-neg_scores)))
        pos_loss = -F.logsigmoid(pos_scores)
        neg_loss = -F.logsigmoid(-neg_scores).sum(dim=1)

        # 返回当前 Batch 的平均 Loss
        return (pos_loss + neg_loss).mean()

    def get_embedding(self):
        """
        训练完成后，通常将两张表相加或直接使用 in_embed 作为最终词向量
        """
        return self.in_embed.weight.data
    
    def Dataloader(self, corpus, vocab, batch_size, word_freqs, K=5):
        """
        生成训练数据：上下文 + 正样本(中心词) + 负样本
        :param word_freqs: 词频列表，索引与 vocab 对应
        :param K: 负样本数量
        """
        contexts, targets = [], []
        for i in range(self.window_size, len(corpus) - self.window_size):
            context = corpus[i - self.window_size:i] + corpus[i + 1:i + self.window_size + 1]
            target = corpus[i]
            contexts.append(context)
            targets.append(target)
        
        contexts = torch.tensor(contexts, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)
        
        # --- 负采样核心逻辑 ---
        # 1. 计算 3/4 次方平滑概率
        word_freqs = torch.tensor(word_freqs, dtype=torch.float)
        smoothed_freqs = torch.pow(word_freqs, 0.75)
        sample_probs = smoothed_freqs / smoothed_freqs.sum()
        
        # 2. 批量采样出所有的负样本 [num_samples, K]
        # torch.multinomial 在底层是高度优化的 alias method 或 CDF 二分查找
        num_samples = len(targets)
        negatives = torch.multinomial(sample_probs, num_samples * K, replacement=True)
        negatives = negatives.view(num_samples, K)
        
        dataset = torch.utils.data.TensorDataset(contexts, targets, negatives)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def train_epoch(self, dataloader, optimizer, device):
        self.train()
        total_loss = 0.0
        for context, target, negatives in dataloader:
            context = context.to(device)
            target = target.to(device)
            negatives = negatives.to(device)
            
            # 清空梯度
            optimizer.zero_grad()
            
            # 前向计算自动返回 loss
            loss = self.forward(context, target, negatives)
            
            # 反向传播和更新
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)