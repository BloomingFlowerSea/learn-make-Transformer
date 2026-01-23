import torch
from torch import nn
import torch.nn.functional as F
import math


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEmbedding, self).__init__()

        # 1. 计算位置编码矩阵
        encoding = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2).float()

        encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

        # 2. 使用 register_buffer 解决设备同步问题
        # 即使你不加 batch 维度，register_buffer 也会管理好这个 tensor
        self.register_buffer('pe', encoding)

    def forward(self, X):
        # 利用广播机制：(batch, seq, dim) + (seq, dim)
        # 这里的 self.pe 会自动根据 X 的设备（CPU/GPU）进行匹配
        X = X + self.pe[:X.size(1), :]
        return X


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=1000, drop_prob=0.5):
        super(TransformerEmbedding, self).__init__()
        self.d_model = d_model
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(d_model, max_len)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, X):
        X = self.token_emb(X) * math.sqrt(self.d_model)  # 匹配X与pos的大小级别
        X = self.pos_emb(X)
        X = self.drop_out(X)
        return X


class MutiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MutiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_qkv(self, X):
        X = X.reshape(X.shape[0], X.shape[1], self.n_head, -1)
        X = X.permute(0, 2, 1, 3)
        return X

    def forward(self, Q, K, V, mask=None):
        n_dim = self.d_model // self.n_head
        Q, K, V = self.w_q(Q), self.w_k(K), self.w_v(V)
        Q, K, V = self.transpose_qkv(Q), self.transpose_qkv(K), self.transpose_qkv(V)

        # B, N, L, H *  B, N, H, L = B, N, L, L
        score = Q @ K.transpose(-1, -2) / math.sqrt(n_dim)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        # B, N, L, L *  B, N, L, H = B, N, L, H
        score = self.softmax(score) @ V
        # B, N, L, H -> B, L, N, H -> B, L, N * H
        score = score.permute(0, 2, 1, 3)
        score = score.reshape(score.shape[0], score.shape[1], -1)
        out = self.w_combine(score)
        return out


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-9):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, X):
        mean = X.mean(-1, keepdim=True)
        var = X.var(-1, unbiased=False, keepdim=True)
        X = (X - mean) / (torch.sqrt(var) + self.eps)
        X = self.gamma * X + self.beta
        return X


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.dense1 = nn.Linear(d_model, hidden)
        self.dense2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, X):
        X = self.relu(self.dense1(X))
        X = self.dropout(X)
        X = self.dense2(X)
        return X


class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MutiHeadAttention(d_model, n_head)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)

    def forward(self, X, mask=None):
        Y = self.addnorm1(X, self.attention(X, X, X, mask))
        return self.addnorm2(Y, self.ffn(Y))


class Encoder(nn.Module):
    def __init__(self, vocb_size, d_model, ffn_hidden,
                 n_head, n_layer, max_len=1000, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(vocb_size, d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, ffn_hidden, n_head, dropout)
                for _ in range(n_layer)
            ]
        )

    def forward(self, X, s_mask=None):
        X = self.embedding(X)
        for layer in self.layers:
            X = layer(X, s_mask)
        return X


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attention1 = MutiHeadAttention(d_model, n_head)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.attention2 = MutiHeadAttention(d_model, n_head)
        self.addnorm2 = AddNorm(d_model, dropout)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.addnorm3 = AddNorm(d_model, dropout)

    def forward(self, dec, enc, t_mask, s_mask):
        X = self.addnorm1(dec, self.attention1(dec, dec, dec, t_mask))
        Y = self.addnorm2(X, self.attention2(X, enc, enc, s_mask))
        Y2 = self.addnorm3(Y, self.ffn(Y))
        return Y2


class Decoder(nn.Module):
    def __init__(self, vocb_size, d_model, ffn_hidden,
                 n_head, n_layer, max_len=1000, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(vocb_size, d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, ffn_hidden, n_head, dropout)
                for _ in range(n_layer)
            ]
        )
        self.dense = nn.Linear(d_model, vocb_size)

    def forward(self, X, enc, t_mask=None, s_mask=None):
        X = self.embedding(X)
        for layer in self.layers:
            X = layer(X, enc, t_mask, s_mask)
        X = self.dense(X)
        return X


class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, enc_voc_size, dec_voc_size,
                 d_model, n_head, ffn_hidden, n_layer, max_len=1000, dropout=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(enc_voc_size, d_model, ffn_hidden, n_head, n_layer, max_len, dropout)
        self.decoder = Decoder(dec_voc_size, d_model, ffn_hidden, n_head, n_layer, max_len, dropout)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_pad_mask(self, Q, K, pad_idx_Q, pad_idx_K):
        # 利用广播机制：(batch, 1, len_q, 1) & (batch, 1, 1, len_k)
        mask_K = K.ne(pad_idx_K).unsqueeze(1).unsqueeze(2)
        mask_Q = Q.ne(pad_idx_Q).unsqueeze(1).unsqueeze(3)
        return mask_Q & mask_K

    def make_causal_mask(self, Q, K):
        # 形状为 (seq_q, seq_k)，广播后适配 (batch, n_head, seq_q, seq_k)
        mask = torch.tril(torch.ones(Q.shape[1], K.shape[1])).bool()
        return mask.to(Q.device)  # 确保在同一设备上

    def forward(self, src, trg):
        # 1. 为 Encoder 生成 Padding Mask
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)

        # 2. 为 Decoder 生成混合 Mask (Padding + Causal)
        trg_pad_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx)
        trg_causal_mask = self.make_causal_mask(trg, trg)
        trg_mask = trg_pad_mask & trg_causal_mask

        # 3. 为 Cross-Attention 生成 Mask (src 与 trg 之间的 padding 关系)
        # 告诉 Decoder 哪些 Encoder 输出的位置是 padding
        src_trg_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)

        # 4. 执行编码和解码
        enc_out = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_out, trg_mask, src_trg_mask)

        return out
