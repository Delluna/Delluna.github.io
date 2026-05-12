---
title: 'Transformer'
date: 2026-05-08
permalink: /posts/2026/05/Transformer/
tags:
  - Transformer
---

A description for Transformer.

# Transformer

## Attention is all you need

论文地址：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)，由Google 团队于2017年提出。核心思想：注意力机制。

算法：

1. 分词
2. Embedding
3. 位置编码
4. 自注意力机制
5. 多头
6. FFN
7. 残差连接&&Layer Norm
8. encoder-decoder
9. Mask Attn

优点：

1. 并行
2. 解决长距离依赖问题
3. 结构灵活，可扩展Scale
4. 支持预训练+微调

原本任务：机器翻译。扩展任务：encoder-only用于理解任务；decoder-only用于生成任务；vit用于图像/视频处理任务；LLM用于AI大模型；...

nlp数据处理过程：

1. 数据预处理：

句子->分词->token(input_ids,attn_mask)

2. 进入模型：

embedding

position embedding

stacked encoder/decoder layer{注意力层(缩放点积注意力，多头注意力，mask注意力),前馈神经网络}

线性层

3. 模型输出logits：

softmax输出下一词的概率分布，根据选择策略（topk,topp,temperature等）选择一个词，自回归输出完整句子。

代码：

```python
# https://zhuanlan.zhihu.com/p/581334630

import torch
import torch.nn as nn

import math


# 位置编码
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()

        # Assume d_model is an even number for convenience
        assert d_model % 2 == 0

        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)
        pos, two_i = torch.meshgrid(i_seq, j_seq)
        pe_2i = torch.sin(pos / 10000**(two_i / d_model))
        pe_2i_1 = torch.cos(pos / 10000**(two_i / d_model))
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(1, max_seq_len, d_model)

        self.register_buffer('pe', pe, False)

    def forward(self, x: torch.Tensor):
        n, seq_len, d_model = x.shape
        pe: torch.Tensor = self.pe
        assert seq_len <= pe.shape[1]
        assert d_model == pe.shape[2]
        rescaled_x = x * d_model**0.5
        return rescaled_x + pe[:, 0:seq_len, :]



# 多头自注意力机制
class MultiHeadAttention(nn.Module):

    def __init__(self, heads: int, d_model: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % heads == 0
        # dk == dv
        self.d_k = d_model // heads
        self.heads = heads
        self.d_model = d_model
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        # batch should be same
        assert q.shape[0] == k.shape[0]
        assert q.shape[0] == v.shape[0]
        # the sequence length of k and v should be aligned
        assert k.shape[1] == v.shape[1]

        n, q_len = q.shape[0:2]
        n, k_len = k.shape[0:2]
        q_ = self.q(q).reshape(n, q_len, self.heads, self.d_k).transpose(1, 2)
        k_ = self.k(k).reshape(n, k_len, self.heads, self.d_k).transpose(1, 2)
        v_ = self.v(v).reshape(n, k_len, self.heads, self.d_k).transpose(1, 2)

        attention_res = attention(q_, k_, v_, mask)
        concat_res = attention_res.transpose(1, 2).reshape(
            n, q_len, self.d_model)
        concat_res = self.dropout(concat_res)

        output = self.out(concat_res)
        return output


MY_INF = 1e12

def attention(q: torch.Tensor,
              k: torch.Tensor,
              v: torch.Tensor,
              mask: Optional[torch.Tensor] = None):
    '''
    Note: The dtype of mask must be bool
    '''
    # q shape: [n, heads, q_len, d_k]
    # k shape: [n, heads, k_len, d_k]
    # v shape: [n, heads, k_len, d_v]
    assert q.shape[-1] == k.shape[-1]
    d_k = k.shape[-1]
    # tmp shape: [n, heads, q_len, k_len]
    tmp = torch.matmul(q, k.transpose(-2, -1)) / d_k**0.5
    if mask is not None:
        tmp.masked_fill_(mask, -MY_INF)
    tmp = F.softmax(tmp, -1)
    # tmp shape: [n, heads, q_len, d_v]
    tmp = torch.matmul(tmp, v)
    return tmp


# 前馈网络
class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(F.relu(x))
        x = self.layer2(x)
        return x

# 编码器
class Encoder(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 pad_idx: int,
                 d_model: int,
                 d_ff: int,
                 n_layers: int,
                 heads: int,
                 dropout: float = 0.1,
                 max_seq_len: int = 120):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, pad_idx)
        self.pe = PositionalEncoding(d_model, max_seq_len)
        self.layers = []
        for i in range(n_layers):
            self.layers.append(EncoderLayer(heads, d_model, d_ff, dropout))
        self.layers = nn.ModuleList(self.layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask: Optional[torch.Tensor] = None):
        x = self.embedding(x)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class EncoderLayer(nn.Module):

    def __init__(self,
                 heads: int,
                 d_model: int,
                 d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(heads, d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask: Optional[torch.Tensor] = None):
        tmp = self.self_attention(x, x, x, src_mask)
        tmp = self.dropout1(tmp)
        x = self.norm1(x + tmp)
        tmp = self.ffn(x)
        tmp = self.dropout2(tmp)
        x = self.norm2(x + tmp)
        return x


# 解码器 
class Decoder(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 pad_idx: int,
                 d_model: int,
                 d_ff: int,
                 n_layers: int,
                 heads: int,
                 dropout: float = 0.1,
                 max_seq_len: int = 120):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, pad_idx)
        self.pe = PositionalEncoding(d_model, max_seq_len)
        self.layers = []
        for i in range(n_layers):
            self.layers.append(DecoderLayer(heads, d_model, d_ff, dropout))
        self.layers = nn.Sequential(*self.layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x,
                encoder_kv,
                dst_mask: Optional[torch.Tensor] = None,
                src_dst_mask: Optional[torch.Tensor] = None):
        x = self.embedding(x)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoder_kv, dst_mask, src_dst_mask)
        return x        

class DecoderLayer(nn.Module):

    def __init__(self,
                 heads: int,
                 d_model: int,
                 d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(heads, d_model, dropout)
        self.attention = MultiHeadAttention(heads, d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self,
                x,
                encoder_kv: torch.Tensor,
                dst_mask: Optional[torch.Tensor] = None,
                src_dst_mask: Optional[torch.Tensor] = None):
        tmp = self.self_attention(x, x, x, dst_mask)
        tmp = self.dropout1(tmp)
        x = self.norm1(x + tmp)
        tmp = self.attention(x, encoder_kv, encoder_kv, src_dst_mask)
        tmp = self.dropout2(tmp)
        x = self.norm2(x + tmp)
        tmp = self.ffn(x)
        tmp = self.dropout3(tmp)
        x = self.norm3(x + tmp)
        return x


# transformer整体框架
class Transformer(nn.Module):

    def __init__(self,
                 src_vocab_size: int,
                 dst_vocab_size: int,
                 pad_idx: int,
                 d_model: int,
                 d_ff: int,
                 n_layers: int,
                 heads: int,
                 dropout: float = 0.1,
                 max_seq_len: int = 200):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, pad_idx, d_model, d_ff,
                               n_layers, heads, dropout, max_seq_len)
        self.decoder = Decoder(dst_vocab_size, pad_idx, d_model, d_ff,
                               n_layers, heads, dropout, max_seq_len)
        self.pad_idx = pad_idx
        self.output_layer = nn.Linear(d_model, dst_vocab_size)

    def generate_mask(self,
                      q_pad: torch.Tensor,
                      k_pad: torch.Tensor,
                      with_left_mask: bool = False):
        # q_pad shape: [n, q_len]
        # k_pad shape: [n, k_len]
        # q_pad k_pad dtype: bool
        assert q_pad.device == k_pad.device
        n, q_len = q_pad.shape
        n, k_len = k_pad.shape

        mask_shape = (n, 1, q_len, k_len)
        if with_left_mask:
            mask = 1 - torch.tril(torch.ones(mask_shape))
        else:
            mask = torch.zeros(mask_shape)
        mask = mask.to(q_pad.device)
        for i in range(n):
            mask[i, :, q_pad[i], :] = 1
            mask[i, :, :, k_pad[i]] = 1
        mask = mask.to(torch.bool)
        return mask

    def forward(self, x, y):

        src_pad_mask = x == self.pad_idx
        dst_pad_mask = y == self.pad_idx
        src_mask = self.generate_mask(src_pad_mask, src_pad_mask, False)
        dst_mask = self.generate_mask(dst_pad_mask, dst_pad_mask, True)
        src_dst_mask = self.generate_mask(dst_pad_mask, src_pad_mask, False)
        encoder_kv = self.encoder(x, src_mask)
        res = self.decoder(y, encoder_kv, dst_mask, src_dst_mask)
        res = self.output_layer(res)
        return res

```

**Transformer Attention（注意力机制）矩阵运算与物理含义**

**维度定义及意义（假设 $n$ 为序列长度）**

- In (Embedding): $n \times d_{embedding}$
- Q (Query): $n \times d_{tmp}$, Q的每一行表示每个token的Q信息。
- K (Key): $n \times d_{tmp}$, $K^{T}$的每一列表示每个token的K信息。
- V (Value): $n \times d_{embedding}$
- Attn: $n \times n$, Attn的第i行第j列表示第i个token的Q信息与第j个token的K信息的相关性，简单理解为 $Attn(i,j) = R(token_{i},token_{j})$。
- Out: $n \times d_{embedding}$ —— “融合上下文后的新特征”

$Q$ 提出需求, $K$ 提供索引, $QK^T$ 计算相关性, $V$ 提供信息。

**多头注意力 (Multi-Head)**

- $Q_{head}, K_{head}: head \times n \times (d_{tmp} // head)$
- $V_{head}: head \times n \times (d_{embedding} // head)$
- $W^O$：其维度通常为 $d_{embedding} \times d_{embedding}$

每个头在不同的特征子空间里独立进行上述的“加权平均”，再通过 concat 恢复维度，经过一个线性映射层 $W^O$ ,使得最终的 $V_{new}$ 每一个 Token 都集成了来自多个子空间的上下文信息。

**矩阵乘法： $V_{new} = Attn \times V$**

- 运算: $V_{new}$ 的第 $i$ 行第 $j$ 列 = $Attn$ 的第 $i$ 行与 $V$ 的第 $j$ 列的点积。
- 含义: 所有token第 $j$ 维特征的 $Attn$ 加权平均得到第 $i$ 个token在第 $j$ 维的特征。
- 结论: $V_{new}$ 的第 $i$ 行是全文所有 Token 原始特征（ $V$ 的每一行）的加权线性组合。 这意味着第 $i$ 个 Token 在“看过”全篇后，吸收了相关 Token 的信息，更新了自己的语义。

**Softmax**

 **`dim=-1` 代表最后一个维度，也就是“列”的方向，但它的操作效果是每一行内部，跨越所有的列进行计算，让每一行的和为 1。**

在计算 $attn = \text{softmax}(\frac{QK^T}{\sqrt{d}}, \text{dim}=-1)$ 时：

* 每一行代表一个 **Query (第 $i$ 个 token)**。
* 我们需要知道这个 Query 对所有 **Key (所有 token)** 的注意力分配。
* Query $i$ 对 Key $1$ 的分值、对 Key $2$ 的分值... 全都在**第 $i$ 行**。
* 为了让 Query $i$ 的总注意力权重为 $100\%$，我们必须对**这一行**做归一化。
* 所以，必须沿着“列”的方向（`dim=-1`）去做 Softmax。
* 它保证了：**每一个 Token (Query) 看到的全局视野 (Keys) 权重加和为 1。**

---

## NLP

[HF LLM Courese](https://huggingface.co/learn/llm-course)

NLP主要分为两大类任务NLU（自然语言理解）和NLG（自然语言生成）。

|      Model      |           Examples           |                            Tasks                             |
| :-------------: | :--------------------------: | :----------------------------------------------------------: |
|  Encoder-only   | BERT, DistilBERT, ModernBERT | Sentence classification, named entity recognition, extractive question answering |
|  Decoder-only   |  GPT, LLaMA, Gemma, SmolLM   |     Text generation, conversational AI, creative writing     |
| Encoder-decoder |   BART, T5, Marian, mBART    |  Summarization, translation, generative question answering   |

### NLU

#### BERT(encoder-only)

Google 团队于2018年提出 核心思想：

1. encoder-only结构
2. 预训练+微调
3. 通过双向注意力机制捕捉上下文语义 

预训练任务：

1. MLM（完形填空）
2. NSP（判断两句子是否连续） 

对比Word2Vec的优点：

1. 解决一词多义的问题（Bert根据上下文语境为一个词生成不同的Embedding ）。

核心算法：

1. 三向量加和
2. tokenization
3. 自编码
4. 训练过程80-10-10掩码策略
5. 特殊标记[CLS]包含句子的全局语义信息

### NLG

#### GPT(decoder-only)

GPT Open AI团队于2018年提出GPT-1

## Vision

[HF Community Computer Vision Course](https://huggingface.co/learn/computer-vision-course/)

|      |        Feature         |                        Image                         |                         Video                         |
| :--: | :--------------------: | :--------------------------------------------------: | :---------------------------------------------------: |
|  1   |          Type          |                Single moment in time                 |             Sequence of images over time              |
|  2   |  Data Representation   |            Typically a 2D array of pixels            |            Typically a 3D array of frames             |
|  3   |       File types       |                  JPEG,PNG,RAW, etc.                  |                  MP4,AVI, MOV, etc.                   |
|  4   |   Data Augmentation    |             Flipping, rotating, cropping             |    Temporal jittering, speed variations, occlusion    |
|  5   |   Feature Extraction   |               Edges, textures, colors                |  Edges, textures, colors, optical flow, trajectories  |
|  6   |    Learning Models     |                         CNNs                         |                     RNNs, 3D CNNs                     |
|  7   | Machine Learning Tasks | Image classification, Segmentation, Object Detection | Video action recognition, temporal modeling, tracking |
|  8   |   Computational Cost   |                    Less expensive                    |                    More expensive                     |
|  9   |      Applications      |    Facial recognition for security access control    |  Sign language interpretation for live communication  |

### Image

#### ViT

https://aistudio.baidu.com/projectdetail/2293050

#### SwinT

### Video

### Diffusion

## MM

CLIP
