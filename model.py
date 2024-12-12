import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    # 这里的n_embd_c就是C，词嵌入维度
    def __init__(self, n_embd_c, bias):
        super().__init__()
        # 初始化权重参数为1，维度是C
        self.weight = nn.Parameter(torch.ones(n_embd_c))
        # 如果使用偏置项，全初始化成0
        self.bias = nn.Parameter(torch.zeros(n_embd_c)) if bias else None
    
    def forward(self, x):
        out = F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
        return out

class CausalSelfAttention(nn.Module):
    # 因果自注意力
    
    def __init__(self, config):
        super().__init__()
        # 断言，确保每个注意力头可以平分所有维度(C)
        assert config.n_embd % config.n_head == 0
        
        # 设置头数、嵌入维度、以及dropout的概率
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.block_size = config.block_size
        #self.bias = config.bias

        # c_attn用于生成Q,K,V  c_lin用于将自注意力好的输出再次映射
        self.c_attn = nn.Linear(self.n_embd, self.n_embd*3, bias=config.bias)
        self.c_lin = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)

        # 设置dropout
        self.atten_dropout = nn.Dropout(config.dropout)
        self.c_dropout = nn.Dropout(config.dropout)

        # 因果掩码
        # 将这个矩阵转换为下三角矩阵，即保留主对角线及其下方的元素，上方的元素将变为0
        self.register_buffer("bias", torch.tril(torch.ones(self.block_size, self.block_size))
                                        .view(1, 1, self.block_size, self.block_size))

    # 向前传播
    def forward(self, x):
        # 获取输入批次大小，输入句子长度，词嵌入维度
        B, T, C = x.size()

        # 自注意力模块计算 Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # 调整形状成(B, n_head, T, head_size)，以便进行多头注意力计算
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2) 
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2) 
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2) 

        # 自注意力计算,将k最后两个维度(T, head_size)变成(head_size, T)
        # 这样和q @ k就是(T, T)的矩阵，这里的T是输入句子的长度，每一个值就是这个字符的权重
        # k.size(-1)就是C
        '''
        att = [
                    [
                        [0.1, 0.2, 0.3, 0.4],  # 第一行：第1个位置对所有位置的注意力得分
                        [0.5, 0.6, 0.7, 0.8],  # 第二行：第2个位置对所有位置的注意力得分
                        [0.9, 1.0, 1.1, 1.2],  # 第三行：第3个位置对所有位置的注意力得分
                        [1.3, 1.4, 1.5, 1.6],  # 第四行：第4个位置对所有位置的注意力得分
                    ]
                ]

        

        '''
        att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))
            
        # 用因果掩码遮住模型的眼睛
        '''
        att = [
                [[0.1000,   -inf,   -inf,   -inf],
                [0.5000, 0.6000,   -inf,   -inf],
                [0.9000, 1.0000, 1.1000,   -inf],
                [1.3000, 1.4000, 1.5000, 1.6000]
            ]
        '''
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # softmax
        att = F.softmax(att, dim=-1)

        # dropout
        att = self.atten_dropout(att)

        # value
        # q 中的每一个头确实会和 k 中的每一个头进行矩阵乘法
        y = att @ v

        # 将多头注意力的输出还原到原始形状，并传递给后续的层
        y = y.transpose(1, 2).contiguous().view(B, T, C) 

        # 通过线性投影层和残差dropout生成最终的输出
        y = self.c_dropout(self.c_lin(y))
        return y




class MLP(nn.Module):
    # 多层感知机 就是前馈神经网络
    def __init__(self, config):
        super().__init__()
        # 第一个线性层将输入维度扩展为4倍，以增加模型的表达能力
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias)
        # 激活函数
        self.gelu = nn.GELU()
        # 第二个线性层将维度还原回嵌入维度
        self.c_linear = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias)
        # dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)

        x = self.c_linear(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    # Transformer模块
    def __init__(self, config):
        super().__init__()
        # 初始化LayerNorm层，用于输入的归一化
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        # 自注意力
        self.attn = CausalSelfAttention(config)
        # 第二个LayerNorm
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        # MLP
        self.mlp = MLP(config)

    def forward(self, x):
        # y = self.ln_1(x)
        # 这里是之前写了一个bug，找了两个小时，是layerNorm忘记return了
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x

@dataclass
class GPTConfig:
    # 模型参数
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

class GPT(nn.Module):
    # GPT完整封装
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # transformer层的实现
        self.transformer = nn.ModuleDict(dict(
            # 词嵌入层,将vocab_size个字符中每一个字符都映射到n_embd维度
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # 位置嵌入层
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        # 线性层，用于将模型的嵌入表示映射回词汇空间，输出下一个词的概率分布。
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 将词嵌入层的权重与线性层的权重共享。这种技术称为权重绑定，有助于模型的学习和减少参数数量。
        self.transformer.wte.weight = self.lm_head.weight  # 权重共享

        # 初始化所有权重,用自定义的_init_weights函数
        self.apply(self._init_weights)

        # 根据 GPT-2 论文，对残差投影应用特殊的缩放初始化
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        # 报告参数数量
        # print("参数数量: %.2fM" % (self.get_num_params()/1e6,))

    # 使用正态分布初始化权重，均值为0，标准差为0.02
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        device = x.device
        # b表示批次大小，t表示时间步（序列长度）
        b, t = x.size()
        assert t <= self.config.block_size
        # 创建一个张量pos，表示输入序列中每个token的位置索引，形状为(t)
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # 向前传播
        # 词嵌入
        tok_embd = self.transformer.wte(x)
        # 位置嵌入
        pos_embd = self.transformer.wpe(pos)
        # 位置嵌入
        #pos_embd = self.transformer.wpe(pos).unsqueeze(0)  # 添加一个维度


        x = tok_embd + pos_embd
        x = self.transformer.drop(x)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        # 如果给定了一些目标
        # logits 的形状为 (b, vocab_size)，其中：

        # b 是批次大小（batch_size），表示一次输入的句子数量。
        # vocab_size 是词汇表大小，表示模型可以预测的所有可能token的数量。

        # att是一个T*T的矩阵，通过每一行来预测下一个token，再将这个token与y中的正确答案进行loss
        # logits 的形状是 (B, T, V)，其中 B 是批次大小，T 是序列长度，V 是词汇表的大小。
        '''
        x = ["The", "cat", "sat", "on", "the"]
        logits = [
        # batch_size = 1 (假设我们只有一个批次)
        # sequence_length = 5 (当前输入的序列长度)
        # vocab_size = 7 (词汇表的大小)

        [[0.2, 1.0, -0.3, 0.5, 1.5, 0.8, -1.2],  # 第1个时间步预测下一个词的分数，只知道the的情况下预测下一个词
        [0.6, 1.3, -0.4, 0.7, 1.2, 0.6, -1.0],  # 第2个时间步预测下一个词的分数，就是知道the cat的情况下预测下一个
        [0.8, 1.5, -0.5, 0.3, 1.6, 1.1, -0.8],  # 第3个时间步预测下一个词的分数
        [1.1, 0.9, -0.2, 0.6, 1.3, 1.2, -0.5],  # 第4个时间步预测下一个词的分数
        [0.9, 1.4, -0.3, 0.8, 1.7, 0.7, -1.1]]  # 第5个时间步预测下一个词的分数
    ]

        '''
        if targets is not None:
            logits = self.lm_head(x)
            '''
            对于每个样本的每个时间步，logits 会给出该位置的所有词汇的分数。
            然后，cross_entropy 函数会通过 softmax 对 logits 进行归一化，得到每个词的概率分布。
            然后，它会根据实际目标（即 targets 中的真实词索引）来计算每个时间步的损失，最终返回 平均损失。

            logits.view(-1, logits.size(-1)): 它将 logits 的形状从 (batch_size, seq_length, vocab_size) 转换为 (batch_size * seq_length, vocab_size)
            然后，cross_entropy 函数会通过 softmax 对 logits 进行归一化，得到每个词的概率分布
            
            '''
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理时的微优化：只在最后一个位置前向传播 lm_head
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        
        return logits, loss

    # 输入一个序列，输出序列+下一个字符(token)
    @torch.no_grad()
    def generate(self, x, max_new_tokens, temperature=1.0, top_k=None):       
        for _ in range(max_new_tokens):
            # 如果序列上下文变得太长，我们必须在块大小处裁剪它
            x_cond = x if x.size(1) <= self.config.block_size else x[:, -self.config.block_size:]
            
            # 向前传播,获取logits
            logits, _ = self(x_cond)

            # 选择最后一个句子(完整句子)的logits来预测下一个词
            # 根据温度裁剪
            logits = logits[:, -1, :]/temperature
            # 只保留top_k
            if top_k is not None:
                # torch.topk函数返回logits中最大的top_k个值（v）
                v, _  = torch.topk(logits,min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # softmax输出概率
            probs = F.softmax(logits,dim=-1)
            # 从probs中选取下一个词的预测
            x_next = torch.multinomial(probs, num_samples=1)
            # 将采样的索引追加到运行序列中并继续
            x = torch.cat((x, x_next), dim=1)
            
        return x


