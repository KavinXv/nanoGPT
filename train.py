import math
import inspect
from dataclasses import dataclass
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from model import GPTConfig, GPT
from tokenizers import Tokenizer

# 超参数设置
batch_size = 12  # 在一次训练或推理中输入到模型中的样本数量  B
block_size = 1024   # 每个输入序列的最大长度   T
max_iters = 15000
eval_iters = 1000
learning_rate = 1e-2        # 学习率
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
best_val_loss = 1e9

weight_decay = 1e-1  # 权重衰减
beta1 = 0.9  # adamw 优化器的 beta1 参数
beta2 = 0.95  # adamw 优化器的 beta2 参数
grad_clip = 1.0  # 梯度裁剪的值，0 表示禁用

# 数据集
data_train = r'./data/train.bin'
data_val = r'./data/val.bin'
out_path = r'./data/out'  # 模型输出目录
meta_path = r'./data/meta.pkl'
os.makedirs(out_path, exist_ok=True)

# 模型配置
n_layer = 10  # Transformer 模型的层数
n_head = 12  # 每个注意力层的头数
n_embd = 768  # 嵌入向量的维度大小
dropout = 0.0  # Dropout 概率，用于防止过拟合。预训练时推荐使用 0，微调时尝试大于 0.1
bias = False  # 是否在 LayerNorm 和 Linear 层中使用偏置项


# config_keys 获取当前环境下所有配置键值对，过滤掉私有变量和非基本数据类型变量
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]

# 保存当前配置为字典，以便后续记录和日志
config = {k: globals()[k] for k in config_keys}

# 尝试从数据集中获取 vocab_size
vocab_size = None
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

tokenizer = Tokenizer.from_file("./data/chinese_bpe_tokenizer.json")  # 加载训练好的 Tokenizer
vocab_size = meta['vocab_size']
print(f"从 {meta_path} 读取到词汇表大小 vocab_size = {vocab_size}")

# 初始化模型
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)

print("模型初始化中...")
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)
print("模型初始化完毕!")
print(sum(p.numel() for p in model.parameters())/1e6, "M 个参数")

# 上下文管理器
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
# 优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    weight_decay=weight_decay,
)

# 获取数据批次
def get_batch(split):
    if split == 'train':
        data = np.memmap(data_train, dtype=np.uint16, mode='r')
    else:
        data = np.memmap(data_val, dtype=np.uint16, mode='r')
    
    # 随机选择 batch_size 个数据快的起始索引,就是随机生成从0到len(data)-block_size中的一个数,共batch_size个
    idx = torch.randint(len(data) - block_size, (batch_size,))
    
    # 通过 tokenizer 编码文本数据为 ID
    x = torch.stack([torch.tensor((data[i:i+block_size]).astype(np.int64)) for i in idx])
    y = torch.stack([torch.tensor((data[i+1:i+block_size+1]).astype(np.int64)) for i in idx])

    # 异步传入GPU
    x = x.pin_memory().to(device, non_blocking=True)
    y = y.pin_memory().to(device, non_blocking=True)

    return x, y

# 计算loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# 训练
X, Y = get_batch('train')
lr = learning_rate
t0 = time.time()

for iter in range(max_iters):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if iter % eval_iters == 0 or iter == max_iters - 1:
        losses = estimate_loss()    
        # if iter > 1500 and losses['val'] >= 6.2:
            # break
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print(tokenizer.decode(model.generate(context, max_new_tokens=500)[0].tolist()))

        # 保存模型检查点
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter,
            'best_val_loss': best_val_loss,
            'config': config,
        }
        print(f"saving last checkpoint to {out_path}")
        torch.save(checkpoint, os.path.join(out_path, 'last_ckpt.pt'))

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            print(f"saving best checkpoint to {out_path}")
            torch.save(checkpoint, os.path.join(out_path, 'best_ckpt.pt'))

    xb, yb = get_batch('train')

    # logits：模型的预测输出，形状为 (batch_size, seq_len, vocab_size)，表示对每个位置的词（token）在整个词表上的概率分布。
    # loss：与 yb 计算的交叉熵损失，用于衡量模型的预测与目标之间的距离，用于反向传播。
    logits, loss = model(xb, yb)

    loss.backward()

    if grad_clip:
        # 如果设置了梯度裁剪，裁剪梯度以防止爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    print(f"step {iter}: train loss {loss:.4f}, time {dt}")
