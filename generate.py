import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# 模型所在文件夹
out_path = r'D:\vscode_code\Deep_learning\nanoGPT\data\out'  # 模型输出目录
# 词表所在文件
meta_path = r'D:\vscode_code\Deep_learning\nanoGPT\data\meta.pkl'
# 指定生成文本的起始内容
start = "\n" 
# 指定生成文本数量
num_samples = 10
# 每个样本生成的最大 token 数量
max_new_tokens = 500
# 控制生成的文本随机性，1.0 表示无变化，低于 1.0 表示更保守，高于 1.0 表示更随机
temperature = 0.8
# 生成时保留 top_k 个最高概率的 token，其余 token 概率置为 0
top_k = 200
# 设置随机种子以确保结果可复现
seed = 1337
# GPU
device = 'cuda'
# 上下文管理器
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16)

# 获取模型
# 获取路径
ckpt_path = os.path.join(out_path, 'last_ckpt.pt')
# 加载模型文件
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
# 初始化GPT模型
model = GPT(gptconf)
model.load_state_dict(checkpoint['model'])

# 模型准备
model.eval()
model.to(device)

# 提取词表
print(f"从{meta_path} 加载词表信息")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
char2num = meta['char2num']
num2char = meta['num2char']

# 编码器
def encode(ch):
    en = [char2num[c] for c in ch]
    return en

# 解码器
def decode(num):
    de = ''.join([num2char[n] for n in num])
    return de

# 将起始文本编码为 token 序列
start_ids = encode(start)
# 将编码结果转换为张量
# None 表示在第一个维度位置插入一个新维度，使得原本的一维张量变成二维张量。
# 具体来说，原来形状为 (N,) 的张量现在变为 (1, N) 的形状。
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# 生成文本
with torch.no_grad():
    # 使用指定的数据类型上下文
    with ctx:
        for k in range(num_samples):
            print(k)
            # 生成文本序列
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            # 分隔生成的样本
            print('---------------')