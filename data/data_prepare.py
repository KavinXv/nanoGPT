import os
import numpy as np
import pickle

# 数据集
data_path = r'D:\vscode\Python\nanoGPT\data\long.txt'

with open(data_path,'r',encoding = 'utf-8') as f:
    data = f.read()
print(f"数据集总长度：{len(data)}")
print()

# 获取所有出现过的字符
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("所有出现过的字符：",''.join(chars))
print(f"字符表大小：{vocab_size:,}")

# 创建从字符到数字的映射，为编码器解码器做准备
# enumerate会生成一系列的元组，每个元组包含两个元素：第一个是当前元素的索引（从 0 开始），第二个是当前元素的值
char2num = {ch:i for i,ch in enumerate(chars)}
num2char = {i:ch for i,ch in enumerate(chars)}

# 编码器
def encode(ch):
    en = [char2num[c] for c in ch]
    return en

# 解码器
def decode(num):
    de = ''.join([num2char[n] for n in num])
    return de

# 划分数据集
data_len = len(data)
train_data = data[:int(data_len*0.9)]
val_data = data[int(data_len*0.9):]

# 编码一下数据集，用于训练
train_ids = encode(train_data)
val_ids = encode(val_data)
# print(val_ids)

# 导出为二进制文件
train_ids = np.array(train_ids,dtype = np.uint16)
val_ids = np.array(val_ids,dtype = np.uint16)

train_ids.tofile(r'D:\vscode\Python\nanoGPT\data\train.bin')
val_ids.tofile(r'D:\vscode\Python\nanoGPT\data\val.bin')

# 保存元信息，用于后续编码解码
meta = {
    'vocab_size':vocab_size,
    'char2num':char2num,
    'num2char':num2char,
}

meta_path = r'D:\vscode\Python\nanoGPT\data\meta.pkl'
with open(meta_path, 'wb') as f:
    pickle.dump(meta, f)