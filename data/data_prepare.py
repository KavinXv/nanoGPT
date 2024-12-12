import os
import numpy as np
import pickle
from tokenizers import Tokenizer

# 加载训练好的 Tokenizer
tokenizer = Tokenizer.from_file("./chinese_bpe_tokenizer.json")

# 数据集
data_path = r'./long.txt'

with open(data_path, 'r', encoding='utf-8') as f:
    data = f.read()

print(f"数据集总长度：{len(data)}")
print()

# 划分数据集
data_len = len(data)
train_data = data[:int(data_len*0.9)]
val_data = data[int(data_len*0.9):]

# 编码数据集，用训练好的 Tokenizer 进行编码
train_ids = tokenizer.encode(train_data).ids
val_ids = tokenizer.encode(val_data).ids

# 打印一些编码后的 ID，以便检查
print("示例训练数据编码后的 ID:", train_ids[:10])  # 打印前10个编码后的 ID

# 导出为二进制文件
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(r'./train.bin')
val_ids.tofile(r'./val.bin')

# 保存元信息，用于后续编码解码
meta = {
    'vocab_size': tokenizer.get_vocab_size(),  # 获取词汇表大小
    'tokenizer': tokenizer  # 直接保存训练好的 Tokenizer 对象
}

# 保存 Tokenizer 和其他元信息
meta_path = r'./meta.pkl'
with open(meta_path, 'wb') as f:
    pickle.dump(meta, f)

print("处理完毕，数据已保存到 train.bin 和 val.bin，元信息已保存到 meta.pkl")
