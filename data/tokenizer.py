import json
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

# 1. 创建 BPE 模型
tokenizer = Tokenizer(models.BPE())

# 2. 预处理器：按字符分隔中文（按需选择 Metaspace、CharDelimiterSplit、Whitespace 等）
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

# 3. 使用默认的解码器
tokenizer.decoder = decoders.ByteLevel()

# 4. 创建 BPE 训练器
trainer = trainers.BpeTrainer(
    vocab_size=30000,  # 词汇表大小
    min_frequency=2,  # 最小词频
    special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<eos>"]  # 特殊符号
)

# 5. 文件路径，确保文件路径正确
file_path = "./long.txt"

# 6. 训练分词器
# 方式一：直接加载文件路径，推荐用于大文件
tokenizer.train([file_path], trainer)

# 方式二：加载文件内容，推荐用于小数据集
# with open(file_path, 'r', encoding='utf-8') as f:
#     data = [json.loads(line)['text'] for line in f]
# tokenizer.train_from_iterator(data, trainer)

# 7. 保存分词器
tokenizer.save("./chinese_bpe_tokenizer.json")

print("分词器训练完成并已保存为 'chinese_bpe_tokenizer.json'")
