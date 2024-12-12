from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers import decoders

# 加载训练好的 Tokenizer
# tokenizer = Tokenizer.from_file("./chinese_bpe_tokenizer.json")
tokenizer = Tokenizer.from_file("./white_tokenizer.json")

# 编码：将中文文本转换为 ID 序列
text = "<s> 许久之后，路明非才从车后厢出来，回到副驾驶座上坐下。 </s>"


encoded = tokenizer.encode(text)

print("Encoded:", encoded.ids)

# 解码：将 ID 序列转换回文本
decoded = tokenizer.decode(encoded.ids)

print("Decoded:", decoded)


