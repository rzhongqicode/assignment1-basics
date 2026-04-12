from collections import Counter
import torch
import regex as re
from tqdm import tqdm
# a = "abc".encode("utf-8")
# b = "def".encode("utf-8")
# print(a)
# print(b)
# print(a<b)
# print(a+b)
# a = [1,2,3]
# print(a[:-1])

# counter = Counter()
# counter["apple"] = 2
# counter["banana"] += 1
# # counter["apple"] -= 2
# del counter["apple"]
# del counter["orange"]
# counter["apple"] = 1
# counter2 = Counter({"apple": 3, "pie": 4})
# print(counter)
# # print(counter.items())
# print(counter2)
# print(counter - counter2)
# a = tuple("仁".encode("utf-8"))
# b = "仁".encode("utf-8")
# print(a)
# print(b)
# # print(bytes([11])+bytes([12]))
# # for byte in
# print(a[0])
# print(type(b[0]))
# print(type(a[0]))

# a = torch.Tensor([[1,2,3],
#                  [4,5,6],
#                  [7,8,9]])
# index = torch.IntTensor([[0,1],
#                      [1,2]])
# print(index)
# b = a[index]
# print(b)

# a = []
# for i in a:
#     print(i)

# print(re.split(r"<\|end\|>", "hello<|end|>world"))
# print(re.split(r"(<\|end\|>)", "hello<|end|>world"))
# a = [1,2]
# b = [3,4]
# c = a.extend(b)
# print(c)
# a = None
# for i in a:
#     print(i)
# a = 4
# b = 8 % 3
# print(b)
# a = torch.Tensor([[1,2],
#                  [3,4]])
# b = torch.Tensor([[1,1],
#                   [0,0]]).to(torch.int)
# c = a[b]
# print(c)


# with open("data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as f:
#     # 这里只会显示 12345it [00:10, 1200it/s] 没有进度条
#     for line in tqdm(f, desc="读取文件"):
#         pass

# 1. 先快速获取总行数
with open("data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

# 2. 再次打开文件，并传入 total
with open("data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as f:
    for line in tqdm(f, total=total_lines, desc="精准读取文件"):
        # 此时就拥有完美的百分比进度条了！
        pass