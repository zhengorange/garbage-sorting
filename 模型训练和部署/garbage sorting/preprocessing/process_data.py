"""
数据总数为54411，细类别数为158，总类别为五类。
划分训练集和测试集，训练集90%，测试集10%。
"""

import json
import glob
import os
import shutil
from tqdm import tqdm

# ima_list = glob.glob("../data/train/*/*")
# print("训练集图像数", len(ima_list))
# ima_list = glob.glob("../data/test/*/*")
# print("测试集图像数", len(ima_list))
# exit()

with open('../data/garbage_classification.json', 'r') as f:
    class_list = {int(k): v for k, v in json.load(f).items()}
ima_list = glob.glob("../data/image/*/*")
print("原始图像数", len(ima_list))
class_items = {}
for item in ima_list:
    kind = class_list[int(item.split('/')[-2])].split("_")[0]
    if kind not in class_items:
        class_items[kind] = [item]
    else:
        class_items[kind].append(item)

print(class_items.keys())

train_set = []
test_set = []
for k, v in class_items.items():
    split = int(len(v) * 0.9)
    train_set += v[:split]
    test_set += v[split:]

print("训练集", len(train_set))
print("测试集", len(test_set))

print(train_set[:3])

for item in tqdm(test_set):
    kind = class_list[int(item.split('/')[-2])].split("_")[0]
    name = class_list[int(item.split('/')[-2])] + "_" + item.split('/')[-1]
    if not os.path.exists("../data/test/{}".format(kind)):
        os.mkdir("../data/test/{}".format(kind))
    shutil.copy(item, "../data/test/{}/{}".format(kind, name))
for item in tqdm(train_set):
    kind = class_list[int(item.split('/')[-2])].split("_")[0]
    name = class_list[int(item.split('/')[-2])] + "_" + item.split('/')[-1]
    if not os.path.exists("../data/train/{}".format(kind)):
        os.mkdir("../data/train/{}".format(kind))
    shutil.copy(item, "../data/train/{}/{}".format(kind, name))

