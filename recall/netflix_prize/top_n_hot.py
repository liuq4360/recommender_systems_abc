#!/usr/bin/env python
# coding=utf-8

# 将training_set转换为三元组(userid，videoid，score)

import os
import numpy as np


N = 30  # 推荐的热门视频个数

cwd = os.getcwd()  # 获取当前工作目录
f_path = os.path.abspath(os.path.join(cwd, "..", ".."))  # 获取上一级目录

train = f_path + "/output/netflix_prize/train.txt"

m = dict()

f_train = open(train)        # 返回一个文件对象

# 计算每个视频累计评分之和
line = f_train.readline()   # 调用文件的 readline()方法
while line:
    d = line.split(",")
    video_id = int(d[1])
    score = int(d[2])
    if video_id in m:
        m[video_id] = m[video_id]+score
    else:
        m[video_id] = score
    line = f_train.readline()
sorted_list = sorted(m.items(), key=lambda item: item[1], reverse=True)
topN = sorted_list[:N]

print(topN)

f_train.close()


hot_rec_map = {"hot": sorted_list[:N]}
hot_path = f_path + "/output/netflix_prize/hot_rec.npy"
np.save(hot_path, hot_rec_map)

# 最终的推荐结果的数据结构如下：
# {"hot": [(1905, 564361), (2452, 462715), (3938, 448276)]} (item_id,score)


