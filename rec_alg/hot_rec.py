#!/usr/bin/env python
# coding=utf-8

# 将training_set转换为三元组(userid，videoid，score)

import os

N = 20  # 推荐的热门视频个数

cwd = os.getcwd()  # 获取当前工作目录
f_path = os.path.abspath(os.path.join(cwd, ".."))  # 获取上一级目录

train = f_path + "/output/train.txt"

m = dict()

f_train = open(train)        # 返回一个文件对象

# 计算每个视频累计评分之和
line = f_train.readline()   # 调用文件的 readline()方法
while line:

    d = line.split(",")
    video_id = d[1]
    score = int(d[2])

    if m.has_key(video_id):
        m[video_id] = m[video_id]+score
    else:
        m[video_id] = score

    line = f_train.readline()

sorted_list = sorted(m.items(), key=lambda item: item[1], reverse=True)

print(sorted_list[:N])

f_train.close()
