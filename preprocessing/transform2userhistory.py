#!/usr/bin/env python
# coding=utf-8

# 将training_set转换用户播放历史，方便后续计算基于item-based协同过滤

import os
import numpy as np

cwd = os.getcwd()  # 获取当前工作目录
f_path = os.path.abspath(os.path.join(cwd, ".."))  # 获取上一级目录
original = open(f_path + '/output/train.txt', 'r')

data_f = f_path + "/output/history.npy"

history_map = dict()

print("===================开始收集用户播放历史==================")
i = 0
line = original.readline()   # 调用文件的 readline()方法
while line:
    if i % 100 == 0:
        print(i)
        i = i + 1
    d = line.split(",")
    user_id = int(d[0])
    video_id = int(d[1])
    score = int(d[2])
    if user_id in history_map:
        s = history_map.get(user_id)
        s.add((video_id, score))
        history_map[user_id] = s
    else:
        s = set()
        s.add((video_id, score))
        history_map[user_id] = s
    line = original.readline()

original.close()

print("===================完成收集用户播放历史==================")

print(len(history_map))

np.save(data_f, history_map)
