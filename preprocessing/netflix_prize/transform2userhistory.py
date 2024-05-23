#!/usr/bin/env python
# coding=utf-8

# 将training_set转换用户播放历史，方便后续计算基于item-based协同过滤

import os
import numpy as np


def user_action_to_map(source_path, store_path):
    """
    将用户的行为转化为dict存到磁盘，主要是将train和test分别存起来，train用于做个性化推荐，
    而test用于离线效果评估

    :param source_path: 存放原始数据的路径
    :param store_path: 收集用户行为后的结构存放的路径，以dict数据结构存为txt文件
    :return: null
    """
    rec_map = dict()
    source = open(source_path, 'r')
    print("===================开始收集用户播放历史==================")
    i = 0
    line = source.readline()   # 调用文件的 readline()方法
    while line:
        if i % 100 == 0:
            print(i)
        i = i + 1
        d = line.split(",")
        user_id = int(d[0])
        video_id = int(d[1])
        score = int(d[2])
        if user_id in rec_map:
            s = rec_map.get(user_id)
            s.add((video_id, score))
            rec_map[user_id] = s
        else:
            s = set()
            s.add((video_id, score))
            rec_map[user_id] = s
        line = source.readline()
    source.close()

    print("===================完成收集用户播放历史==================")

    print(len(rec_map))

    np.save(store_path, rec_map)


cwd = os.getcwd()  # 获取当前工作目录
f_path = os.path.abspath(os.path.join(cwd, "..", ".."))  # 获取上一级目录

train_path = f_path + "/output/netflix_prize/train.txt"
test_path = f_path + "/output/netflix_prize/test.txt"

train_store_path = f_path + "/output/netflix_prize/train_play_action.npy"
test_store_path = f_path + "/output/netflix_prize/test_play_action.npy"

user_action_to_map(train_path, train_store_path)
user_action_to_map(test_path, test_store_path)

# 最终的用户行为map数据结构如下：
# {2097129: set([(3049, 2), (3701, 4), (3756, 3)]), 1048551: set([(3610, 4), (571, 3)])}

