#!/usr/bin/env python
# coding=utf-8

# 计算推荐系统的precision和recall指标

import os
import numpy as np


def precision(rec_list, play_list):
    """
    为某个用户计算准确率
    :param rec_list: 给该用户的推荐列表，  类型 <type 'numpy.ndarray'>
    :param play_list: 该用户的真实播放列表， 类型 <type 'numpy.ndarray'>
    :return: 准确率
    """
    inter = set(rec_list).intersection(set(play_list))
    return float(len(inter))/len(rec_list)


def recall(rec_list, play_list):
    """
    为某个用户计算召回率
    :param rec_list: 给该用户的推荐列表，  类型 <type 'numpy.ndarray'>
    :param play_list: 该用户的真实播放列表， 类型 <type 'numpy.ndarray'>
    :return: 召回率
    """
    inter = set(rec_list).intersection(set(play_list))
    return float(len(inter))/len(play_list)


cwd = os.getcwd()  # 获取当前工作目录
f_path = os.path.abspath(os.path.join(cwd, ".."))  # 获取上一级目录

play_f = f_path + "/output/play.npy"
rec_f = f_path + "/output/rec.txt"

play_map = np.load(play_f, allow_pickle=True).item()
rec_f = open(f_path + '/output/text.txt', 'r')


recall_accumulate = 0
precision_accumulate = 0
rec_u_set = set()

print("===================开始为计算总体召回&排序==================")
i = 0
line = rec_f.readline()   # 调用文件的 readline()方法
while line:
    if i % 100 == 0:
        print(i)
        i = i + 1
    d = line.split(",")
    user_id = int(d[0])
    rec_u_set.add(user_id)
    if user_id in play_map:
        play_u = play_map[user_id]
        rec_u = d[1:]
        precision_u = precision(rec_u, play_u)
        recall_u = recall(rec_u, play_u)
        precision_accumulate = precision_accumulate + precision_u
        recall_accumulate = recall_accumulate + rec_u
    line = rec_f.readline()

rec_f.close()

print("===================计算完总体召回&排序==================")


user_num = len(rec_u_set.intersection((set(play_map.keys()))))

precision = precision_accumulate/user_num
recall = recall_accumulate/user_num

print("precision=" + precision)
print("recall=" + recall)

