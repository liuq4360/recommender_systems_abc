#!/usr/bin/env python
# coding=utf-8

# 基于item-based协同过滤的思路来为用户进行推荐，基于用户的协同过滤实现思路是类似的，本工程不再重复实现。

import os
import numpy as np


rec_num = 30

cwd = os.getcwd()  # 获取当前工作目录
f_path = os.path.abspath(os.path.join(cwd, "..", ".."))  # 获取上一级目录

play_action_f = f_path + "/output/netflix_prize/train_play_action.npy"
similarity_f = f_path + "/output/netflix_prize/similarity_rec.npy"


play_action = np.load(play_action_f, allow_pickle=True).item()

similarity = np.load(similarity_f, allow_pickle=True).item()


# play_action: {2097129: set([(3701, 4), (3756, 3)]), 1048551: set([(3610, 4), (571, 3)])}
# similarity：{2345: [(1905, 0.5), (2452, 0.3), (3938, 0.1)]}
item_based_rec_map = dict()
for u, u_play in play_action.items():
    u_rec = dict()
    for (vid, u_score) in u_play:
        if vid in similarity:
            for (vid_s, vid_score) in similarity[vid]:
                if vid_s in u_rec:
                    u_rec[vid_s] = u_rec[vid_s] + u_score*vid_score
                else:
                    u_rec[vid_s] = u_score*vid_score
    if len(u_rec) >= rec_num:
        sorted_list = sorted(u_rec.items(), key=lambda item: item[1], reverse=True)
        res = sorted_list[:rec_num]
        item_based_rec_map[u] = res

print(item_based_rec_map)

item_based_rec_path = f_path + "/output/netflix_prize/item_based_rec.npy"
np.save(item_based_rec_path, item_based_rec_map)

# item-based推荐的数据结构如下：
# {u1: [(1905, 0.5), (2452, 0.3), (3938, 0.1)]}



