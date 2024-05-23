#!/usr/bin/env python
# coding=utf-8

# 基于朴素贝叶斯来进行个性化推荐，采用书中第书中第8章的思路实现，具体实现细节参考书。

import os
import numpy as np
from scipy.sparse import dok_matrix

rec_num = 30

cwd = os.getcwd()  # 获取当前工作目录
f_path = os.path.abspath(os.path.join(cwd, "..", ".."))  # 获取上一级目录

play_action_f = f_path + "/output/netflix_prize/train_play_action.npy"
play_action = np.load(play_action_f, allow_pickle=True).item()

train = f_path + "/output/netflix_prize/train.txt"
f_train = open(train)  # 返回一个文件对象
user_s = set()
video_s = set()
# 获取所有的用户id和视频id
line = f_train.readline()  # 调用文件的 readline()方法
while line:
    d = line.split(",")
    user_id = int(d[0])
    video_id = int(d[1])
    user_s.add(user_id)
    video_s.add(video_id)
    line = f_train.readline()

f_train.close()
user = list(set(user_s))
video = list(set(video_s))


video_map = dict()
source = open(train, 'r')
line = source.readline()   # 调用文件的 readline()方法
while line:
    d = line.split(",")
    user_id = int(d[0])
    video_id = int(d[1])
    score = int(d[2])
    if video_id in video_map:
        s = video_map.get(video_id)
        s.add((user_id, score))
        video_map[video_id] = s
    else:
        s = set()
        s.add((user_id, score))
        video_map[video_id] = s
    line = source.readline()
source.close()


# Get recommendations for a single user
mf_rec_map = dict()
for uid in user:  # 所有用户id列表
    u_rec = []
    for vid in video:  # 所有视频id列表，需要计算每个用户对每个视频的评分
        uid_p_set = play_action[uid]  # {565: set([(3894, 3)]), 20: set([(3860, 4), (2095, 5)])}
        uid_action_vid = set([x[0] for x in list(uid_p_set)])
        if vid not in uid_action_vid:
            P_uid_vid = 0.0
            for s in [1, 2, 3, 4, 5]:  # Netflix prize数据集一共有5个评分等级，分别是1，2，3，4，5。
                temp = len([x[0] for x in list(video_map[vid]) if x[1] == s])
                if temp != 0:
                    f1 = 1.0*temp/len(video_map[vid])
                    f2 = 1.0
                    for (video_id, score) in uid_p_set:
                        f2 = f2 * len([x[0] for x in list(video_map[video_id]) if x[1] == score])/temp
                    P_uid_vid_s = f1 * f2 * s
                else:
                    P_uid_vid_s = 0.0
                P_uid_vid = max(P_uid_vid, P_uid_vid_s)
        u_rec.append((vid, P_uid_vid))
    if len(u_rec) >= rec_num:
        sorted_list = sorted(u_rec, key=lambda item: item[1], reverse=True)
    res = sorted_list[:rec_num]
    mf_rec_map[uid] = res

print(mf_rec_map)

mf_rec_path = f_path + "/output/netflix_prize/naive_bayes_rec.npy"
np.save(mf_rec_path, mf_rec_map)


