#!/usr/bin/env python
# coding=utf-8

r"""
基于关联规则来实现推荐系统的召回，这里用的是fp-growth算法，利用pyfpgrowth包，参考如下：
https://fp-growth.readthedocs.io/en/latest/readme.html

关于fpgrowth算法的相关论文：

    .. [1] Haoyuan Li, Yi Wang, Dong Zhang, Ming Zhang, and Edward Y. Chang. 2008.
Pfp: parallel fp-growth for query recommendation.
    In Proceedings of the 2008 ACM conference on Recommender systems (RecSys '08).
Association for Computing Machinery, New York, NY, USA, 107-114.
DOI: https://doi.org/10.1145/1454008.1454027
    .. [2] Jiawei Han, Jian Pei, and Yiwen Yin. 2000.
Mining frequent patterns without candidate generation.
    SIGMOD Rec. 29, 2 (June 2000), 1-12.
DOI: https://doi.org/10.1145/335191.335372

"""

import os
import numpy as np
import pyfpgrowth
import pandas as pd

rec_num = 30

cwd = os.getcwd()  # 获取当前工作目录
f_path = os.path.abspath(os.path.join(cwd, "..", ".."))  # 获取上一级目录

play_action_f = f_path + "/output/netflix_prize/train_play_action.npy"
play_action = np.load(play_action_f, allow_pickle=True).item()  # {565: set([(3894, 3)]), 20: set([(3860, 4), (2095, 5)])}

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

vid_score_list = [list(x) for x in play_action.values()]  # [[(3860, 4), (2095, 5)], [(3894, 3)]]

transactions = [[k for (k, t) in x] for x in vid_score_list]  # [[3860, 2095], [3894]]
# transaction:[[1, 2, 5], [2, 4], [2, 3], [1, 2, 4], [1, 3], [2, 3], [1, 3], [1, 2, 3, 5], [1, 2, 3]]
min_support = 0.007
# float(input("Enter support %: "))
patterns = pyfpgrowth.find_frequent_patterns(transactions, int(len(transactions)*min_support))
# {(1, 2): 4, (1, 2, 3): 2, (1, 3): 4, (1,): 6, (2,): 7, (2, 4): 2, (1, 5): 2, (5,): 2, (2, 3): 4,
# (2, 5): 2, (4,): 2, (1, 2, 5): 2}
# for k, v in patterns.items():
#     print(k, v)
# print(len(patterns))

min_confidence = 0.05
rules = pyfpgrowth.generate_association_rules(patterns, min_confidence)
# {(2, 5): ((1,), 1.0), (1,): ((3,), 0.6666666666666666), (4,): ((2,), 1.0)}
print(rules)
print(len(rules))

# Get recommendations for a single user
# rules ~ # {(2, 5): ((1,), 1.0), (1,): ((3,), 0.6666666666666666), (4,): ((2,), 1.0)}
X = [x for x in rules.keys()] # X ~ [(1, 5), (5,), (2, 5), (4,)]
mf_rec_map = dict()
for uid in user:  # 所有用户id列表, eg. 3894
    video_score = play_action[uid]  # set([(708, 4), (2122, 4), (1744, 4), (2660, 4)])
    A = set([t[0] for t in video_score])
    video_score_map = dict(video_score)
    u_rec = dict()
    for x in X:
        if set(x).issubset(A):
            (Y, confidence) = rules[x]  # ((2,), 1.0)
            s = np.sum([video_score_map[t] for t in set(x)])*1.0/len(set(x))
            for y in set(Y):
                if y not in A:
                    if y in u_rec:
                        u_rec[y] = u_rec[y] + confidence * s
                    else:
                        u_rec[y] = confidence * s
                # 上面用到了confidence * s，主要是方便给最终的推荐结果进行排序，将 X => Y 这个关联规则
                # 中的X在用户行为中的评分先求平均，即s，然后乘以 X => Y 的置信度confidence
    if len(u_rec) >= rec_num:
        sorted_list = sorted(u_rec.items(), key=lambda item: item[1], reverse=True)
    res = sorted_list[:rec_num]
    mf_rec_map[uid] = res

print(mf_rec_map)

mf_rec_path = f_path + "/output/netflix_prize/association_rules_rec.npy"
np.save(mf_rec_path, mf_rec_map)
