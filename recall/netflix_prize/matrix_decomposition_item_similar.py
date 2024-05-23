#!/usr/bin/env python
# coding=utf-8

# 利用implicit库来进行ALS矩阵分解实现。有关implicit介绍，请参考：https://github.com/benfred/implicit。
# 利用矩阵分解获得的物品特征向量来计算物品之间的相似度

import os
import numpy as np
from scipy.sparse import dok_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight

rec_num = 30

cwd = os.getcwd()  # 获取当前工作目录
f_path = os.path.abspath(os.path.join(cwd, "..", ".."))  # 获取上一级目录

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

user.sort()
video.sort()

user_num = len(user)
video_num = len(video)

print("===================开始构建用户id索引==================")
uid2idx_map = dict()  # 获得用户id到index的映射关系，index从0开始编码
idx2uid_map = dict()  # 获得index到用户id的映射关系，index从0开始编码
index = 0
for uid in user:
    uid2idx_map[uid] = index
    idx2uid_map[index] = uid
    index = index + 1

print("===================开始构建视频id索引==================")
vid2idx_map = dict()  # 获得视频id到index的映射关系，index从0开始编码
idx2vid_map = dict()  # 获得index到视频id的映射关系，index从0开始编码
index = 0
for vid in video:
    vid2idx_map[vid] = index
    idx2vid_map[index] = vid
    index = index + 1

print("===================开始构建用户行为矩阵==================")
# 构建用户行为矩阵
Mat = dok_matrix((video_num, user_num), dtype=np.float32)  # 行是视频、列是用户

f_train = open(train)  # 返回一个文件对象
line = f_train.readline()  # 调用文件的 readline()方法
i = 0
while line:
    if i % 100 == 0:
        print(i)
    i = i + 1
    d = line.split(",")
    user_id = int(d[0])
    video_id = int(d[1])
    score = int(d[2])
    u_idx = uid2idx_map[user_id]
    v_idx = vid2idx_map[video_id]
    Mat[v_idx, u_idx] = score
    line = f_train.readline()

f_train.close()

print("===================完成构建用户行为矩阵==================")
# print(Mat.shape)

plays = Mat.tocsr()  # Compressed Sparse Row matrix
# Mat_csr_trans = Mat_csr.transpose()  # Compressed Sparse Row matrix

print("===================plays.shape==================")
print(plays.shape)

plays = bm25_weight(plays, K1=100, B=0.8)

plays = plays.tocsr()

model = AlternatingLeastSquares(factors=64, regularization=0.05)
model.approximate_recommend = False
model.fit(plays)
# Get similar recommend for one movie
mf_sim_map = dict()
# similarity：{2345: [(1905, 0.5), (2452, 0.3), (3938, 0.1)]}
user_plays = plays.T.tocsr()
for vid in video:
    idx = vid2idx_map[vid]
    sim = model.similar_items(idx, N=rec_num)
    sim_rec = [(idx2vid_map[x[0]], x[1]) for x in sim]
    mf_sim_map[vid] = sim_rec

print(mf_sim_map)

mf_sim_path = f_path + "/output/netflix_prize/mf_sim.npy"
np.save(mf_sim_path, mf_sim_map)

