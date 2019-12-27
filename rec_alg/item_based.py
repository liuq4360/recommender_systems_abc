#!/usr/bin/env python
# coding=utf-8

# 基于item-based协同过滤的思路来为用户进行推荐

import os
import numpy as np


rec_num = 30

cwd = os.getcwd()  # 获取当前工作目录
f_path = os.path.abspath(os.path.join(cwd, ".."))  # 获取上一级目录

history_f = f_path + "/output/history.npy"
similarity_f = f_path + "/output/similarity.npy"


history = np.load(history_f, allow_pickle=True).item()

similarity = np.load(similarity_f, allow_pickle=True).item()


rec_file = f_path + "/output/rec.txt"
rec_f = open(rec_file, 'w')  # 打开文件，如果文件不存在则创建它，该文件是存储最终的三元组

data = f_path + "/output/rec.txt"
fp = open(data, 'w')  # 打开文件，如果文件不存在则创建它，该文件是存储最终的三元组


for u, u_his in history.items():
    rec = dict()
    for vid in u_his:
        if vid in similarity:
            [score, vid_s] = similarity[vid]
            for i in range(len(vid_s)):
                if vid_s[i] in rec:
                    rec[vid_s[i]] = rec[vid_s[i]] + score[i]
                else:
                    rec[vid_s[i]] = score[i]
    if len(rec) >= rec_num:
        sorted_list = sorted(rec.items(), key=lambda item: item[1], reverse=True)
        res = sorted_list[:rec_num]
        tmp = "" + u + ","
        for j in range(len(res)-1):
            tmp = tmp + res[j][0] + ","
        tmp = tmp + res[len(res)-1][0] + "\n"
        fp.write(tmp)

fp.close()



