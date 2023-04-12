#!/usr/bin/env python
# coding=utf-8

# 将电影的metadata转化为合适的格式存储

import os
import numpy as np

cwd = os.getcwd()  # 获取当前工作目录
f_path = os.path.abspath(os.path.join(cwd, ".."))

data = f_path + "/data/movie_titles.txt"

f_data = open(data)        # 返回一个文件对象

metadata_map = dict()

line = f_data.readline()   # 调用文件的 readline()方法
while line:
    tmp = line.strip('\n').split(',')
    vid = int(tmp[0])
    year = tmp[1]
    title = tmp[2]
    metadata_map[vid] = (year, title)
    line = f_data.readline()

f_data.close()

store_path = f_path + "/output/movie_metadata.npy"
np.save(store_path, metadata_map)

print(metadata_map)
print(len(metadata_map))

# metadata_map 的数据结构如下：
# {1781:(2004,Noi the Albino), 1790:(1966,Born Free)}
