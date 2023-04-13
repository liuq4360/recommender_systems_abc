#!/usr/bin/env python
# coding=utf-8

# 将以dict数据结构形式生成的推荐结果存放到Redis中供web接口访问

import os
import numpy as np
import redis
import json


def save_rec(path, db):
    """
    将某个算法的推荐结果存放到Redis中
        # 最终的推荐结果的数据结构如下：
        # {"hot": [(1905, 564361), (2452, 462715), (3938, 448276)]}
    :param path: 推荐结果存放的路径
    :param db: 推荐结果存放的db
    :return: null
    """
    rec = np.load(path, allow_pickle=True).item()
    r = redis.Redis(host='localhost', port=6379, db=db)
    rec_dict = np.load(path, allow_pickle=True).item()
    for key, value in rec_dict.items():
        r.zadd(key, dict(value))
    r.close()


def save_metadata(path, db):
    """
    将电影的metadata存放到Redis中
        # metadata_map 的数据结构如下：
        # {1781:(2004,Noi the Albino), 1790:(1966,Born Free)}
    :param path: 电影metadata存放的路径
    :param db: 电影metadata存放的db
    :return: null
    """
    rec = np.load(path, allow_pickle=True).item()
    r = redis.Redis(host='localhost', port=6379, db=db)
    metadata_dict = np.load(path, allow_pickle=True).item()
    for key, value in metadata_dict.items():
        (year, title) = value
        print(year)
        print(title)
        j = json.dumps({"year": year, "title": title})
        # "{\"title\": \"DJ Shadow: In Tune and On Time\", \"year\": \"2004\"}"
        r.hset("metadata", key, j)
    r.close()


# 将热门推荐、相似推荐、item-based个性化推荐结果存于Redis中
cwd = os.getcwd()  # 获取当前工作目录
f_path = os.path.abspath(os.path.join(cwd, "../save2db"))  # 获取上一级目录

hot_rec_p = f_path + "/output/hot_rec.npy"
hot_rec_db = 0
save_rec(hot_rec_p, hot_rec_db)

similarity_rec_p = f_path + "/output/similarity_rec.npy"
similarity_rec_db = 1
save_rec(similarity_rec_p, similarity_rec_db)

item_based_rec_p = f_path + "/output/item_based_rec.npy"
item_based_rec_db = 2
save_rec(item_based_rec_p, item_based_rec_db)


metadata_p = f_path + "/output/movie_metadata.npy"
metadata_db = 3
save_metadata(metadata_p, metadata_db)
