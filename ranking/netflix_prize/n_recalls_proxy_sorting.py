#!/usr/bin/env python
# coding=utf-8

r"""
基于代理函数f(x)来为各个召回的结果进行排序。

"""

import os
import numpy as np


def movie_year_map(path="/Users/gongyouliu/Desktop/olddata/code/github/recommender_systems_abc/data/netflix_prize/movie_titles.txt"):
    """
    获取每个电影的年代，基于年代来作为计算代理函数的一个参数
    :param path: 存储电影基础信息的目录。
    :return: dic ~ {movie_id:year,...}
    """
    f_data = open(path)  # 返回一个文件对象
    year_dict = dict()
    line = f_data.readline()   # 调用文件的 readline()方法
    while line:
        tmp = line.strip('\n').split(',')
        vid = int(tmp[0])
        year = 1900
        if tmp[1] != "NULL":
            year = int(tmp[1])
        year_dict[vid] = year
        line = f_data.readline()
    f_data.close()
    return year_dict


def avg_movie_score(path="/Users/gongyouliu/Desktop/olddata/code/github/recommender_systems_abc/output/netflix_prize/train.txt"):
    """
    计算每个电影的平均评分及总播放次数，平均评分越多，播放次数越大，说明越受欢迎。
    :param path: 存储用户行为数据的路径
    :return: dict ~ {movie_id:(avg_score, total_plays),...}
    """
    f_data = open(path)  # 返回一个文件对象
    avg__total_dict = dict()
    line = f_data.readline()   # 调用文件的 readline()方法
    while line:
        tmp = line.strip('\n').split(',')
        vid = int(tmp[1])
        score = int(tmp[2])
        if vid in avg__total_dict:
            (total_score, times) = avg__total_dict[vid]
            avg__total_dict[vid] = (total_score + score, times + 1)
        else:
            avg__total_dict[vid] = (score, 1)
        line = f_data.readline()
    f_data.close()
    for key, value in avg__total_dict.items():
        avg__total_dict[key] = value[0]/value[1], value[1]
    return avg__total_dict


def proxy_function(t):
    """
    代理函数实现。
    :param t: t ~ (vid, score)。代理函数需要基于视频本身的信息获得该视频的代理函数得分。
    :return:
    """
    vid = t[0]
    # score = t[1]
    year_dict = movie_year_map()  # 计算所有电影的年代，dic ~ {movie_id:year,...}
    # 下面是计算所有电影平均得分及总次数，dict ~ {movie_id:(avg_score, total_plays),...}
    avg__total_dict = avg_movie_score()
    year = year_dict[vid] # 电影的年代
    avg_score, times = avg__total_dict[vid]  # 电影的平均评分及总播放次数
    f = (year*1.0/2023) * avg_score * np.log(times)
    return vid, f


def proxy_ranking(recall_list, n):
    """
    代理函数实现。
    :param recall_list: [recall_1, recall_2, ..., recall_k].
    :param n:推荐数量
    每个recall的数据结构是 recall_i ~ [(v1,s1),(v2,s2),...,(v_t,s_t)]
    :return:
    """
    recommend = []
    recall_num = len(recall_list)
    for i in range(recall_num):
        recall_i = recall_list[i]
        recommend.extend([proxy_function(s) for s in recall_i])
    sorted_list = sorted(recommend, key=lambda item: item[1], reverse=True)
    return sorted_list[:n]


if __name__ == "__main__":
    # print(movie_year_map())
    print(avg_movie_score())
    rec = proxy_ranking([[("x", 0.6), ("y", 0.56), ("z", 0.48)], [("x", 0.1), ("q", 0.36), ("t", 0.38)]], 3)
    print(rec)
