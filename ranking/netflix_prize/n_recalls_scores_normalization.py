#!/usr/bin/env python
# coding=utf-8

r"""
基于多个召回的结果进行归一化，获得最终的排序。

"""

import os
import numpy as np


def normalization(recall):
    """
    对recall进行归一化，采用正态分布归一化。
    :param recall: 类似 [(v1,s1),(v2,s2),...,(v_t,s_t)]这样的数据结构
    :return: norm_list
    """
    score_list = [s[1] for s in recall]
    mean = np.mean(score_list)
    std = np.std(score_list)
    return [(s[0], (s[1]-mean)/std) for s in recall]


def normalization_ranking(recall_list, n):
    """
    归一化函数实现。
    :param recall_list: [recall_1, recall_2, ..., recall_k].
    :param n:推荐数量
    每个recall的数据结构是 recall_i ~ [(v1,s1),(v2,s2),...,(v_t,s_t)]
    :return:
    """
    recommend = []
    recall_num = len(recall_list)
    for i in range(recall_num):
        recommend.extend(normalization(recall_list[i]))
    sorted_list = sorted(recommend, key=lambda item: item[1], reverse=True)
    return sorted_list[:n]


if __name__ == "__main__":
    print(normalization([("x", 0.6), ("y", 0.56), ("z", 0.48)]))
    rec = normalization_ranking([[("x", 0.6), ("y", 0.56), ("z", 0.48)], [("x", 0.1), ("q", 0.36), ("t", 0.38)]], 3)
    print(rec)
