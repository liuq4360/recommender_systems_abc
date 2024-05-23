#!/usr/bin/env python
# coding=utf-8

r"""
多个召回随机打散排序。

"""

import os
import numpy as np
import random


def shuffle_ranking(recall_list, n):
    """
    多个召回随机打散排序。
    :param recall_list: [recall_1, recall_2, ..., recall_k].
    :param n:推荐数量
    每个recall的数据结构是 recall_i ~ [(v1,s1),(v2,s2),...,(v_t,s_t)]
    :return:
    """
    recommend = []
    recall_num = len(recall_list)
    for i in range(recall_num):
        recommend.extend([x for x in recall_list[i]])
    random.shuffle(recommend)
    return recommend[0:n]


if __name__ == "__main__":
    rec = shuffle_ranking([[("x", 0.6), ("y", 0.56), ("z", 0.48)], [("x", 0.1), ("q", 0.36), ("t", 0.38)]], 5)
    print(rec)
