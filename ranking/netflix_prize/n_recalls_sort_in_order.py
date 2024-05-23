#!/usr/bin/env python
# coding=utf-8

r"""
按照召回的指定顺序排序。

"""

import os
import numpy as np


def order_ranking(recall_list, n):
    """
    按照召回的指定顺序排序，这里recall_list的顺序就是我们指定的顺序。
    这里没有考虑多个召回的recall中包含重复的情况，一般可以先去重然后调用该排序，或者先多取一些，然后对
    最终的推荐结果去重。
    :param recall_list: [recall_1, recall_2, ..., recall_k].
    :param n: 推荐的物品的个数
    每个recall的数据结构是 recall_i ~ [(v1,s1),(v2,s2),...,(v_t,s_t)]
    :return: recommend
    """
    recommend = []
    recall_num = len(recall_list)
    for i in range(n):
        t = i % recall_num  # 求余数，这里t就是这一次需要从哪个recall中获取推荐结果
        s = int(i/recall_num)  # 求被除数，这里说明的是第几轮，确定从recall中取第几个元素
        recommend.append(recall_list[t][s])
    return recommend


if __name__ == "__main__":
    rec = order_ranking([[("x", 0.6), ("y", 0.56), ("z", 0.48)], [("x", 0.1), ("q", 0.36), ("t", 0.38)]], 5)
    print(rec)
