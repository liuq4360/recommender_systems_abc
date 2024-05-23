#!/usr/bin/env python
# coding=utf-8

# 利用 jaccard 相似性计算的物品相似度，为每个用户喜欢的种子物品（比如最近购买的3个）召回喜欢的相关物品。

import numpy as np


def seeds_recall(seeds, rec_num):
    """
    基于用户喜欢的种子物品，为用户召回关联物品。
    :param seeds: list，用户种子物品 ~ [item1,item2, ..., item_i]
    :param rec_num: 最终召回的物品数量
    :return: list ~ [(item1,score1),(item2,score2), ..., (item_k,score_k)]
    """

    jaccard_sim_rec_path = "../../output/netflix_prize/jaccard_sim_rec.npy"
    sim = np.load(jaccard_sim_rec_path, allow_pickle=True).item()
    recalls = []
    for seed in seeds:
        recalls.extend(sim[seed])
    # 可能不同召回的物品有重叠，那么针对重叠的，可以将score累加，然后根据score降序排列。
    tmp_dict = dict()
    for (i, s) in recalls:
        if i in tmp_dict:
            tmp_dict[i] = tmp_dict[i] + s
        else:
            tmp_dict[i] = s
    rec = sorted(tmp_dict.items(), key=lambda item: item[1], reverse=True)
    return rec[0:rec_num]

