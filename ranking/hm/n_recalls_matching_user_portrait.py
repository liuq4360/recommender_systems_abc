#!/usr/bin/env python
# coding=utf-8

r"""
基于用户兴趣画像，利用召回物品跟用户画像的匹配度来进行排序。

"""

import numpy as np

article_dict = np.load("../../output/hm/article_dict.npy", allow_pickle=True).item()
user_portrait = np.load("../../output/hm/user_portrait.npy", allow_pickle=True).item()


def user_portrait_similarity(portrait, article_id):
    """
    计算某个article与用户画像的相似度。
    :param portrait: 用户画像。
        { 'product_code': set([108775, 116379])
          'product_type_no': set([253, 302, 304, 306])
          'graphical_appearance_no': set([1010016, 1010017])
          'colour_group_code': set([9, 11, 13])
          'perceived_colour_value_id': set([1, 3, 4, 2])
          'perceived_colour_master_id': set([11, 5 ,9])
        }
    :param article_id: 物品id。
    :return: sim，double，相似度。
    """
    feature_dict = article_dict[article_id]
    # article_dict[957375001]
    # {'product_code': 957375, 'product_type_no': 72,
    # 'graphical_appearance_no': 1010016, 'colour_group_code': 9,
    # 'perceived_colour_value_id': 4, 'perceived_colour_master_id': 5}

    sim = 0.0
    features = {'product_code', 'product_type_no', 'graphical_appearance_no',
                'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id'}
    for fea in features:
        fea_value = feature_dict[fea]
        if fea_value in portrait[fea]:
            sim = sim + 1.0  # 只要用户的某个画像特征中包含这个物品的该画像值，那么就认为跟用户的兴趣匹配
    return sim/6


def user_portrait_ranking(portrait, recall_list, n):
    """
    利用用户画像匹配度进行排序。
    :param portrait: 用户画像。
        { 'product_code': set([108775, 116379])
          'product_type_no': set([253, 302, 304, 306])
          'graphical_appearance_no': set([1010016, 1010017])
          'colour_group_code': set([9, 11, 13])
          'perceived_colour_value_id': set([1, 3, 4, 2])
          'perceived_colour_master_id': set([11, 5 ,9])
        }
    :param recall_list: [recall_1, recall_2, ..., recall_k].
        每个recall的数据结构是 recall_i ~ [(v1,s1),(v2,s2),...,(v_t,s_t)]
    :param n: 推荐数量
    :return: rec ~ [(v1,s1),(v2,s2),...,(v_t,s_t)]
    """
    rec_dict = dict()
    for recall in recall_list:
        for (article_id, _) in recall:
            sim = user_portrait_similarity(portrait, article_id)
            if article_id in rec_dict:
                rec_dict[article_id] = rec_dict[article_id] + sim
                # 如果多个召回列表，召回了相同的物品，那么相似性相加。
            else:
                rec_dict[article_id] = sim
    rec = sorted(rec_dict.items(), key=lambda item: item[1], reverse=True)
    return rec[0:n]


if __name__ == "__main__":

    rec_num = 5

    customer = "00083cda041544b2fbb0e0d2905ad17da7cf1007526fb4c73235dccbbc132280"
    customer_portrait = user_portrait[customer]

    recall_1 = [(111586001, 0.45), (112679048, 0.64), (158340001, 0.26)]
    recall_2 = [(176550016, 0.13), (189616038, 0.34), (212629035, 0.66)]
    recall_3 = [(233091021, 0.49), (244853032, 0.24), (265069020, 0.71)]

    recalls = [recall_1, recall_2, recall_3]
    rec = user_portrait_ranking(customer_portrait, recalls, rec_num)
    print(rec)
