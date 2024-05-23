#!/usr/bin/env python
# coding=utf-8

# 基于 jaccard 相似性计算物品相似度，为每个物品计算最相似的N个物品。

import numpy as np
import pandas as pd


def jaccard_similarity(set_1, set_2):
    """
    计算两个集合的jaccard相似度。jaccard_similarity = || set_1 & set_2 || / || set_1 | set_2 ||
    :param set_1: 集合1
    :param set_2: 集合2
    :return: 相似度
    """
    return len(set_1 & set_2)*1.0/len(set_1 | set_2)


def article_jaccard_similarity(article_1, article_2):
    """
    计算两个article的jaccard相似性。
    :param article_1: 物品1的metadata，数据结构是一个dict，基于articles.csv的行构建的。
    :param article_2: 物品2的metadata，数据结构是一个dict，基于articles.csv的行构建的。
    :return: sim，返回 article_1 和 article_2 的相似性。
    """
    sim = 0.0
    for key in article_1.keys():
        sim = sim + jaccard_similarity(set(article_1[key]), set(article_2[key]))
    return sim/len(article_1)


if __name__ == "__main__":
    art = pd.read_csv("../../data/hm/articles.csv")
    # art = art[['prod_name', 'product_type_name', 'product_group_name', 'graphical_appearance_name',
    # 'colour_group_name', 'perceived_colour_value_name', 'perceived_colour_master_name', 'department_name',
    # 'index_name', 'index_group_name', 'section_name', 'garment_group_name']]
    # art_1 = dict(art.loc[0])  # 取第一行
    # art_2 = dict(art.loc[3])  # 取第二行
    # print(article_jaccard_similarity(art_1, art_2))

    jaccard_sim_rec_map = dict()
    rec_num = 30
    articles = art.iloc[:, 0].drop_duplicates().to_list()  # 取第一列的值，然后去重，转为list
    for a in articles:
        row_a = art[art['article_id'] == a]  # 取 art中 'article_id' 列值为 a 的行
        tmp_a = row_a[['prod_name', 'product_type_name', 'product_group_name', 'graphical_appearance_name',
                       'colour_group_name', 'perceived_colour_value_name', 'perceived_colour_master_name',
                       'department_name', 'index_name', 'index_group_name', 'section_name', 'garment_group_name']]
        art_a = dict(tmp_a.loc[tmp_a.index[0]])
        sim_dict = dict()
        for b in articles:
            if a != b:
                row_b = art[art['article_id'] == b]
                tmp_b = row_b[['prod_name', 'product_type_name', 'product_group_name', 'graphical_appearance_name',
                               'colour_group_name', 'perceived_colour_value_name', 'perceived_colour_master_name',
                               'department_name', 'index_name', 'index_group_name', 'section_name', 'garment_group_name'
                               ]]
                art_b = dict(tmp_b.loc[tmp_b.index[0]])
                sim_ = article_jaccard_similarity(art_a, art_b)
                sim_dict[b] = sim_
        sorted_list = sorted(sim_dict.items(), key=lambda item: item[1], reverse=True)
        res = sorted_list[:rec_num]
        jaccard_sim_rec_map[a] = res

    jaccard_sim_rec_path = "../../output/netflix_prize/jaccard_sim_rec.npy"
    np.save(jaccard_sim_rec_path, jaccard_sim_rec_map)
