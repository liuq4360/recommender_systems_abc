#!/usr/bin/env python
# coding=utf-8

# 为每个物品生成对应的特征。


import numpy as np
import pandas as pd


if __name__ == "__main__":
    # 为每个物品生成对应的特征，这里我们只用到了product_code、product_type_no、graphical_appearance_no、
    # colour_group_code、perceived_colour_value_id、perceived_colour_master_id这6个特征。
    art = pd.read_csv("../../data/hm/articles.csv")

    article_dict = dict()  # {12:{id1,id2,...,id_k}, 34:{id1,id2,...,id_k}}, 这里面每个物品对应的特征权重都一样

    for _, row in art.iterrows():
        article_id = row['article_id']

        product_code = row['product_code']
        product_type_no = row['product_type_no']
        graphical_appearance_no = row['graphical_appearance_no']
        colour_group_code = row['colour_group_code']
        perceived_colour_value_id = row['perceived_colour_value_id']
        perceived_colour_master_id = row['perceived_colour_master_id']

        feature_dict = dict()
        feature_dict['product_code'] = product_code
        feature_dict['product_type_no'] = product_type_no
        feature_dict['graphical_appearance_no'] = graphical_appearance_no
        feature_dict['colour_group_code'] = colour_group_code
        feature_dict['perceived_colour_value_id'] = perceived_colour_value_id
        feature_dict['perceived_colour_master_id'] = perceived_colour_master_id

        article_dict[article_id] = feature_dict

    # print(article_dict)
    np.save("../../output/hm/article_dict.npy", article_dict)

