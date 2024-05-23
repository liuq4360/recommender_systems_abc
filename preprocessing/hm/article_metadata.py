#!/usr/bin/env python
# coding=utf-8

# 为每个物品生成对应的特征。


import numpy as np
import pandas as pd


if __name__ == "__main__":
    art = pd.read_csv("../../data/hm/articles.csv")

    meta_dict = dict()

    for _, row in art.iterrows():
        article_id = row['article_id']
        prod_name = row['prod_name']
        product_type_name = row['product_type_name']
        graphical_appearance_name = row['graphical_appearance_name']
        colour_group_name = row['colour_group_name']
        perceived_colour_value_name = row['perceived_colour_value_name']
        perceived_colour_master_name = row['perceived_colour_master_name']
        department_name = row['department_name']
        detail_desc = row['detail_desc']
        info_dict = dict()
        info_dict['prod_name'] = prod_name
        info_dict['product_type_name'] = product_type_name
        info_dict['graphical_appearance_name'] = graphical_appearance_name
        info_dict['colour_group_name'] = colour_group_name
        info_dict['perceived_colour_value_name'] = perceived_colour_value_name
        info_dict['perceived_colour_master_name'] = perceived_colour_master_name
        info_dict['department_name'] = department_name
        info_dict['detail_desc'] = detail_desc
        meta_dict[article_id] = info_dict

    # print(article_dict)
    np.save("../../output/hm/article_metadata_dict.npy", meta_dict)

