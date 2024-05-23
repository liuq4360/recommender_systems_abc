#!/usr/bin/env python
# coding=utf-8

# 基于物品的特征，为每个特征生成对应的倒排索引，倒排索引存到Redis中。
# 需要生成倒排索引的特征包括如下几个：

# product_code, 产品code，7位数字字符，如 0108775，是 article_id 的前 7 位。
# prod_name, 产品名，如 Strap top（系带上衣）

# product_type_no, 产品类型no，2位或者3位数字，有 -1 值。
# product_type_name, 产品类型名。如 Vest top（背心）

# graphical_appearance_no, 图案外观no，如 1010016。
# graphical_appearance_name, 图案外观名，如 Solid（固体;立体图形）

# colour_group_code, 颜色组code，如 09，2位数字
# colour_group_name, 颜色组名称， 如 Black。

# perceived_colour_value_id, 感知颜色值id。 -1，1，2，3，4，5，6，7，一共这几个值。
# perceived_colour_value_name, 感知颜色值名称。如 Dark（黑暗的），Dusty Light等

# perceived_colour_master_id, 感知颜色主id。1位或者2位数字。
# perceived_colour_master_name, 感知颜色主名称。 如 Beige（浅褐色的）


import numpy as np
import pandas as pd


if __name__ == "__main__":
    art = pd.read_csv("../../data/hm/articles.csv")

    product_code_unique = np.unique(art[["product_code"]])  # 取某一列的所有唯一值，array([108775, 111565, ..., 959461])
    product_type_no_unique = np.unique(art[["product_type_no"]])
    graphical_appearance_no_unique = np.unique(art[["graphical_appearance_no"]])
    colour_group_code_unique = np.unique(art[["colour_group_code"]])
    perceived_colour_value_id_unique = np.unique(art[["perceived_colour_value_id"]])
    perceived_colour_master_id_unique = np.unique(art[["perceived_colour_master_id"]])

    product_code_portrait_dict = dict()  # {12:{id1,id2,...,id_k}, 34:{id1,id2,...,id_k}}, 这里面每个物品对应的特征权重都一样
    product_type_no_portrait_dict = dict()
    graphical_appearance_no_portrait_dict = dict()
    colour_group_code_portrait_dict = dict()
    perceived_colour_value_id_portrait_dict = dict()
    perceived_colour_master_id_portrait_dict = dict()

    for _, row in art.iterrows():
        article_id = row['article_id']

        product_code = row['product_code']
        product_type_no = row['product_type_no']
        graphical_appearance_no = row['graphical_appearance_no']
        colour_group_code = row['colour_group_code']
        perceived_colour_value_id = row['perceived_colour_value_id']
        perceived_colour_master_id = row['perceived_colour_master_id']

        if product_code in product_code_portrait_dict:
            product_code_portrait_dict[product_code].add(article_id)
        else:
            product_code_portrait_dict[product_code] = set([article_id])

        if product_type_no in product_type_no_portrait_dict:
            product_type_no_portrait_dict[product_type_no].add(article_id)
        else:
            product_type_no_portrait_dict[product_type_no] = set([article_id])

        if graphical_appearance_no in graphical_appearance_no_portrait_dict:
            graphical_appearance_no_portrait_dict[graphical_appearance_no].add(article_id)
        else:
            graphical_appearance_no_portrait_dict[graphical_appearance_no] = set([article_id])

        if colour_group_code in colour_group_code_portrait_dict:
            colour_group_code_portrait_dict[colour_group_code].add(article_id)
        else:
            colour_group_code_portrait_dict[colour_group_code] = set([article_id])

        if perceived_colour_value_id in perceived_colour_value_id_portrait_dict:
            perceived_colour_value_id_portrait_dict[perceived_colour_value_id].add(article_id)
        else:
            perceived_colour_value_id_portrait_dict[perceived_colour_value_id] = set([article_id])

        if perceived_colour_master_id in perceived_colour_master_id_portrait_dict:
            perceived_colour_master_id_portrait_dict[perceived_colour_master_id].add(article_id)
        else:
            perceived_colour_master_id_portrait_dict[perceived_colour_master_id] = set([article_id])

    # print(product_code_portrait_dict)
    # print(product_type_no_portrait_dict)
    # print(graphical_appearance_no_portrait_dict)
    # print(colour_group_code_portrait_dict)
    # print(perceived_colour_value_id_portrait_dict)
    # print(perceived_colour_master_id_portrait_dict)

    np.save("../../output/hm/product_code_portrait_dict.npy", product_code_portrait_dict)
    np.save("../../output/hm/product_type_no_portrait_dict.npy", product_type_no_portrait_dict)
    np.save("../../output/hm/graphical_appearance_no_portrait_dict.npy", graphical_appearance_no_portrait_dict)
    np.save("../../output/hm/colour_group_code_portrait_dict.npy", colour_group_code_portrait_dict)
    np.save("../../output/hm/perceived_colour_value_id_portrait_dict.npy", perceived_colour_value_id_portrait_dict)
    np.save("../../output/hm/perceived_colour_master_id_portrait_dict.npy", perceived_colour_master_id_portrait_dict)
