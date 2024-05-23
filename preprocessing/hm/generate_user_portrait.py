#!/usr/bin/env python
# coding=utf-8

# 基于用户行为数据，为每个用户生成用户画像。

import numpy as np
import pandas as pd


if __name__ == "__main__":
    trans = pd.read_csv("../../data/hm/transactions_train.csv")

    user_portrait = dict()

    article_dict = np.load("../../output/hm/article_dict.npy", allow_pickle=True).item()

    for _, row in trans.iterrows():
        customer_id = row['customer_id']
        article_id = row['article_id']

        feature_dict = article_dict[article_id]
        # article_dict[957375001]
        # {'product_code': 957375, 'product_type_no': 72,
        # 'graphical_appearance_no': 1010016, 'colour_group_code': 9,
        # 'perceived_colour_value_id': 4, 'perceived_colour_master_id': 5}

        product_code = feature_dict['product_code']
        product_type_no = feature_dict['product_type_no']
        graphical_appearance_no = feature_dict['graphical_appearance_no']
        colour_group_code = feature_dict['colour_group_code']
        perceived_colour_value_id = feature_dict['perceived_colour_value_id']
        perceived_colour_master_id = feature_dict['perceived_colour_master_id']

        if customer_id in user_portrait:

            portrait_dict = user_portrait[customer_id]
            # { 'product_code': set([108775, 116379])
            #   'product_type_no': set([253, 302, 304, 306])
            #   'graphical_appearance_no': set([1010016, 1010017])
            #   'colour_group_code': set([9, 11, 13])
            #   'perceived_colour_value_id': set([1, 3, 4, 2])
            #   'perceived_colour_master_id': set([11, 5 ,9])
            #   }

            if 'product_code' in portrait_dict:
                portrait_dict['product_code'].add(product_code)
            else:
                portrait_dict['product_code'] = set([product_code])

            if 'product_type_no' in portrait_dict:
                portrait_dict['product_type_no'].add(product_type_no)
            else:
                portrait_dict['product_type_no'] = set([product_type_no])

            if 'graphical_appearance_no' in portrait_dict:
                portrait_dict['graphical_appearance_no'].add(graphical_appearance_no)
            else:
                portrait_dict['graphical_appearance_no'] = set([graphical_appearance_no])

            if 'colour_group_code' in portrait_dict:
                portrait_dict['colour_group_code'].add(colour_group_code)
            else:
                portrait_dict['colour_group_code'] = set([colour_group_code])

            if 'perceived_colour_value_id' in portrait_dict:
                portrait_dict['perceived_colour_value_id'].add(perceived_colour_value_id)
            else:
                portrait_dict['perceived_colour_value_id'] = set([perceived_colour_value_id])

            if 'perceived_colour_master_id' in portrait_dict:
                portrait_dict['perceived_colour_master_id'].add(perceived_colour_master_id)
            else:
                portrait_dict['perceived_colour_master_id'] = set([perceived_colour_master_id])

            user_portrait[customer_id] = portrait_dict

        else:
            portrait_dict = dict()
            portrait_dict['product_code'] = set([product_code])
            portrait_dict['product_type_no'] = set([product_type_no])
            portrait_dict['graphical_appearance_no'] = set([graphical_appearance_no])
            portrait_dict['colour_group_code'] = set([colour_group_code])
            portrait_dict['perceived_colour_value_id'] = set([perceived_colour_value_id])
            portrait_dict['perceived_colour_master_id'] = set([perceived_colour_master_id])

            user_portrait[customer_id] = portrait_dict

    np.save("../../output/hm/user_portrait.npy", user_portrait)
