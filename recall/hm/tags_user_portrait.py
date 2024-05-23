#!/usr/bin/env python
# coding=utf-8

# 基于用户画像，为该用户生成基于画像的个性化召回。

import numpy as np
import random


if __name__ == "__main__":

    rec_num = 30

    user_portrait = np.load("../../output/hm/user_portrait.npy", allow_pickle=True).item()

    product_code_portrait_dict = np.load("../../output/hm/product_code_portrait_dict.npy", allow_pickle=True).item()
    product_type_no_portrait_dict = np.load("../../output/hm/product_type_no_portrait_dict.npy",
                                            allow_pickle=True).item()
    graphical_appearance_no_portrait_dict = np.load("../../output/hm/graphical_appearance_no_portrait_dict.npy",
                                                    allow_pickle=True).item()
    colour_group_code_portrait_dict = np.load("../../output/hm/colour_group_code_portrait_dict.npy",
                                              allow_pickle=True).item()
    perceived_colour_value_id_portrait_dict = np.load("../../output/hm/perceived_colour_value_id_portrait_dict.npy",
                                                      allow_pickle=True).item()
    perceived_colour_master_id_portrait_dict = np.load("../../output/hm/perceived_colour_master_id_portrait_dict.npy",
                                                       allow_pickle=True).item()
    # {12:{id1,id2,...,id_k}, 34:{id1,id2,...,id_k}}, 这里面每个物品对应的特征权重都一样

    customer_rec = dict()

    for customer in user_portrait.keys():

        portrait_dict = user_portrait[customer]
        # { 'product_code': set([108775, 116379])
        #   'product_type_no': set([253, 302, 304, 306])
        #   'graphical_appearance_no': set([1010016, 1010017])
        #   'colour_group_code': set([9, 11, 13])
        #   'perceived_colour_value_id': set([1, 3, 4, 2])
        #   'perceived_colour_master_id': set([11, 5 ,9])
        #   }

        product_code_rec = set()
        product_type_no_rec = set()
        graphical_appearance_no_rec = set()
        colour_group_code_rec = set()
        perceived_colour_value_id_rec = set()
        perceived_colour_master_id_rec = set()

        rec = []

        # 针对6类特征画像类型，用户在某个类型中都可能有兴趣点，针对每个兴趣点获得对应的物品id，将同一个画像类型
        # 中所有的兴趣点的物品推荐聚合到一起，最后对该兴趣画像类型，只取 rec_num 个推荐。
        # 最后，对6个兴趣画像类型的推荐，最终合并在一起，只取 rec_num 个作为最终的推荐。
        if 'product_code' in portrait_dict:
            product_code_set = portrait_dict['product_code']
            for product_code in product_code_set:
                product_code_rec = product_code_rec | product_code_portrait_dict[product_code]
            rec = rec.append(random.sample(product_code_rec, rec_num))

        if 'product_type_no' in portrait_dict:
            product_type_no_set = portrait_dict['product_type_no']
            for product_type_no in product_type_no_set:
                product_type_no_rec = product_type_no_rec | product_type_no_portrait_dict[product_type_no]
            rec = rec.append(random.sample(product_type_no_rec, rec_num))

        if 'graphical_appearance_no' in portrait_dict:
            graphical_appearance_no_set = portrait_dict['graphical_appearance_no']
            for graphical_appearance_no in graphical_appearance_no_set:
                graphical_appearance_no_rec = graphical_appearance_no_rec | graphical_appearance_no_portrait_dict[graphical_appearance_no]
            rec = rec.append(random.sample(graphical_appearance_no_rec, rec_num))

        if 'colour_group_code' in portrait_dict:
            colour_group_code_set = portrait_dict['colour_group_code']
            for colour_group_code in colour_group_code_set:
                colour_group_code_rec = colour_group_code_rec | colour_group_code_portrait_dict[colour_group_code]
            rec = rec.append(random.sample(colour_group_code_rec, rec_num))

        if 'perceived_colour_value_id' in portrait_dict:
            perceived_colour_value_id_set = portrait_dict['perceived_colour_value_id']
            for perceived_colour_value_id in perceived_colour_value_id_set:
                perceived_colour_value_id_rec = perceived_colour_value_id_rec | perceived_colour_value_id_portrait_dict[perceived_colour_value_id]
            rec = rec.append(random.sample(perceived_colour_value_id_rec, rec_num))

        if 'perceived_colour_master_id' in portrait_dict:
            perceived_colour_master_id_set = portrait_dict['perceived_colour_master_id']
            for perceived_colour_master_id in perceived_colour_master_id_set:
                perceived_colour_master_id_rec = perceived_colour_master_id_rec | perceived_colour_master_id_portrait_dict[perceived_colour_master_id]
            rec = rec.append(random.sample(perceived_colour_master_id_rec, rec_num))

        rec = random.sample(rec, rec_num)

        customer_rec[customer] = rec

    np.save("../../output/hm/customer_rec.npy", customer_rec)
