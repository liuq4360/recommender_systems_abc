#!/usr/bin/env python
# coding=utf-8

r"""
利用scikit-learn 中的 kmeans 算法来进行召回。我们只实现基于物品的聚类算法。
另外，在实现过程中只利用物品本身的数据进行聚类，理论上item2vec数据也是可以用的。
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from pandas import DataFrame
from sklearn.cluster import KMeans
import random


if __name__ == "__main__":
    art = pd.read_csv("../../data/hm/articles.csv")
    # customers = pd.read_csv("../../data/hm/customers.csv")
    # len(pd.unique(art['article_id'])) 某列唯一值的个数
    # trans = pd.read_csv("../../data/hm/transactions_train.csv")

    # 类似用户画像部分，我们只关注下面6个类别特征，先将类别特征one-hot编码，然后进行聚类。
    art = art[['article_id', 'product_code', 'product_type_no', 'graphical_appearance_no',
               'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id']]
    # 'product_code' : 47224 个不同的值。
    # 'product_type_no'：132 个不同的值。
    # 'graphical_appearance_no'：30 个不同的值。
    # 'colour_group_code'：50 个不同的值。
    # 'perceived_colour_value_id'：8 个不同的值。
    # 'perceived_colour_master_id'：20 个不同的值。

    # product_code：取出现次数最多的前10个，后面的合并。
    most_freq_top10_prod_code = np.array(Counter(art.product_code).most_common(10))[:, 0]
    # 如果color不是最频繁的10个color,那么就给定一个默认值0，减少one-hot编码的维度
    art['product_code'] = art['product_code'].apply(lambda t: t if t in most_freq_top10_prod_code else -1)

    # product_type_no：取出现次数最多的前10个，后面的合并。
    most_frequent_top10_product_type_no = np.array(Counter(art.product_type_no).most_common(10))[:, 0]
    # 如果color不是最频繁的10个color,那么就给定一个默认值0，减少one-hot编码的维度
    art['product_type_no'] = art['product_type_no'].apply(
        lambda t: t if t in most_frequent_top10_product_type_no else -1)

    one_hot = OneHotEncoder(handle_unknown='ignore')

    one_hot_data = art[['product_code', 'product_type_no', 'graphical_appearance_no',
                        'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id']]

    one_hot.fit(one_hot_data)

    feature_array = one_hot.transform(np.array(one_hot_data)).toarray()
    # 两个ndarray水平合并，跟data['id']合并，方便后面两个DataFrame合并
    feature_array_add_id = np.hstack((np.asarray([art['article_id'].values]).T, feature_array))
    # one_hot_features_df = DataFrame(feature_array, columns=one_hot.get_feature_names())
    df_train = DataFrame(feature_array_add_id, columns=np.hstack((np.asarray(['article_id']),
                                                                  one_hot.get_feature_names_out())))

    df_train['article_id'] = df_train['article_id'].apply(lambda t: int(t))

    # df_train = df_train.drop(columns=['article_id'])

    # index = 0 写入时不保留索引列。
    df_train.to_csv('../../output/hm/kmeans_train.csv', index=0)

    n_clusters = 1000
    # X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    # k_means = KMeans(n_clusters=2, random_state=0).fit(X)

    # n_clusters: 一共聚多少类，默认值8
    # init：选择中心点的初始化方法，默认值k-means++
    # n_init：算法基于不同的中心点运行多少次，最后的结果基于最好的一次迭代的结果，默认值10
    # max_iter: 最大迭代次数，默认值300
    k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10,
                     max_iter=300).fit(df_train.drop(columns=['article_id']).values)

    # 训练样本中每条记录所属的类别
    print(k_means.labels_)
    # 预测某个样本属于哪个聚类
    # print(k_means.predict(np.random.rand(1, df_train.shape[1])))
    print(k_means.predict(np.random.randint(20, size=(2, df_train.drop(columns=['article_id']).shape[1]))))
    # 每个聚类的聚类中心
    print(k_means.cluster_centers_)

    result_array = np.hstack((np.asarray([df_train['article_id'].values]).T,
                              np.asarray([k_means.labels_]).T))

    # 将物品id和具体的类别转化为DataFrame。
    cluster_result = DataFrame(result_array, columns=['article_id', 'cluster'])

    # index = 0 写入时不保留索引列。
    cluster_result.to_csv('../../output/hm/kmeans.csv', index=0)
    # read
    # cluster_result = pd.read_csv('../../output/hm/kmeans.csv')

    # 给用户推荐的物品数量的数量
    rec_num = 10

    df_cluster = pd.read_csv('../../output/hm/kmeans.csv')

    # 每个id对应的cluster的映射字典。
    id_cluster_dict = dict(df_cluster.values)

    tmp = df_cluster.values
    cluster_ids_dict = {}
    for i in range(tmp.shape[0]):
        [id_, cluster_] = tmp[i]
        if cluster_ in cluster_ids_dict.keys():
            cluster_ids_dict[cluster_] = cluster_ids_dict[cluster_] + [id_]
        else:
            cluster_ids_dict[cluster_] = [id_]

    # 一共有多少个类
    # cluster_num = len(cluster_ids_dict)
    # 打印出每一个类有多少个元素，即每类有多少物品
    for x, y in cluster_ids_dict.items():
        print("cluster " + str(x) + " : " + str(len(y)))

    # source_df = pd.read_csv("../../data/hm/articles.csv")

    # 基于聚类，为每个物品关联k个最相似的物品。
    def article_similar_recall(art_id, k):
        rec = cluster_ids_dict.get(id_cluster_dict.get(art_id))
        if art_id in rec:
            rec.remove(art_id)
        return random.sample(rec, k)

    article_id = 952937003
    topn_sim = article_similar_recall(article_id, rec_num)

    meta_info_dict = pd.read_csv("../../output/hm/article_metadata_dict.npy")

    print(meta_info_dict[article_id])
    for item in topn_sim:
        print(meta_info_dict[item])
