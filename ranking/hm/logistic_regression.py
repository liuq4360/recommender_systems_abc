#!/usr/bin/env python
# coding=utf-8

r"""
利用scikit-learn 中的 logistic回归 算法来进行召回。模型的主要特征有如下3类：
1、用户相关特征：基于customers.csv表格中的数据。下面6个字段都作为特征。
    FN,   35%有值，值为1；65% 缺失。
    Active,  34% 1， 66% 无值。
    club_member_status, 俱乐部成员状态， 93% ACTIVE，7% PRE-CREATE（预先创建）。
    fashion_news_frequency, 时尚新闻频率， 64% NONE，35% Regularly（有规律地），1% 其它值。
    age, 年龄，有缺失值，1%缺失。
    postal_code, 邮政代码，很长的字符串。例如，52043ee2162cf5aa7ee79974281641c6f11a68d276429a91f8ca0d4b6efa8100。
2、物品相关特征：基于articles.csv表格中的数据。下面6个字段作为特征，还有很多字段没用到，也可能能用，读者可以自己探索。
    product_code, 产品code，7位数字字符，如 0108775，是 article_id 的前 7 位。
    product_type_no, 产品类型no，2位或者3位数字，有 -1 值。
    graphical_appearance_no, 图案外观no，如 1010016。
    colour_group_code, 颜色组code，如 09，2位数字
    perceived_colour_value_id, 感知颜色值id。 -1，1，2，3，4，5，6，7，一共这几个值。
    perceived_colour_master_id, 感知颜色主id。1位或者2位数字。
3、用户行为相关特征：基于transactions_train.csv数据。下面2个特征需要使用。
    t_dat，时间，就是商品的购买时间。如2019-09-18，只精确到日。
    price, 价格
    sales_channel_id, 销售渠道id，值只有1、2两个，估计是线上、线下两个。
   另外再准备几个用户行为统计特征，具体如下：
    用户购买频次：总购买次数/用户最近购买和最远购买之间的星期数。
    用户客单价：该用户所有购买的平均价格。
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.utils import shuffle
from collections import Counter

if __name__ == "__main__":
    art = pd.read_csv("../../data/hm/articles.csv")
    cust = pd.read_csv("../../data/hm/customers.csv")

    cust.loc[:, ['FN']] = cust.loc[:, ['FN']].fillna(0)
    cust.loc[:, ['Active']] = cust.loc[:, ['Active']].fillna(0)
    cust.loc[:, ['club_member_status']] = cust.loc[:, ['club_member_status']].fillna('other')
    cust.loc[:, ['fashion_news_frequency']] = cust.loc[:, ['fashion_news_frequency']].fillna('other')
    cust.loc[:, ['age']] = cust.loc[:, ['age']].fillna(int(cust['age'].mode()[0]))
    cust['age'] = cust['age']/100.0

    # len(pd.unique(art['article_id']))  # 某列唯一值的个数
    trans = pd.read_csv("../../data/hm/transactions_train.csv")  # 都是正样本。
    # 到目前为止经历的年数。
    trans['label'] = 1
    positive_num = trans.shape[0]

    # 数据中没有负样本，还需要人工构建一些负样本。
    # 负样本中的price，用目前正样本的price的平均值。
    price = trans[['article_id', 'price']].groupby('article_id').mean()
    price_dict = price.to_dict()['price']
    # 负样本的sales_channel_id用正样本的中位数。
    channel = trans[['article_id', 'sales_channel_id']].groupby('article_id').median()
    channel['sales_channel_id'] = channel['sales_channel_id'].apply(lambda x: int(x))
    channel_dict = channel.to_dict()['sales_channel_id']
    t = trans['t_dat']
    date = t.mode()[0]  # 用众数来表示负样本的时间。 '2019-09-28'

# 采用将正样本的customer_id、article_id两列随机打散的思路（这样customer_id和article_id就可以
    # 随机组合了）来构建负样本。
    cust_id = shuffle(trans['customer_id']).to_list()
    art_id = shuffle(trans['article_id']).to_list()
    data = {'customer_id': cust_id, 'article_id': art_id}
    negative_df = pd.DataFrame(data, index=list(range(positive_num, 2*positive_num, 1)))
    negative_df['t_dat'] = date
    negative_df['price'] = negative_df['article_id'].apply(lambda i: price_dict[i])
    negative_df['sales_channel_id'] = negative_df['article_id'].apply(lambda i: channel_dict[i])

    # 调整列的顺序，跟正样本保持一致
    negative_df = negative_df[['t_dat', 'customer_id', 'article_id', 'price', 'sales_channel_id']]
    negative_df['label'] = 0
    df = pd.concat([trans, negative_df], ignore_index=True)  # 重新进行索引
    df['t_dat'] = pd.to_datetime(df['t_dat']).rsub(pd.Timestamp('now').floor('d')).dt.days/365.0
    df = shuffle(df)
    df.reset_index(drop=True, inplace=True)

    df = df.merge(cust, on=['customer_id'],
                  how='left').merge(art, on=['article_id'], how='left')

    df = df[['customer_id', 'article_id', 't_dat', 'price', 'sales_channel_id',
             'product_code', 'product_type_no', 'graphical_appearance_no', 'colour_group_code',
             'perceived_colour_value_id', 'perceived_colour_master_id', 'FN',
             'Active', 'club_member_status', 'fashion_news_frequency', 'age', 'postal_code', 'label']]

    # product_code：取出现次数最多的前10个，后面的合并。
    most_frequent_top10_product_code = np.array(Counter(df.product_code).most_common(10))[:, 0]
    # 如果color不是最频繁的10个color,那么就给定一个默认值0，减少one-hot编码的维度
    df['product_code'] = df['product_code'].apply(lambda x: x if x in most_frequent_top10_product_code else -1)

    # product_type_no：取出现次数最多的前10个，后面的合并。
    most_frequent_top10_product_type_no = np.array(Counter(df.product_type_no).most_common(10))[:, 0]
    # 如果color不是最频繁的10个color,那么就给定一个默认值0，减少one-hot编码的维度
    df['product_type_no'] = df['product_type_no'].apply(lambda x: x if x in most_frequent_top10_product_type_no else -1)

    # postal_code：取出现次数最多的前10个，后面的合并。
    most_frequent_top100_postal_code = np.array(Counter(df.postal_code).most_common(100))[:, 0]
    # 如果color不是最频繁的10个color,那么就给定一个默认值0，减少one-hot编码的维度
    df['postal_code'] = df['postal_code'].apply(lambda x: x if x in most_frequent_top100_postal_code else "other")

    df.to_csv('../../output/hm/logistic_source_data.csv', index=0)
    # df = pd.read_csv("../../output/hm/logistic_source_data.csv")

    one_hot = OneHotEncoder(handle_unknown='ignore')

    one_hot_data = df[['sales_channel_id', 'product_code', 'product_type_no', 'graphical_appearance_no',
                       'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id',
                       'FN', 'Active', 'club_member_status', 'fashion_news_frequency', 'postal_code']]

    one_hot.fit(one_hot_data)

    feature_array = one_hot.transform(np.array(one_hot_data)).toarray()
    # 两个ndarray水平合并，跟data['id']合并，方便后面两个DataFrame合并
    feature_array_add_id = np.hstack((np.asarray([df['customer_id'].values]).T,
                                      np.asarray([df['article_id'].values]).T, feature_array))
    one_hot_df = DataFrame(feature_array_add_id,
                           columns=np.hstack((np.asarray(['customer_id']),
                                              np.asarray(['article_id']),
                                              one_hot.get_feature_names_out())))

    one_hot_df['customer_id'] = one_hot_df['customer_id'].apply(lambda x: int(x))
    one_hot_df['article_id'] = one_hot_df['article_id'].apply(lambda x: int(x))

    # 三类特征合并。
    final_df = df.merge(one_hot_df, on=['customer_id', 'article_id'], how='left')

    final_df = final_df.drop(columns=['sales_channel_id', 'product_code', 'product_type_no', 'graphical_appearance_no',
                                      'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id',
                                      'FN', 'Active', 'club_member_status', 'fashion_news_frequency', 'postal_code'])

    # index = 0 写入时不保留索引列。
    final_df.to_csv('../../output/hm/logistic_model_data.csv', index=0)
    # read
    # data_and_features_df = pd.read_csv(data_path + '/' + r'logistic_model_data.csv')

    # 将数据集划分为训练集logistic_train_df和测试集logistic_test_df。训练集logistic_train_df
    # 用于logistic回归模型的训练，而测试集logistic_test_df用于测试训练好的logistic回归模型的效果。
    logistic_train_df, logistic_test_df = train_test_split(final_df,
                                                           test_size=0.3, random_state=42)

    logistic_train_df.to_csv('../../output/hm/logistic_train_data.csv', index=0)
    logistic_test_df.to_csv('../../output/hm/logistic_test_data.csv', index=0)

    """
    该脚本主要完成3件事情：
    1. 训练logistic回归模型；
    2. 针对测试集进行预测；
    3. 评估训练好的模型在测试集上的效果；
    
    这个脚本中的所有操作都可以借助scikit-learn中的函数来实现，非常简单。
    这里为了简单起见，将模型训练、预测与评估都放在这个文件中了。
    
    关于logistic回归模型各个参数的含义及例子可以参考，https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
    
    关于模型评估的案例可以参考：https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_display_object_visualization.html#sphx-glr-auto-examples-miscellaneous-plot-display-object-visualization-py
    
    """

    logistic_train_df = pd.read_csv('../../output/hm/logistic_train_data.csv')

    """
    下面代码是训练logistic回归模型。
    """

    clf = LogisticRegression(penalty='l2',
                             solver='liblinear', tol=1e-6, max_iter=1000)

    X_train = logistic_train_df.drop(columns=['customer_id', 'article_id', 'label', ])
    y_train = logistic_train_df['label']

    clf.fit(X_train, y_train)

    # clf.coef_
    # clf.intercept_
    # clf.classes_

    """
    下面的代码用上面训练好的logistic回归模型来对测试集进行预测。
    """
    logistic_test_df = pd.read_csv('../../output/hm/logistic_test_data.csv')

    X_test = logistic_test_df.drop(columns=['customer_id', 'article_id', 'label', ])

    y_test = logistic_test_df['label']

    # logistic回归模型预测出的结果为y_score
    y_score = clf.predict(X_test)

    # 包含概率值的预测
    # y_score = clf.predict_proba(X_test)

    # np.unique(Z)
    # Counter(Z).most_common(2)
    # logistic_test_df.label.value_counts()

    """
    下面的代码对logistic回归模型进行效果评估，主要有3种常用的评估方法：
    1. 混淆矩阵：confusion matrix
    2. roc曲线：roc curve
    3. 精准度和召回率：precision recall
    """

    # confusion matrix
    # 混淆矩阵参考百度词条介绍：https://baike.baidu.com/item/%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5/10087822?fr=aladdin
    y_score = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_score)
    cm_display = ConfusionMatrixDisplay(cm).plot()

    # roc curve
    # ROC 和 AUC 的介绍见：
    # 1. https://baijiahao.baidu.com/s?id=1671508719185457407&wfr=spider&for=pc
    # 2. https://blog.csdn.net/yinyu19950811/article/details/81288287
    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=clf.classes_[1])
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

    # precision recall
    # 准确率和召回率的介绍参考：
    # 1. https://www.zhihu.com/question/19645541/answer/91694636

    pre, recall, _ = precision_recall_curve(y_test, y_score, pos_label=clf.classes_[1])
    pr_display = PrecisionRecallDisplay(precision=pre, recall=recall).plot()

