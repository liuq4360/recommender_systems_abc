#!/usr/bin/env python
# coding=utf-8

r"""
    利用 pytorch 实现wide & deep模型，我们用开源的pytorch-widedeep来实现。
    代码仓库：https://github.com/jrzaurin/pytorch-widedeep
    参考文档：https://pytorch-widedeep.readthedocs.io/en/latest/index.html
"""

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from pytorch_widedeep import Trainer
from pytorch_widedeep.preprocessing import WidePreprocessor, TabPreprocessor
from pytorch_widedeep.models import Wide, TabMlp, WideDeep
from pytorch_widedeep.metrics import Accuracy, Precision, Recall, F1Score
from pytorch_widedeep.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import shuffle
from collections import Counter
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    """
        下面这里的样本处理类似logistics回归排序的处理方法。
    """
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
    trans = trans.sample(n=50000, random_state=1)  # 随机抽取50000行
    trans.reset_index(drop=True, inplace=True)
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

    df.to_csv('../../output/hm/wide_deep_source_data.csv', index=0)
    # df = pd.read_csv("../../output/hm/logistic_source_data.csv")

    # 将数据集划分为训练集train_df和测试集test_df。
    # 用于logistic回归模型的训练，而测试集logistic_test_df用于测试训练好的logistic回归模型的效果。
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    train_df.to_csv('../../output/hm/wide_deep_train_data.csv', index=0)
    test_df.to_csv('../../output/hm/wide_deep_test_data.csv', index=0)

    df_train = pd.read_csv("../../output/hm/wide_deep_train_data.csv")
    df_test = pd.read_csv("../../output/hm/wide_deep_test_data.csv")
    df_train = df_train.drop(columns=['customer_id', 'article_id', ''])
    df_test = df_test.drop(columns=['customer_id', 'article_id'])

    # ['t_dat', 'price', 'sales_channel_id',
    #  'product_code', 'product_type_no', 'graphical_appearance_no',
    #  'colour_group_code', 'perceived_colour_value_id',
    #  'perceived_colour_master_id', 'FN', 'Active', 'club_member_status',
    #  'fashion_news_frequency', 'age', 'postal_code']

    # Define the 'column set up'
    wide_cols = [
        "sales_channel_id",
        "Active",
        "club_member_status",
        "fashion_news_frequency",
    ]
    crossed_cols = [("product_code", "product_type_no"), ("product_code", "FN"),
                    ("graphical_appearance_no", "FN"), ("colour_group_code", "FN"),
                    ("perceived_colour_value_id", "FN"), ("perceived_colour_master_id", "FN")]

    cat_embed_cols = [
        "sales_channel_id",
        "product_code",
        "product_type_no",
        "graphical_appearance_no",
        "colour_group_code",
        "perceived_colour_value_id",
        "perceived_colour_master_id",
        "FN",
        "Active",
        "club_member_status",
        "fashion_news_frequency",
        "postal_code"
    ]
    continuous_cols = ["t_dat", "price", "age"]
    target = "label"
    target = df_train[target].values

    # prepare the data
    wide_preprocessor = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
    X_wide = wide_preprocessor.fit_transform(df_train)

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols, continuous_cols=continuous_cols  # type: ignore[arg-type]
    )
    X_tab = tab_preprocessor.fit_transform(df_train)

    # build the model
    wide = Wide(input_dim=np.unique(X_wide).shape[0], pred_dim=1)
    tab_mlp = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        cat_embed_dropout=0.1,
        continuous_cols=continuous_cols,
        mlp_hidden_dims=[400, 200],
        mlp_dropout=0.5,
        mlp_activation="leaky_relu",
    )
    model = WideDeep(wide=wide, deeptabular=tab_mlp)

    # train and validate
    accuracy = Accuracy(top_k=2)
    precision = Precision(average=True)
    recall = Recall(average=True)
    f1 = F1Score()
    early_stopping = EarlyStopping()
    model_checkpoint = ModelCheckpoint(
        filepath="../../output/hm/tmp_dir/wide_deep_model",
        save_best_only=True,
        verbose=1,
        max_save=1,
    )
    trainer = Trainer(model, objective="binary",
                      optimizers=torch.optim.AdamW(model.parameters(), lr=0.001),
                      callbacks=[early_stopping, model_checkpoint],
                      metrics=[accuracy, precision, recall, f1])
    trainer.fit(
        X_wide=X_wide,
        X_tab=X_tab,
        target=target,
        n_epochs=30,
        batch_size=256,
        val_split=0.2
    )

    # predict on test
    X_wide_te = wide_preprocessor.transform(df_test)
    X_tab_te = tab_preprocessor.transform(df_test)
    pred = trainer.predict(X_wide=X_wide_te, X_tab=X_tab_te)
    # pred_prob = trainer.predict_proba(X_wide=X_wide_te, X_tab=X_tab_te)  # 预测概率
    y_test = df_test['label']
    print(accuracy_score(y_test, pred))

    # Save and load
    trainer.save(
        path="../../output/hm/model_weights",
        model_filename="wd_model.pt",
        save_state_dict=True,
    )

    # prepared the data and defined the new model components:
    # 1. Build the model
    model_new = WideDeep(wide=wide, deeptabular=tab_mlp)
    model_new.load_state_dict(torch.load("../../output/hm/model_weights/wd_model.pt"))

    # 2. Instantiate the trainer
    trainer_new = Trainer(model_new, objective="binary",
                          optimizers=torch.optim.AdamW(model.parameters(), lr=0.001),
                          callbacks=[early_stopping, model_checkpoint],
                          metrics=[accuracy, precision, recall, f1])

    # 3. Either start the fit or directly predict
    pred = trainer_new.predict(X_wide=X_wide_te, X_tab=X_tab_te)
