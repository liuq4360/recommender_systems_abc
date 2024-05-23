#!/usr/bin/env python
# coding=utf-8

r"""
利用gensim框架来实现item嵌入，采用item2vec算法实现。
gensim：https://github.com/RaRe-Technologies/gensim
document：https://radimrehurek.com/gensim/

"""

import numpy as np
from gensim.models import Word2Vec
import pandas as pd


if __name__ == "__main__":
    # common_texts = [
    #     ['human', 'interface', 'computer'],
    #     ['survey', 'user', 'computer', 'system', 'response', 'time'],
    #     ['eps', 'user', 'interface', 'system'],
    #     ['system', 'human', 'system', 'eps'],
    #     ['user', 'response', 'time'],
    #     ['trees'],
    #     ['graph', 'trees'],
    #     ['graph', 'minors', 'trees'],
    #     ['graph', 'minors', 'survey']
    # ]
    # sentences = LineSentence(datapath('lee_background.cor'))
    trans = pd.read_csv("../../data/hm/transactions_train.csv")
    tmp_df = trans[['customer_id', 'article_id']]
    grouped_df = tmp_df.groupby('customer_id')
    groups = grouped_df.groups
    train_data = []
    for customer_id in groups.keys():
        customer_df = grouped_df.get_group(customer_id)
        tmp_lines = list(customer_df['article_id'].values)
        lines = []
        for word in tmp_lines:
            lines.append(str(word))
        train_data.append(lines)

    model = Word2Vec(sentences=train_data, vector_size=100, window=5, min_count=3, workers=4)
    model.save("../../output/hm/word2vec.model")
    # model = Word2Vec.load("../../output/hm/word2vec.model")
    # vector = model.wv['computer']  # get numpy vector of a word
    sims = model.wv.most_similar('657395002', topn=10)  # get other similar words
    print(sims)





