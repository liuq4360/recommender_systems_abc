#!/usr/bin/env python
# coding=utf-8

r"""
    利用 tensorflow2 实现。
    代码仓库：https://github.com/hyez/Deep-Youtube-Recommendations
    参考文档：
"""
import tensorflow as tf
import datetime
import os
import numpy as np
import pandas as pd
import random

# data 读取
users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv(
    '../../data/movie_lens/u.user', sep='|', names=users_cols, encoding='latin-1')

watch_cols = ['user_id', 'movie_id', 'watch_hist_time']
watches = pd.read_csv(
    '../../data/movie_lens/u.watch', sep='\t', names=watch_cols, encoding='latin-1')

search_cols = ['user', 'search_hist']
searches = pd.read_csv(
    '../../data/movie_lens/u.search', sep='\t', names=search_cols, encoding='latin-1')

# The movies file contains a binary feature for each genre.
genre_cols = [
    "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
movies_cols = [
                  'movie_id', 'title', 'release_date', "video_release_date", "imdb_url"
              ] + genre_cols
movies = pd.read_csv(
    '../../data/movie_lens/u.item', sep='|', names=movies_cols, encoding='latin-1')

# Since the ids start at 1, we shift them to start at 0.
# users["user_id"] = users["user_id"].apply(lambda x: str(x-1))
# movies["movie_id"] = movies["movie_id"].apply(lambda x: str(x-1))
movies["year"] = movies['release_date'].apply(lambda x: str(x).split('-')[-1])
# watches["movie_id"] = watches["movie_id"].apply(lambda x: str(x-1))
# watches["user_id"] = watches["user_id"].apply(lambda x: str(x-1))
# searches["user_id"] = searches["user"].apply(lambda x: str(x-1))

# 添加example_age
movies['example_age'] = (pd.to_datetime("now") - pd.to_datetime(movies['release_date'])) \
                        / np.timedelta64(1, 'D')


# normalize
def normalize_col(df, col_name):
    df[col_name] = (df[col_name] - df[col_name].min()) / (df[col_name].max() - df[col_name].min())
    return df


movies = normalize_col(movies, 'example_age')
watches = normalize_col(watches, 'watch_hist_time')


# data 合并
data = watches.merge(movies, on='movie_id').merge(users, on='user_id')
data['user_id'] = data['user_id'].astype(int)
data['movie_id'] = data['movie_id'].astype(int)
data = data.set_index(['user_id']).sort_index()
data = data.reset_index()
data['movie_name'] = data['title'].str[:-6]  # title部分剪裁


# occupation 编码
occupations = data["occupation"].unique().tolist()
occupations_encoded = {x: i for i, x in enumerate(occupations)}
occupationsencoded2occupations = {i: x for i, x in enumerate(occupations)}

# search history 编码
search_hists = searches["search_hist"].unique().tolist()
search_encoded = {x: i for i, x in enumerate(search_hists)}
searchencoded2search = {i: x for i, x in enumerate(search_hists)}

# 用户索引编码
user_ids = data["user_id"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}

# 电影索引编码
movie_ids = data["movie_id"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

# 电影标题编码
title_ids = data["title"].unique().tolist()
title2title_encoded = {x: i for i, x in enumerate(title_ids)}
title_encoded2title = {i: x for i, x in enumerate(title_ids)}

# 更改为编码
data["user"] = data["user_id"].map(user2user_encoded)
data["movie"] = data["movie_id"].map(movie2movie_encoded)
data["title_d"] = data["title"].map(title2title_encoded)
searches["search_hist"] = searches["search_hist"].map(search_encoded)
data["occupation"] = data["occupation"].map(occupations_encoded)
# searches["search_hist"] = searches["search_hist"]
searches = normalize_col(searches, 'search_hist')

watch_hist = data.groupby(['user'])['movie_id'].apply(list).reset_index()
search_hist = searches.groupby(['user'])['search_hist'].apply(list).reset_index()
watch_hist_time = data.groupby(['user'])['watch_hist_time'].apply(list).reset_index()
example_age = data.groupby(['user'])['example_age'].apply(list).reset_index()

user_video_list = data.pivot(index='user_id', columns='movie_id', values='movie').reset_index()
user_video_list.fillna(data["movie_id"].max()+1, inplace=True)

sample_data = data[['user', 'occupation', 'sex']]
sample_data = sample_data.reset_index()
sample_data = sample_data.drop('index', axis=1)
sample_data = sample_data.drop_duplicates()

user_movie_list = pd.merge(sample_data, watch_hist, how='left')
user_movie_list = pd.merge(user_movie_list,  watch_hist_time, how='left')
user_movie_list = pd.merge(user_movie_list, search_hist, how='left')
user_movie_list = pd.merge(user_movie_list, example_age, how='left')
user_movie_list['search_hist'] = user_movie_list['search_hist'].apply(lambda x: x if type(x) is list else [])  # NaN处理
user_movie_list['predict_labels'] = user_movie_list['movie_id'].apply(
    lambda x: int(random.uniform(0, data["movie"].max())))  # label作为最后一个值…

train_data = user_movie_list[(user_movie_list.user >= 1) &
                             (user_movie_list.user <= 5)]
test_data = user_movie_list[(user_movie_list.user >= 6) &
                            (user_movie_list.user <= 10)]

EMBEDDING_DIMS = 16
DENSE_UNITS = 64
DROPOUT_PCT = 0.0
ALPHA = 0.0
NUM_CLASSES = data["movie"].max() + 2
LEARNING_RATE = 0.003


class MaskedEmbeddingsAggregatorLayer(tf.keras.layers.Layer):
    def __init__(self, agg_mode='sum', **kwargs):
        super(MaskedEmbeddingsAggregatorLayer, self).__init__(**kwargs)
        if agg_mode not in ['sum', 'mean']:
            raise NotImplementedError('mode {} not implemented!'.format(agg_mode))
        self.agg_mode = agg_mode

    @tf.function
    def call(self, inputs, mask=None):
        masked_embeddings = tf.ragged.boolean_mask(inputs, mask)
        if self.agg_mode == 'sum':
            aggregated = tf.reduce_sum(masked_embeddings, axis=1)
        elif self.agg_mode == 'mean':
            aggregated = tf.reduce_mean(masked_embeddings, axis=1)
        return aggregated

    def get_config(self):
        # this is used when loading a saved model that uses a custom layer
        return {'agg_mode': self.agg_mode}


class L2NormLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(L2NormLayer, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs, mask=None):
        if mask is not None:
            inputs = tf.ragged.boolean_mask(inputs, mask).to_tensor()
        return tf.math.l2_normalize(inputs, axis=-1)

    def compute_mask(self, inputs, mask):
        return mask


input_watch_hist = tf.keras.Input(shape=(None, ), name='watch_hist')
input_watch_hist_time = tf.keras.layers.Input(shape=(None,), name='watch_hist_time')
input_search_hist = tf.keras.layers.Input(shape=(None,), name='search_hist')
input_example_age = tf.keras.Input(shape=(None, ), name='example_age')
input_occupation = tf.keras.Input(shape=(None, ), name='occupation')

#--- layers
features_embedding_layer = tf.keras.layers.Embedding(input_dim=NUM_CLASSES, output_dim=EMBEDDING_DIMS,
                                                     mask_zero=True, trainable=True, name='features_embeddings')
labels_embedding_layer = tf.keras.layers.Embedding(input_dim=NUM_CLASSES, output_dim=EMBEDDING_DIMS,
                                                   mask_zero=True, trainable=True, name='labels_embeddings')

avg_embeddings = MaskedEmbeddingsAggregatorLayer(agg_mode='mean', name='aggregate_embeddings')

dense_1 = tf.keras.layers.Dense(units=DENSE_UNITS, name='dense_1')
dense_2 = tf.keras.layers.Dense(units=DENSE_UNITS, name='dense_2')
dense_3 = tf.keras.layers.Dense(units=DENSE_UNITS, name='dense_3')
l2_norm_1 = L2NormLayer(name='l2_norm_1')

dense_output = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax, name='dense_output')

#--- features
features_embeddings = features_embedding_layer(input_watch_hist)
l2_norm_features = l2_norm_1(features_embeddings)
avg_features = avg_embeddings(l2_norm_features)

labels_watch_embeddings = labels_embedding_layer(input_watch_hist_time)
l2_norm_watched = l2_norm_1(labels_watch_embeddings)
avg_watched = avg_embeddings(l2_norm_watched)

labels_search_embeddings = labels_embedding_layer(input_search_hist)
l2_norm_searched = l2_norm_1(labels_search_embeddings)
avg_searched = avg_embeddings(l2_norm_searched)

labels_example_age_embeddings = labels_embedding_layer(input_example_age)
l2_norm_example_age = l2_norm_1(labels_example_age_embeddings)
avg_example_age = avg_embeddings(l2_norm_example_age)

labels_occupation_embeddings = labels_embedding_layer(input_occupation)
l2_norm_occupation = l2_norm_1(labels_occupation_embeddings)
avg__occupation = avg_embeddings(l2_norm_occupation)


print(avg_features)
print(avg_watched)
print(avg_searched)
print(avg_example_age)
print(input_occupation)

# 임베딩 벡터들 연결
concat_inputs = tf.keras.layers.Concatenate(axis=1)([avg_features,
                                                     avg_watched,
                                                     avg_searched,
                                                     avg_example_age,
                                                     #                                                      avg__occupation
                                                     ])
# Dense Layers
dense_1_features = dense_1(concat_inputs)
dense_1_relu = tf.keras.layers.ReLU(name='dense_1_relu')(dense_1_features)
dense_1_batch_norm = tf.keras.layers.BatchNormalization(name='dense_1_batch_norm')(dense_1_relu)

dense_2_features = dense_2(dense_1_relu)
dense_2_relu = tf.keras.layers.ReLU(name='dense_2_relu')(dense_2_features)
# dense_2_batch_norm = tf.keras.layers.BatchNormalization(name='dense_2_batch_norm')(dense_2_relu)

dense_3_features = dense_3(dense_2_relu)
dense_3_relu = tf.keras.layers.ReLU(name='dense_3_relu')(dense_3_features)
dense_3_batch_norm = tf.keras.layers.BatchNormalization(name='dense_3_batch_norm')(dense_3_relu)
outputs = dense_output(dense_3_batch_norm)

#Optimizer
optimiser = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

#--- prep model
model = tf.keras.models.Model(
    inputs=[input_watch_hist,
            input_watch_hist_time,
            input_search_hist,
            input_example_age,
            #             input_occupation,
            ],
    outputs=[outputs]
)
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['acc'])

model.summary()

history = model.fit([tf.keras.preprocessing.sequence.pad_sequences(train_data['movie_id']),
                     tf.keras.preprocessing.sequence.pad_sequences(train_data['watch_hist_time'], dtype=float),
                     tf.keras.preprocessing.sequence.pad_sequences(train_data['search_hist'], dtype=float) + 1e-10,
                     tf.keras.preprocessing.sequence.pad_sequences(train_data['example_age'], dtype=float),
                     #            tf.keras.preprocessing.sequence.pad_sequences(train_data['occupation'], dtype=float),
                     ], train_data['predict_labels'].values,
                    steps_per_epoch=1, epochs=50)

model.save("candidate_generation.h5")

pred = model.predict([tf.keras.preprocessing.sequence.pad_sequences(test_data['movie_id']),
                      tf.keras.preprocessing.sequence.pad_sequences(test_data['watch_hist_time'], dtype=float),
                      tf.keras.preprocessing.sequence.pad_sequences(test_data['search_hist'], dtype=float) + 1e-10,
                      tf.keras.preprocessing.sequence.pad_sequences(test_data['example_age'], dtype=float)
                      ])

# candidate generation:
# 每个用户抽取top-7个推荐数据
N = 6
k = np.sort((-pred).argsort()[:, :N])
print(k)
k = k.flatten()
k[k > data["movie"].max()] = 0
k = np.unique(k)

