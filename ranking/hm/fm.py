#!/usr/bin/env python
# coding=utf-8

r"""
  基于 xlearn（pip3 install xlearn 或者直接从源码来安装） 包来实现 fm 算法。

  https://github.com/aksnzhy/xlearn

  输入数据格式：
          CSV format:

           y    value_1  value_2  ..  value_n

           0      0.1     0.2     0.2   ...
           1      0.2     0.3     0.1   ...
           0      0.1     0.2     0.4   ...

  example:
    # Load dataset
    iris_data = load_iris()
    X = iris_data['data']
    y = (iris_data['target'] == 2)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

    # param:
    #  0. binary classification
    #  1. model scale: 0.1
    #  2. epoch number: 10 (auto early-stop)
    #  3. learning rate: 0.1
    #  4. regular lambda: 1.0
    #  5. use sgd optimization method
    linear_model = xl.LRModel(task='binary', init=0.1,
                              epoch=10, lr=0.1,
                              reg_lambda=1.0, opt='sgd')

    # Start to train
    linear_model.fit(X_train, y_train,
                     eval_set=[X_val, y_val],
                     is_lock_free=False)

    # Generate predictions
    y_pred = linear_model.predict(X_val)

"""

import xlearn as xl
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    # Training task
    fm_model = xl.create_fm()  # Use factorization machine
    fm_model.setTrain("../../output/hm/fm_train_data.csv")  # Training data

    param = {"task": "binary",
             "lr": 0.2,
             "lambda": 0.002,
             "metric": 'acc',
             "epoch": 20,
             "opt": 'sgd',
             "init": 0.1,
             "k": 15  # Dimensionality of the latent factors
             }

    # Use cross-validation
    # fm_model.cv(param)

    # Start to train
    # The trained model will be stored in model.out
    fm_model.fit(param, '../../output/hm/fm_model.out')

    # Prediction task
    fm_model.setTest("../../output/hm/predict_data.csv")  # Test data
    fm_model.setSigmoid()  # Convert output to 0-1

    # Start to predict
    # The output result will be stored in output.txt
    fm_model.predict("../../output/hm/fm_model.out", "../../output/hm/fm_predict.csv")

    # fm_model = xl.FMModel(task='binary', init=0.1,
    #                       epoch=10, k=4, lr=0.1,
    #                       reg_lambda=0.01, opt='sgd',
    #                       metric='acc')
    # # Start to train
    # fm_model.fit(X_train,
    #              y_train,
    #              eval_set=[X_val, y_val])
    #
    # # print model weights
    # print(fm_model.weights)
    #
    # # Generate predictions
    # y_pred = fm_model.predict(X_val)
