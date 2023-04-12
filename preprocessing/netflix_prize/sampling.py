#!/usr/bin/env python
# coding=utf-8

# 将三元组(userid，videoid，score)数据，按照7：3的比例随机分为训练集和测试集

import os
import random

cwd = os.getcwd()  # 获取当前工作目录
f_path = os.path.abspath(os.path.join(cwd, ".."))

data = f_path + "/output/data.txt"

train = f_path + "/output/train.txt"
test = f_path + "/output/test.txt"

f_train = open(train, 'w')
f_test = open(test, 'w')

f_data = open(data)        # 返回一个文件对象
line = f_data.readline()   # 调用文件的 readline()方法
while line:

    rand = random.random()
    if rand < 0.7:
        f_train.write(line)
    else:
        f_test.write(line)

    line = f_data.readline()

f_data.close()
f_train.close()
f_test.close()

