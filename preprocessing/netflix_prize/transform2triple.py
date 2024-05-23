#!/usr/bin/env python
# coding=utf-8

# 将training_set转换为三元组(userid，videoid，score)

import os


cwd = os.getcwd()  # 获取当前工作目录
f_path = os.path.abspath(os.path.join(cwd, "..", ".."))  # 获取上一级目录
all_files = os.listdir(f_path + '/data/netflix_prize/mini_training_set')

data = f_path + "/output/netflix_prize/data.txt"
fp = open(data, 'w')  # 打开文件，如果文件不存在则创建它，该文件是存储最终的三元组


for path in all_files:
    with open(f_path+"/data/netflix_prize/mini_training_set/"+path, "r") as f:    # 设置文件对象
        lines = f.readlines()
    video_id = lines[0].strip(':\n')
    for l in lines[1:]:
        user_score_time = l.strip('\n').split(',')
        user_id = user_score_time[0]
        score = user_score_time[1]
        fp.write(user_id+","+video_id+","+score+"\n")

fp.close()

