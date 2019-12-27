#!/usr/bin/env python
# coding=utf-8

# 对于每个用户请求，通过推荐接口，获取用户的推荐结果


from flask import Flask
import redis
import json

app = Flask(__name__)


@app.route("/rec/hot")
def hot_rec():
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.zrevrangebyscore()

    return "Hello, World!"

