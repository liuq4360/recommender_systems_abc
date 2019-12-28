#!/usr/bin/env python
# coding=utf-8

# 对于每个用户请求，通过推荐接口，获取用户的推荐结果


from flask import Flask, request
import redis
import json

app = Flask(__name__)


@app.route("/rec/hot/")
def hot_rec():
    type_ = request.form.get("type")
    if type_ == "":
        return json.dumps({"status": 400, "reason": "wrong parameters"})
    else:
        r = redis.Redis(host='localhost', port=6379, db=0)
        res = r.zrevrange("hot", 0, -1, "withscores")  # <type 'list'> [('1905', 564361.0), ('2452', 462715.0)]
        # res = r.zrevrange("hot", 0, -1)  # ['1905', '2452', '3938', '3962']
        rec = []
        score = []
        for (r, s) in res:
            rec.append(r)
            score.append(s)
        js = json.dumps({"status": 200, "rec_type": "hot", "rec": rec, "score": score, "type": type_})
        return js


@app.route("/rec/sim/<int:vid>")
def similarity_rec():
    vid = request.form.get("vid", type=int, default=None)
    r = redis.Redis(host='localhost', port=6379, db=1)
    res = r.zrevrange(str(vid), 0, -1, "withscores")  # <type 'list'> [('1905', 564361.0), ('2452', 462715.0)]
    # res = r.zrevrange("hot", 0, -1)  # ['1905', '2452', '3938', '3962']
    rec = []
    score = []
    for (r, s) in res:
        rec.append(r)
        score.append(s)
    js = json.dumps({"status": 200, "rec_type": "similarity", "vid": vid, "rec": rec, "score": score})
    return js


@app.route("/rec/personal/<int:uid>")
def item_based_rec():
    uid = request.form.get("uid", type=int, default=None)
    r = redis.Redis(host='localhost', port=6379, db=2)
    res = r.zrevrange(str(uid), 0, -1, "withscores")  # <type 'list'> [('1905', 564361.0), ('2452', 462715.0)]
    # res = r.zrevrange("hot", 0, -1)  # ['1905', '2452', '3938', '3962']
    rec = []
    score = []
    for (r, s) in res:
        rec.append(r)
        score.append(s)
    js = json.dumps({"status": 200, "rec_type": "item-based", "uid": uid, "rec": rec, "score": score})
    return js


# 参考文档
# json： https://docs.python.org/2/library/json.html
# redis：https://pypi.org/project/redis/ ， https://redis.io/
# flask：https://palletsprojects.com/p/flask/ ， https://dormousehole.readthedocs.io/en/latest/

# 启动： env FLASK_APP=get_rec.py flask run