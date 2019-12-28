#!/usr/bin/env python
# coding=utf-8

# 对于每个用户请求，通过推荐接口，获取用户的推荐结果


from flask import Flask, request
import redis
import json

app = Flask(__name__)


metadata_r = redis.Redis(host='localhost', port=6379, db=3)
metadata_map = metadata_r.hgetall("metadata")
metadata_r.close()


def construct_rec_info(rec_list, score_list):
    """
    为推荐列表构造合适的前端展示的数据结构

    :param rec_list: 推荐的电影列表
    :param score_list: 推荐的电影的得分
    :return: rec_items: [{"vid":1,"score"：0.3，"year":2014, "title":"ABC"},...]
    """
    rec_items = []
    for i in range(len(rec_list)):
        vid = rec_list[i]
        score = score_list[i]
        info = metadata_map[vid]
        info_dict = json.loads(info)  # {u'year': u'2004', u'title': u'Thomas & Friends: The Early Years'}
        rec_dict = dict({"vid": vid, "score": score}, **info_dict)
        rec_items.append(rec_dict)
    return rec_items


@app.route("/rec/hot", methods=['GET'])
def hot_rec():
    r = redis.Redis(host='localhost', port=6379, db=0)
    res = r.zrevrange("hot", 0, -1, "withscores")  # <type 'list'> [('1905', 564361.0), ('2452', 462715.0)]
    # res = r.zrevrange("hot", 0, -1)  # ['1905', '2452', '3938', '3962']
    r.close()
    rec = []
    score = []
    for (r, s) in res:
        rec.append(r)
        score.append(s)
    rec_items = construct_rec_info(rec, score)
    js = json.dumps({"status": 200, "alg": "hot", "items": rec_items})
    return js


@app.route("/rec/sim")
def similarity_rec():
    args = request.args
    if "vid" in args:
        vid = args["vid"]
        if vid != '' and vid.isdigit():
            r = redis.Redis(host='localhost', port=6379, db=1)
            res = r.zrevrange(str(vid), 0, -1, "withscores")  # <type 'list'> [('1905', 564361.0), ('2452', 462715.0)]
            # res = r.zrevrange("hot", 0, -1)  # ['1905', '2452', '3938', '3962']
            r.close()
            rec = []
            score = []
            for (r, s) in res:
                rec.append(r)
                score.append(s)
            rec_items = construct_rec_info(rec, score)
            info = metadata_map[vid]
            info_dict = json.loads(info)
            js = json.dumps({"status": 200, "alg": "similarity", "vid": vid, "info": info_dict,  "items": rec_items})
            return js
        else:
            js = json.dumps({"status": 404, "wrong": "vid format is wrong or don\'t have data!"})
            return js
    else:
        js = json.dumps({"status": 404, "wrong": "don\'t have vid parameters"})
        return js


@app.route("/rec/personal")
def item_based_rec():
    args = request.args
    if "uid" in args:
        uid = args["uid"]
        if uid != '' and uid.isdigit():
            r = redis.Redis(host='localhost', port=6379, db=2)
            res = r.zrevrange(str(uid), 0, -1, "withscores")  # <type 'list'> [('1905', 564361.0), ('2452', 462715.0)]
            # res = r.zrevrange("hot", 0, -1)  # ['1905', '2452', '3938', '3962']
            r.close()
            rec = []
            score = []
            for (r, s) in res:
                rec.append(r)
                score.append(s)
            rec_items = construct_rec_info(rec, score)
            js = json.dumps({"status": 200, "alg": "item-based", "uid": uid, "items": rec_items})
            return js
        else:
            js = json.dumps({"status": 404, "wrong": "uid format is wrong or don\'t have data!"})
            return js
    else:
        js = json.dumps({"status": 404, "wrong": "don\'t have uid parameters"})
        return js


if __name__ == "__main__":
    app.run(debug=True, threaded=True)

# 参考文档
# json： https://docs.python.org/2/library/json.html
# redis：https://pypi.org/project/redis/ ， https://redis.io/
# flask：https://palletsprojects.com/p/flask/ ， https://dormousehole.readthedocs.io/en/latest/

# 启动： env FLASK_APP=get_rec.py flask run

# curl "http://127.0.0.1:5000/rec/personal?uid=2591649"
# curl "http://127.0.0.1:5000/rec/sim?vid=25"
# curl "http://127.0.0.1:5000/rec/hot"
