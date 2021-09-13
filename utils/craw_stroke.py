import pickle
import os
from urllib import request
# import importlib, sys
# encoding = 'utf-8'
# importlib.reload(sys)
import json
import re

stroke_type = {'点':0,'横':1,'竖':2,'撇':3,'捺':4,'竖弯':5,'竖弯钩':6,'横折':7,
        '横折弯钩':8,'竖钩':9,'横折钩':10,'竖折折钩':11,'竖折':12,'提':13,'撇点':14,
        '竖提':15,'横折提':16,'弯钩':17,'斜钩':18,'卧钩':19,'横钩':20,'横撇弯钩':21,
        '横折折折钩':22,'横折弯':23,'撇折':24,'横撇':25,'横折折撇':26,'竖折撇':27,
        '竖撇':28}

order_lost_log = "./pattern/order_lost.txt"
img_lost_log = "./pattern/img_lost.txt"
not_fount_log = "./pattern/not_found.txt"

def get_zh_list():
    zh_character = []
    zh_codes = []
    with open('./pattern/char_dict','br') as f:
        dict = pickle.load(f)
        for key, _ in dict.items():
            byte_str = str(key.encode('utf-8'))
            zh_code = "%" + byte_str[4:6] + "%" + byte_str[8:10] + "%" + byte_str[12:14]
            zh_character.append(key)
            zh_codes.append(zh_code)
    return zh_character, zh_codes

# 笔顺序列 和 笔顺图片 目录位置
# detail->resultData->ciyuData->bishunhz(list)/bstianzige(Array)
def craw_stroke(url, zh_char):
    start = "window\.__INITIAL_STATE__="
    end = ";\(function\(\)\{var s;\(s=document\.currentScript\|\|document\.scripts\[document\.scripts\.length\-1\]\)"
    stroke_order = []
    order_img = []

    req = request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36')
    with request.urlopen(req) as f:
        # for k, v in f.getheaders():
        #     print('%s: %s' % (k, v))
        data = f.read().decode('utf-8')
        script_result = re.findall(start + "(.*?)" + end, data)[0]
        json_result = json.loads(script_result)
        if 'ciyuData' in json_result['detail']['resultData']:
            ciyuData = json_result['detail']['resultData']['ciyuData']
            if 'bishunhz' in ciyuData:
                stroke_order = ciyuData['bishunhz']
            else:
                with open(order_lost_log, mode="a+") as f:
                    f.write(zh_char)
                print("【ERROR】", zh_char, "笔顺信息未能找到")
            if 'bstianzige' in ciyuData:
                for img_json in ciyuData['bstianzige']:
                    order_img.append(img_json['img'])
            else:
                with open(img_lost_log, mode="a+") as f:
                    f.write(zh_char)
                print("【ERROR】", zh_char, "图片信息未能找到")
        else:
            with open(not_fount_log, mode="a+") as f:
                f.write(zh_char)
            print("【ERROR】", zh_char, "没有找到相关数据")
        

    return stroke_order, order_img

def save_order(path, zh_char, order_list):
    order_str = ""
    for stroke in order_list:
        candidate = stroke.split("/")
        found = False
        for i in candidate:
            if i in stroke_type:
                order_str = order_str + str(stroke_type[i]) + " "
                found = True
                break
        if not found:
            print("【ERROR】", candidate, "在笔顺类型中不存在")
            order_str = order_str + "ERROR "
        
    with open(path, mode="a+") as f:
        f.write(zh_char + " " + order_str + "\n")

def get_checkpoint(path):
    start_index = 0
    with open(path, mode="r+") as f:
        content = f.read()
        if content != None:
            start_index = int(content) + 1
    return start_index

def update_checkpoint(path, index):
    with open(path, mode="w+") as f:
        f.write(str(index))

if __name__ == "__main__":
    url = "https://hanyu.sogou.com/result?query="
    checkpoint_path = "./pattern/ckpt.txt"
    order_path = "./pattern/writing_order.txt"

    # 读取存档点，以便断点续爬
    start_index = get_checkpoint(checkpoint_path)

    zh_character, zh_codes = get_zh_list()
    for index in range(start_index, len(zh_character)):
        zh_char = zh_character[index]
        zh_code = zh_codes[index]

        req_url = url + zh_code
        order_zh, order_img = craw_stroke(req_url, zh_char)
        
        img_save_path = os.getcwd() + "/pattern/" + zh_char + '/'
        if not os.path.exists(img_save_path):
            os.mkdir(img_save_path)

        # 追加写入笔顺序列
        save_order(order_path, zh_char, order_zh)

        # 保存笔顺图片
        count = 0
        for img_url in order_img:
            with open(img_save_path + str(count) + ".jpg", "wb") as f:
                f.write((request.urlopen("https:" + img_url)).read())
                count += 1
        print(index, "【" + zh_char + "】笔顺与图片保存完毕")

        # 更新存档点
        update_checkpoint(checkpoint_path, index)