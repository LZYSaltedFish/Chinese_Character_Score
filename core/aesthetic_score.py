import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt
import pickle
from text_location import *

char_feat_path = './utils/character_feature_dict'
stroke_feat_path = './utils/stroke_feature_dict'

def gravity_core(img):  # 计算汉字的重心特征向量
    '''
    计算汉字的重心特征向量
    ----------
    :param img [ndarray]: 笔画二值图像，笔画为白色
    :return [list]: 归一化的重心坐标
    :return [list]: 绝对重心坐标
    '''
    h, w = img.shape
    x_axis = 0
    y_axis = 0
    total = np.sum(img[:, :])
    for row in range(h):
        y_axis += row * np.sum(img[row, :])
    Gy = y_axis / total
    for col in range(w):
        x_axis += col * np.sum(img[:, col])
    Gx = x_axis / total

    return [Gx/w, Gy/h], [Gx, Gy]


def grid_vector(img, n_divide = 3):   # 计算图像的网格特征向量
    '''
    计算汉字的网格特征向量
    ----------
    :param img [ndarray]: 笔画二值图像，笔画为白色
    :param n_divide [int]: 
    :return [list]: 
    '''
    w, h = img.shape
    spanX = int(w/n_divide)
    spanY = int(h/n_divide)
    grid_area = spanX * spanY
    left_border = 0
    up_border = 0

    grid_vector = []
    for row in range(n_divide):
        left_border = 0
        for col in range(n_divide):
            white_num = np.sum(img[up_border:up_border+spanY, left_border:left_border+spanX])
            grid_vector.append(white_num / grid_area)
            left_border += spanX
        up_border += spanY

    return grid_vector

def cosine_similarity(vec1, vec2):
    '''
    计算向量余弦相似度
    ----------
    :param vec1 [list]: 向量1，元素应为可计算数值
    :param vec2 [list]: 向量2，元素应为可计算数值
    :return [float]: 相似性数值，取值范围为[-1,1]
    '''
    sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
    return sim

def stroke_score(stroke_list, char_id, dict_path, gravity_ratio=0.2):
    '''
    根据单笔画分别评分
    ----------
    :param img [ndarray]: 笔画二值图像列表，笔画为白色
    :param char_id [str]: 汉字编号
    :param dict_path [str]: 模板特征字典文件路径
    :param gravity_ratio [float]: 重心特征评分所占比例，为0到1之间的值
    :return [float]: 笔画评分
    '''
    if gravity_ratio>1.0 or gravity_ratio<0:
        print("单笔画重心评分比例错误，应为0到1之间的值")
        gravity_ratio = 0.2
    grid_ratio = 1.0 - gravity_ratio
    pattern_feat = []
    score = []

    with open(dict_path, 'rb') as f:
        dict = pickle.load(f)
        pattern_feat = dict[char_id]
        if len(pattern_feat):
            if len(pattern_feat)==len(stroke_list):
                for i in range(len(pattern_feat)):  # 对每个笔画分别计算特征向量，计算评分
                    gravity, _ = gravity_core(stroke_list[i])
                    grid = grid_vector(stroke_list[i])
                    gravity_similarity = cosine_similarity(gravity, pattern_feat[i][0])
                    grid_similarity = cosine_similarity(grid, pattern_feat[i][1])

                    # test
                    print("单笔画重心评分", gravity_similarity)
                    print("单笔画网格评分", grid_similarity)
                    print()

                    score.append(gravity_similarity * gravity_ratio + grid_similarity * grid_ratio)
                
                return np.mean(score)
            else:
                print("【ERROR】 笔画图片数量与模板不一致！")
                sys.exit(0)
                return -1
        else:
            print("模板数据丢失！")
            return -1

def char_score(char_img, char_id, dict_path, gravity_ratio=0.2):
    '''
    对整字进行评分
    ----------
    :param img [ndarray]: 汉字二值图像，文字部分为白色
    :param char_id [str]: 汉字编号
    :param dict_path [str]: 模板特征字典文件路径
    :param gravity_ratio [float]: 重心特征评分所占比例，为0到1之间的值
    :return [float]: 整字评分
    '''
    if gravity_ratio>1.0 or gravity_ratio<0:
        print("单笔画重心评分比例错误，应为0到1之间的值")
        gravity_ratio = 0.2
    grid_ratio = 1.0 - gravity_ratio
    
    with open(dict_path, 'rb') as f:
        dict = pickle.load(f)
        pattern_feat = dict[char_id]
        if len(pattern_feat):
            gravity, _ = gravity_core(char_img)
            grid = grid_vector(char_img)
            gravity_similarity = cosine_similarity(gravity, pattern_feat[0])
            grid_similarity = cosine_similarity(grid, pattern_feat[1])
            
            # test
            print("整字重心评分", gravity_similarity)
            print("整字网格评分", grid_similarity)
            print()

            return (gravity_similarity * gravity_ratio + grid_similarity * grid_ratio)
        else:
            print("模板数据丢失！")
            return -1

def aesthetic_score(stroke_list, char_id):
    '''
    计算汉字的书写质量综合评分
    ----------
    :param stroke_list [list]: 笔画二值图像列表，笔画为白色
    :return [float]: 书写评分
    '''
    # 计算单笔画评分
    strokes_score = stroke_score(stroke_list, char_id, stroke_feat_path)
    # 计算整字评分
    char_img = img_merge(stroke_list)
    character_score = char_score(char_img, char_id, char_feat_path)

    # 权值相加得到综合评分
    stroke_ratio = 0.5
    char_ratio = 1.0 - stroke_ratio
    final_score = strokes_score * stroke_ratio + character_score * char_ratio

    return final_score

if __name__ == '__main__':
    stroke_list = []
    for i in range(4):
        img = cv2.imread("./test_video/strokes/" + str(i) + ".jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        stroke_list.append(img)
    stroke_list = text_location(stroke_list)    # 文本定位与切割

    score = aesthetic_score(stroke_list, '0750')
    print(score)