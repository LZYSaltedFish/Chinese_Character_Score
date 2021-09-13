import os
import pickle
import numpy as np
import cv2

img_path = "E:/Character_Extraction/Chinese_Character_Extraction/pattern/"
feat_path = "./stroke_pattern/"
dict_path = "./utils/char_dict"

def stroke_extract(img_path):
    '''
    提取模板的单笔画二值图
    ----------
    :param img_path [str]: 笔顺序列图像路径
    :return: [list] 笔画图像列表，字迹为白色(255)，背景为黑色(0)
    '''
    src_img = []
    strokes = []

    for f in os.listdir(img_path):
        file_path = os.path.join(img_path, f)
        if not os.path.isdir(file_path):
            # print(file_path)
            img = cv2.imread(file_path)
            if not img is None:
                stroke = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                stroke = cv2.bitwise_not(stroke)
                _, stroke = cv2.threshold(stroke, 100, 255, cv2.THRESH_BINARY)
                src_img.append(stroke)
            else:
                print("未能读取图片")

    for i in range(len(src_img)):
        stroke = np.zeros((1,))
        if i==0:
            # stroke = cv2.bitwise_not(src_img[i])
            stroke = src_img[i]
        else:
            stroke = cv2.subtract(src_img[i], src_img[i-1])
            # stroke = cv2.bitwise_not(stroke)
        strokes.append(stroke)
    return strokes

def character_extract(img_path):    # 提取出整个汉字二值图
    src_img = []
    for f in os.listdir(img_path):
        file_path = os.path.join(img_path, f)
        if not os.path.isdir(file_path):
            # print(file_path)
            img = cv2.imread(file_path)
            if not img is None:
                src_img.append(img)
            else:
                print("未能读取图片")
    if len(src_img):
        result_img = src_img[-1]
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
        result_img = cv2.bitwise_not(result_img)
        _, result_img = cv2.threshold(result_img, 100, 255, cv2.THRESH_BINARY)
        return result_img
    else:
        print("文件夹为空")
        return None

def gravity_core(img):
    '''
    计算笔迹的重心特征
    ----------
    :param img [ndarray]: 二值图像，笔迹为白色(255)，背景为黑色(0)
    :return: [list] 重心在图像中的相对位置；[list] 重心在图像中的绝对坐标
    '''
    h, w = img.shape
    x_axis = 0
    y_axis = 0
    total = np.sum(img[:, :])   # 所有白色点之和
    for row in range(h):
        y_axis += row * np.sum(img[row, :]) # 行数 * 该行白色点之和
    Gy = y_axis / total # 求得 y 方向上的重心位置，即重心所在行数
    for col in range(w):    # x 方向同理
        x_axis += col * np.sum(img[:, col])
    Gx = x_axis / total

    return [Gx/w, Gy/h], [Gx, Gy]


def grid_vector(img, n_divide = 3):
    '''
    计算笔迹的网格特征
    ----------
    :param img [ndarray]: 二值图像，笔迹为白色(255)，背景为黑色(0)
    :param n_divide [int]: 每条边上的等分数
    :return: [list] 网格特征向量
    '''
    h, w = img.shape
    spanX = int(w/n_divide)
    spanY = int(h/n_divide)
    grid_sum = spanX * spanY
    left_border = 0
    up_border = 0

    grid_vector = []
    for row in range(n_divide):
        left_border = 0
        for col in range(n_divide):
            white_sum = np.sum(img[up_border:up_border+spanY, left_border:left_border+spanX])
            grid_vector.append(white_sum / 255 / grid_sum)
            left_border += spanX
        up_border += spanY

    return grid_vector

if __name__ == '__main__':
    # with open('./utils/stroke_feature_dict', 'wb+') as f:
    #     patt_feat = {}

    #     for dir in os.listdir(img_path):    # 遍历目录下的每一个模板字
    #         print("正在处理", dir, "...")
    #         file_path = os.path.join(img_path, dir)
    #         if os.path.isdir(file_path):
    #             stroke_img = stroke_extract(file_path)  # 提取出单笔画二值图

    #             feat_list = []  # 对每个笔画计算两类特征
    #             for img in stroke_img:
    #                 gvt_rate, gvt_pos = gravity_core(img)
    #                 grid = grid_vector(img)
    #                 feat_list.append([gvt_rate, grid])
                
    #             patt_feat.update({dir : feat_list})
        
    #     pickle.dump(patt_feat, f)   # 模板笔画特征持久化
    #     print("所有模板特征处理完成！")


    # with open('./utils/character_feature_dict', 'wb+') as f:
    #     patt_feat = {}

    #     for dir in os.listdir(img_path):
    #         print("正在处理", dir, "...")
    #         file_path = os.path.join(img_path, dir)
    #         if os.path.isdir(file_path):
    #             char_img = character_extract(file_path)

    #             feat_list = []
    #             if not char_img is None:
    #                 gvt_rate, _ = gravity_core(char_img)
    #                 grid = grid_vector(char_img)
    #                 feat_list = [gvt_rate, grid]
    #             else:
    #                 feat_list = []

    #             patt_feat.update({dir : feat_list})

    #     pickle.dump(patt_feat, f)
    #     print("所有模板整字特征处理完毕！")

    with open('./utils/character_feature_dict', 'rb') as f:
        dict = pickle.load(f)
        print(dict['0032'][0])
        print(dict['0032'][1])