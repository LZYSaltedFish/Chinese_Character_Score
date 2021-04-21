import cv2
import numpy as np
from matplotlib import pyplot as plt

def gravity_core(img):  # 计算汉字的重心特征向量
    h, w = img.shape
    x_axis = 0
    y_axis = 0
    total = (h*w) - np.sum(img[:, :])
    for row in range(h):
        y_axis += row * (w - np.sum(img[row, :]))
        # print(w-np.sum(img[row, :]))
    Gy = y_axis / total
    # print('y_axis = ', y_axis)
    # print('total = ', total)
    # print('Gy = ', Gy)
    for col in range(w):
        x_axis += col * (h - np.sum(img[:, col]))
    Gx = x_axis / total

    return [Gx/w, Gy/h], [Gx, Gy]


def grid_vector(img, n_divide = 3):   # 计算图像的网格特征向量
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
            black_num = grid_area - np.sum(img[up_border:up_border+spanY, left_border:left_border+spanX])
            grid_vector.append(black_num / grid_area)
            left_border += spanX
        up_border += spanY

    return grid_vector

def cosine_similarity(vec1, vec2):
    sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
    return sim

def aesthetic_score(img, pattern):    # 计算图像和模板的特征余弦距离，得到美感评分
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pattern_gray = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gray, 127, 1, cv2.THRESH_BINARY)
    _, pattern_bin = cv2.threshold(pattern_gray, 127, 1, cv2.THRESH_BINARY)

    # 计算重心特征向量
    gravity_img, gShow_img = gravity_core(img_bin)
    gravity_pattern, gShow_pattern = gravity_core(pattern_bin)

    # 绘图展示
    plt.subplot(221), plt.imshow(img, 'brg'), plt.title('Origin Image')
    plt.subplot(222), plt.imshow(pattern, 'brg'), plt.title('Origin Pattern')
    plt.subplot(223), plt.imshow(img_bin, 'gray'), plt.title('Binary Image')
    plt.scatter(gShow_img[1], gShow_img[0], marker='o')
    plt.subplot(224), plt.imshow(pattern_bin, 'gray'), plt.title('Binary Pattern')
    plt.scatter(gShow_pattern[1], gShow_pattern[0], marker='o')
    plt.show()
    
    # 计算网格特征向量
    grid_img = grid_vector(img_bin)
    grid_pattern = grid_vector(pattern_bin)

    print('[LOG] Gravity Core Vector | Test Image | ', gravity_img)
    print('[LOG] Gravity Core Vector | Pattern | ', gravity_pattern)
    print('[LOG] Grid Vector | Test Image | ', grid_img)
    print('[LOG] Grid Vector | Pattern | ', grid_pattern)

    # 分别计算 重心特征 和 网格特征 的余弦相似度
    gravity_sim = cosine_similarity(gravity_img, gravity_pattern)
    grid_sim = cosine_similarity(grid_img, grid_pattern)

    # 加权平均得到最终得分
    final_score = gravity_sim * 0.3 + grid_sim * 0.7
    print('[LOG] Aesthetic Score | ', final_score)
    return final_score

if __name__ == '__main__':
    img_path = './test_img/normal/splited.jpg'
    pattern_path = './pattern/square_00001.jpg'

    img = cv2.imread(img_path)
    pattern = cv2.imread(pattern_path)
    score = aesthetic_score(img, pattern)