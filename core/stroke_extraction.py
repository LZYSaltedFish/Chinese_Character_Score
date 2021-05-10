import cv2
import numpy as np
import time
import math
from matplotlib import pyplot as plt
import core

NEW_IMG_WIDTH = 1280
VIDEO_NAME = 'fixed5'
CENTER_THRESH = 10
FILTER_THRESH = 25
STROKE_NUM = 6

def cal_center(img):
    '''
    计算图像中黑色像素的中心点（重心）坐标
    ----------
    :param img: 原图像灰度图
    :return: [np.array] 中心点坐标
    '''
    height, width = img.shape
    row_sum = 0
    col_sum = 0

    total = np.sum(img[:, :]) / 255
    for row in range(height):
        row_sum += np.sum(img[row, :]) * row
    row_sum /= 255
    for col in range(width):
        col_sum += np.sum(img[:, col]) * col
    col_sum /= 255

    return np.array([row_sum/total, col_sum/total])

def change_point_filter(change_point, max_thresh):
    '''
    切换点过滤
    ----------
    :param change_point [list]: 切换点列表
    :param max_thresh [int]: 最大阈值，间隔小于该阈值的帧会被删掉
    :return: [list] 过滤后的切换点列表
    '''
    left_index = []
    prev = -100
    for cp in change_point:
        if cp - prev > max_thresh:
            left_index.append(cp)
            prev = cp

    return left_index

def resize_img(img, rotate, new_w):
    '''
    图像尺寸放缩
    ----------
    :param img [ndarray]: 原图像灰度图
    :param rotate [bool]: 是否旋转
    :param new_w [int]: 放缩后的宽度
    :return: [ndarray] 放缩后的图像
    '''
    if rotate:
        img = np.rot90(img, -1)
    [height, width, pixels] = img.shape
    new_width = new_w if width > new_w else width
    new_height = int(height * new_width / width)
    img = cv2.resize(img, (new_width, new_height), 
        interpolation=cv2.INTER_CUBIC)
    
    return img

def show_strokes(strokes_list, stroke_num):
    '''
    拼接并显示笔画图像
    ----------
    :param strokes_list [list]: 笔画图像列表
    :param stroke_num [int]: 该字包含的笔画数
    '''
    # 选择合适的拼接行数
    rows_num = int(stroke_num**0.5) if (int(stroke_num**0.5) > 0) else 1
    cols_num = int(stroke_num/rows_num)

    # 图像拼接
    joint = np.zeros((1,))
    for i in range(rows_num):
        row_joint = np.zeros((1,))
        for j in range(cols_num):
            if j==0:
                row_joint = strokes_list[i*cols_num+j]
            else:
                row_joint = np.hstack((row_joint, strokes_list[i*cols_num+j]))
        if i==0:
            joint = row_joint
        else:
            joint = np.vstack((joint, row_joint))

    # 图像显示与保存
    cv2.imwrite("./test_video/result/" + VIDEO_NAME + "_strokes2.jpg", joint)
    cv2.namedWindow("strokes", 0)
    cv2.resizeWindow("strokes", int(joint.shape[1]/rows_num), int(joint.shape[0]/rows_num))
    cv2.imshow("strokes", joint)
    cv2.waitKey()
    cv2.destroyAllWindows()

def reserve_max_cnt(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = contours[np.argmax([cv2.contourArea(cnt) for cnt in contours])]

    h, w = img.shape
    result = np.zeros((h,w), dtype=np.uint8)
    for p in cnt:
        result[p[0][1], p[0][0]] = 255
    return result

def get_strokes(video_path, frames_num, frames_interval, 
    stroke_num, show_diff=False, show_change_frame=False, save_video=False):
    '''
    读取视频并提取笔迹
    ----------
    :param video_path [str]: 视频地址
    :param frames_num [int]: 帧差法提取笔迹增长点时使用的连续帧数量，为不小于2的整数
    :param frames_interval [int]: 帧间隔，每隔frames_interval就有一个帧参与计算
    :param stroke_num [int]: 该汉字包含的笔画数
    :param show_diff [bool]: 是否显示笔迹增长视频
    :param show_change_frame [bool]: 是否显示切换点帧图像
    :param save_video [bool]: 是否保存视频
    '''
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter()
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_savepath = "./test_video/result/" + VIDEO_NAME + "_" + str(frames_num) + str(frames_interval) + ".mp4"
        out = cv2.VideoWriter(video_savepath, fourcc, 10, (1280,2160), False)
    pre_frames = [] # 连续的多个帧

    cur_frame = 0
    interval_count = 0
    last_center = np.array([0,0])
    change_point = [0]

    time_start = time.time()

    while(1):
        ret, frame = cap.read() # 读取一帧视频

        if ret:
            cur_frame +=  1
            interval_count += 1
            
            if interval_count >= frames_interval:
                interval_count = 0
                # 缩小图像
                frame = resize_img(frame, True, NEW_IMG_WIDTH)

                # ------------    一、 Canny 边缘检测    -------------------
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edge = cv2.Canny(frame_gray, 120, 250)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel, iterations=1)

                # ------------    二、 获取笔迹增长点    -------------------
                pre_frames.append(edge)
                if len(pre_frames) >= frames_num:

                    # 先两两相与，再两两相减，最后取并集（改进强化法）
                    and_list = []
                    for i in range(frames_num-1):
                        and_list.append(cv2.bitwise_and(pre_frames[i], pre_frames[i+1]))
                    last_diff = np.zeros((1,))
                    for i in range(len(and_list)-1):
                        diff = cv2.subtract(and_list[i+1], and_list[i])
                        if i==1:
                            last_diff = diff
                        else:
                            last_diff = cv2.bitwise_or(last_diff, diff)
                    
                    pre_frames.pop(0)

                    stroke_growth_m2 = last_diff
                    stroke_growth_m2 = cv2.morphologyEx(stroke_growth_m2, cv2.MORPH_OPEN, kernel, iterations=1)

                    # ------------    三、 计算中心点    -------------------
                    # 中心点偏移超过阈值的位置视为切换点，保存帧下标
                    center = cal_center(stroke_growth_m2)
                    if cur_frame > frames_interval:
                        if np.linalg.norm(center - last_center) >= CENTER_THRESH:
                            change_point.append(cur_frame)
                    last_center = center

                    # 【可选】 显示原视频、笔迹边缘、笔迹增长视频
                    if show_diff:
                        joint = np.vstack((frame_gray, edge))
                        joint = np.vstack((joint, stroke_growth_m2))
                        if save_video:
                            out.write(joint)

                        h, w = joint.shape
                        cv2.namedWindow("edge", 0)
                        h = int(h/2)
                        w = int(w/2)
                        cv2.resizeWindow("edge", w, h)
                        cv2.imshow("edge", joint)
                        cv2.waitKey(20)
        else:
            break
                    
    time_end = time.time()
    print('计算中心点用时：', time_end - time_start, "秒")
    change_point = change_point_filter(change_point, FILTER_THRESH)
    if len(change_point) % STROKE_NUM == 0:
        change_point.append(change_point[-1])

    # 【可选】 显示所有被认为是“切换点”的帧图像
    if show_change_frame:
        font = cv2.FONT_HERSHEY_SIMPLEX # 默认字体
        for i in change_point:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            r, f = cap.read()
            if r:
                f = resize_img(f, True, NEW_IMG_WIDTH)
                h, w, _ = f.shape

                f = cv2.putText(f, str(i), (40, 40), font, 1.2, (0,0,255), 2)
                cv2.namedWindow("stroke", 0)
                cv2.resizeWindow("stroke", w, h)
                cv2.imshow("stroke", f)
                cv2.waitKey(2000)


    # ------------    四、 获取各段笔画    -------------------
    if len(change_point) < stroke_num+1:
        print("【ERROR】无法正常检测到笔画，请在不同笔画之间稍稍停顿")
    else:
        print(change_point)
        stroke_perpoint = int(len(change_point) / stroke_num)
        strokes_list = []
        last_frame = np.zeros((1,))
        for i in range(0,len(change_point), stroke_perpoint):
            cap.set(cv2.CAP_PROP_POS_FRAMES, change_point[i])
            r, f = cap.read()
            if r:
                # 缩小图像，边缘检测
                cur_frame = resize_img(f, True, NEW_IMG_WIDTH)
                cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
                edge = cv2.Canny(cur_frame, 120, 250)
                # 一系列开闭运算，提高笔画质量（消除噪声，增强笔画）
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel, iterations=1)
                edge = cv2.morphologyEx(edge, cv2.MORPH_OPEN, kernel, iterations=1)
                edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel, iterations=3)

                # 相邻起始帧与结束帧相减，得到各笔画
                if i!=0:
                    result = cv2.subtract(edge, last_frame)
                    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=3)    # 笔画交叉会导致笔段断开，因此执行闭运算连接断笔
                    result = reserve_max_cnt(result)    # 保留最大连通域
                    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=3)
                    strokes_list.append(result)
                last_frame = edge
        
        # 图像拼接并显示
        show_strokes(strokes_list, stroke_num)

    # out.release()
    cap.release()
    cv2.destroyAllWindows() # 关闭所有窗口
    print(len(change_point))
    print(change_point)

if __name__ == "__main__":
    path = "./test_video/" + VIDEO_NAME + ".mp4"
    time_start = time.time()
    
    get_strokes(path, 4, 5, STROKE_NUM, False, False, False)

    time_end = time.time()
    print('总共用时：', time_end - time_start, "秒")

# TODO MSER+NMS 定位切割
# TODO 与模板笔画重心对比