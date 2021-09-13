import cv2
import numpy as np
from matplotlib import pyplot as plt

def MSER(test_img):  # MSER 方法（最大稳定极值区域）
    '''
    获取候选文本矩形框
    ----------
    :param test_img [ndarray]: 笔画二值图像，笔画为白色
    :return [list]: 矩形框列表，表中元素为矩形框四边坐标，依次为左、上、右、下
    '''
    # img_copy = test_img.copy()
    # img_copy2 = test_img.copy()
    mser = cv2.MSER_create(_min_area=300)
    regions, _ = mser.detectRegions(test_img)  # 获取文本区域
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]  # 绘制文本区域
    # cv2.polylines(img_copy2, hulls, 1, (0, 255, 0))
    # plt.subplot(221), plt.imshow(img_copy2, 'brg')
    # plt.title('MSER polygon')

    # 将不规则多边形框转为矩形框
    keep = []
    for c in hulls:
        x, y, w, h = cv2.boundingRect(c)
        # 筛掉宽高比例过大的方框以及过小的方框
        if (h > w * 1.5) or (w > h * 1.5) or h < 25 or w < 25:
            continue
        keep.append([x, y, x + w, y + h])
        # cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 255, 0), 1)
    # plt.imshow(img_copy, 'gray')
    # plt.title('MSER rectangle')
    # plt.show()
    return keep

def NMS(boxes, overlapThresh=0.2):  # NMS 方法（非极大值抑制）
    '''
    筛除不符合条件的候选框
    ----------
    :param boxes [list]: 候选框列表，元素为矩形框四边坐标，依次为左、上、右、下
    :param overlapThresh [float]: 两框重叠面积最大阈值，取值范围在0-1之间
    :return [list]: 筛除后的候选框列表，元素结构与输入参数相同
    '''
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 按右下坐标排序
    idxs = np.argsort(y2)

    # 开始遍历，并删除重复的框
    while len(idxs) > 0:
        # 将最右下方的框放入pick数组
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # 找剩下的其余框中最大坐标和最小坐标
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # 计算重叠面积占对应框的比例，即 IoU
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]

        # 如果 IoU 大于指定阈值，则删除
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")

def MSER_NMS(test_img):
    '''
    获取文字区域候选框
    ----------
    :param test_img [ndarray]: 笔画二值图像，笔画为白色
    :return [list]: 候选框列表，元素为矩形框四边坐标，依次为左、上、右、下
    '''
    # test_copy = test_img.copy()

    keep = MSER(test_img)
    print("[LOG] after MSER, %d bounding boxes" % (len(keep)))
    keep2 = np.array(keep)

    pick = NMS(keep2, 0.2)
    print("[LOG] after NMS, %d bounding boxes" % (len(pick)))

    # for (startX, startY, endX, endY) in pick:
    #     cv2.rectangle(test_copy, (startX, startY), (endX, endY), (255, 185, 120), 2)
    # plt.imshow(test_copy, 'gray')
    # plt.title('After NMS')
    # plt.show()
    return pick

def Square_Box(box, img_height, img_width):    # 转换为正方形框，将汉字置于正中央，以便后续书写评分
    '''
    矩形框转正方形框
    ----------
    :param box [list]: 候选矩形框四边坐标，依次为左、上、右、下
    :return [list]: 调整后的正方形框四边坐标
    '''
    # 转为正方形框
    width = box[2]-box[0]
    height = box[3]-box[1]
    if width > height:
        offset = (width-height)/2
        box[1] -= offset
        box[3] += offset
    else:
        offset = (height-width)/2
        box[0] -= offset
        box[2] += offset

    # 将文本框边长扩大1/7
    offset = int((box[2]-box[0])/14)
    resize_box = [box[0]-offset, box[1]-offset, box[2]+offset, box[3]+offset]
    if resize_box[0]>0 and resize_box[1]>0 and resize_box[2]<img_width and resize_box[3]<img_height:
        return resize_box
    return box

def Central_Box(test_img, boxes):
    '''
    获取中央候选框
    ----------
    :param test_img [ndarray]: 原图像灰度图，用于效果展示
    :param boxes [list]: 候选框列表，表中元素为矩形框四边坐标，依次为左、上、右、下
    :return [list]: 正方形框四边坐标，依次为左、上、右、下
    '''
    img_copy = test_img.copy()
    h, w = test_img.shape
    centre = (h/2, w/2)
    distance = []
    for box in boxes:
        x = (box[0]+box[2])/2
        y = (box[1]+box[3])/2
        distance.append((centre[0]-x)**2 + (centre[1]-y)**2)
    idx = np.argsort(distance)
    final_box = Square_Box(boxes[idx[0]], h, w)

    # (startX, startY, endX, endY) = Square_Box(boxes[idx[0]], h, w)
    # cv2.rectangle(img_copy, (startX, startY), (endX, endY), (255, 185, 120), 2)
    # plt.imshow(img_copy, 'gray')
    # plt.title('Final Box')
    # plt.show()
    # plt.imshow(img_copy[final_box[1]:final_box[3], final_box[0]:final_box[2]], 'gray')
    # plt.title('Splited Image')
    # plt.show()

    return final_box

def img_merge(img_list):
    '''
    单笔画图像合并
    ----------
    :param img_list [list]: 笔画二值图像列表，笔画为白色
    :return [ndarray]: 合并后的整字二值图像，文字部分为白色
    '''
    if len(img_list):
        result = img_list[0]
        for i in img_list:
            result = cv2.bitwise_or(result, i)
        return result
    return None

def text_location(stroke_list):
    '''
    文本区域定位与切割
    ----------
    :param stroke_list [list]: 笔画二值图像列表，笔画为白色
    :return [list]: 切割后的笔画二值图像列表，笔画为白色
    '''
    # 通过 MSER 和 NMS 进行“文本区域定位”
    merged_img = img_merge(stroke_list)
    candidate_boxes = MSER_NMS(merged_img)
    box = Central_Box(merged_img, candidate_boxes)

    # 通过中心优先算法进行“文本区域切割”
    splited_list = []
    for stroke in stroke_list:
        splited_list.append(stroke[box[1]:box[3], box[0]:box[2]])
    
    return splited_list

if __name__ == '__main__':
    img_list = []
    for n in range(4):
        img = cv2.imread("./test_video/strokes/" + str(n) + ".jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list.append(img)
    char_img = img_merge(img_list)

    if not char_img is None:
        stroke_list = text_location(img_list)
        h, w = stroke_list[0].shape
        cv2.namedWindow("split")
        cv2.resizeWindow("split", w, h)
        for s in stroke_list:
            cv2.imshow("split", s)
            cv2.waitKey(0)