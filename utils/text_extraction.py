import cv2
import numpy as np
from matplotlib import pyplot as plt

def MSER(test_img,g_test):  # MSER 方法（最大稳定极值区域）
    img_copy = test_img.copy()
    img_copy2 = test_img.copy()
    mser = cv2.MSER_create(_min_area=300)
    regions, _ = mser.detectRegions(g_test)  # 获取文本区域
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]  # 绘制文本区域
    cv2.polylines(img_copy2, hulls, 1, (0, 255, 0))
    plt.subplot(221), plt.imshow(img_copy2, 'brg')
    plt.title('MSER polygon')

    # 将不规则多边形框转为矩形框
    keep = []
    for c in hulls:
        x, y, w, h = cv2.boundingRect(c)
        # 筛掉宽高比例过大的方框以及过小的方框
        if (h > w * 1.5) or (w > h * 1.5) or h < 25 or w < 25:
            continue
        keep.append([x, y, x + w, y + h])
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 255, 0), 1)
    plt.subplot(222), plt.imshow(img_copy, 'brg')
    plt.title('MSER rectangle')
    return keep

def NMS(boxes, overlapThresh):  # NMS 方法（非极大值抑制）
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
    test_copy = test_img.copy()
    g_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    keep = MSER(test_img, g_test)
    print("[LOG] after MSER, %d bounding boxes" % (len(keep)))
    keep2 = np.array(keep)

    pick = NMS(keep2, 0.2)
    print("[LOG] after NMS, %d bounding boxes" % (len(pick)))

    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(test_copy, (startX, startY), (endX, endY), (255, 185, 120), 2)
    plt.subplot(223), plt.imshow(test_copy, 'brg')
    plt.title('After NMS')
    return pick

def Square_Box(box):    # 转换为正方形框，将汉字置于正中央，以便后续书写评分
    width = box[2]-box[0]
    height = box[3]-box[1]
    if width > height:
        box[1]-=(width-height)/2
        box[3]+=(width-height)/2
    else:
        box[0]-=(height-width)/2
        box[2]+=(height-width)/2
    return box

def Central_Box(test_img, boxes):
    img_copy = test_img.copy()
    h, w, _ = test_img.shape
    centre = (h/2, w/2)
    distance = []
    for box in boxes:
        x = (box[0]+box[2])/2
        y = (box[1]+box[3])/2
        distance.append((centre[0]-x)**2 + (centre[1]-y)**2)
    idx = np.argsort(distance)
    (startX, startY, endX, endY) = Square_Box(boxes[idx[0]])

    cv2.rectangle(img_copy, (startX, startY), (endX, endY), (255, 185, 120), 2)
    plt.subplot(224), plt.imshow(img_copy, 'brg')
    plt.title('Final Outcome')
    plt.show()

    return boxes[idx[0]]

if __name__ == '__main__':
    # 读取图片，转为灰度图，以便后续文本检测
    img_path = './test_img/normal/normal_single_1.jpg'
    # img_path = './test_img/blocked/blocked3.jpg'
    img = cv2.imread(img_path)

    # 通过 MSER 和 NMS 得到初步候选区域框
    candidate_boxes = MSER_NMS(img)

    # 通过中心优先算法得到最终的文本区域框
    box = Central_Box(img, candidate_boxes)
    split_img = img[box[1]:box[3], box[0]:box[2]]
    cv2.imwrite('./test_img/normal/splited.jpg', split_img)