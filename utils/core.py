from hccr import *
from text_extraction import *
from aesthetic_score import *

img_seq = ['medium_score', 'low_score']
predict_path = './pred_img'

if __name__ == '__main__':
    for i in img_seq:
        img_path = './test_img/aesthetic/' + i + '.jpg'
        # img_path = './test_img/blocked/blocked3.jpg'
        split_path = predict_path + '/00000/splited.jpg'

        # 通过 MSER 和 NMS 算法进行文本定位和文本切割
        img_extraction = cv2.imread(img_path)
        candidate_boxes = MSER_NMS(img_extraction)
        box = Central_Box(img_extraction, candidate_boxes)
        split_img = img_extraction[box[1]:box[3], box[0]:box[2]]
        cv2.imwrite(split_path, split_img)

        # 载入ckpt模型，进行手写汉字识别
        classes = predict_img(predict_path)
        pattern_path = './pattern/' + str(classes[0]).rjust(5, '0') + '/pattern.jpg'
        img_aesthetic = cv2.imread(split_path)
        pattern = cv2.imread(pattern_path)
        score = aesthetic_score(img_aesthetic, pattern)