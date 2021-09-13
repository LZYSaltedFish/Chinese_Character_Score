from stroke_extraction import *
from aesthetic_score import *
import pickle

def get_char_id(character, dict_path):
    with open(dict_path, 'rb') as f:
        dict1 = pickle.load(f)
        return str(dict1[character]).rjust(4, '0')

if __name__ == '__main__':
    video_path = "./test_video/fixed5.mp4"
    STROKE_NUM = 7
    strokes_list = get_strokes(video_path, 4, 5, STROKE_NUM, False, False, False)
    print(len(strokes_list))

    dict_path = './utils/char_dict'
    char_id = get_char_id('李', dict_path)
    print(char_id)
    score = aesthetic_score(strokes_list, char_id)
    print(score)