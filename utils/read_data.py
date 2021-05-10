import tensorflow as tf
import pickle
import os

trainDataPath = "E:/毕业设计/数据集/dataset_HWDB1/train"
testDataPath = "E:/毕业设计/数据集/dataset_HWDB1/test"

def read_data(dataPath):
    count = 0
    image_files=[]
    label_arr=[]
    for char in os.listdir(dataPath):
        if count % 10 == 0:
            print("# PROGRESS #", count, " / 3755")
        for image in os.listdir(os.path.join(dataPath,char)):
            image_files.append(os.path.join(dataPath,char,image))
            label_arr.append(int(char))
        count += 1
    images = tf.convert_to_tensor(image_files)
    labels = tf.convert_to_tensor(label_arr)
    return images,labels

images,labels = read_data(trainDataPath)
images_eval,labels_eval = read_data(testDataPath)

def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string,channels=1)
    image_resized = tf.image.resize_images(image_decoded, [80, 80])
    return image_resized, label

def train_input_fn(images,labels):
    print("# READING TRAIN DATASET #")
    dataset = tf.data.Dataset.from_tensor_slices((images,labels))
    dataset = dataset.shuffle(buffer_size=905000).repeat()
    dataset = dataset.map(_parse_function).batch(512)
    print("# READ TRAIN DATASET DONE #")
    return dataset.make_one_shot_iterator().get_next()

def eval_input_fn(images_eval, labels_eval):
    print("# READING EVALUATE DATASET #")
    dataset = tf.data.Dataset.from_tensor_slices((images_eval, labels_eval))
    dataset = dataset.shuffle(buffer_size=230000)
    dataset = dataset.map(_parse_function).batch(512)
    print("# READ EVALUETE DATASET DONE #")
    return dataset.make_one_shot_iterator().get_next()