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
        print("# PROGRESS #", count, " / 10")
        for image in os.listdir(os.path.join(dataPath,char)):
            image_files.append(os.path.join(dataPath,char,image))
            label_arr.append(int(char))
        count += 1
        if count==10:
            break
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
    dataset = dataset.shuffle(buffer_size=10000).repeat()
    dataset = dataset.map(_parse_function).batch(10)
    print("# READ TRAIN DATASET DONE #")
    return dataset.make_one_shot_iterator().get_next()

def eval_input_fn(images_eval, labels_eval):
    print("# READING EVALUATE DATASET #")
    dataset = tf.data.Dataset.from_tensor_slices((images_eval, labels_eval))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(_parse_function).batch(10)
    print("# READ EVALUETE DATASET DONE #")
    return dataset.make_one_shot_iterator().get_next()

def cnn_model_fn(features, labels, mode):
    input_layer =tf.reshape(features,[-1,80,80,1])
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      name="conv1")
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2,name="pool1")
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      name="conv2")
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],strides=2,name="pool2")
    conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    pool3_flat = tf.reshape(pool3, [-1, 10 * 10 * 128])
    dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10,name="logits")
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      learning_rate = 0.0001
      decay_rate = 0.001**(tf.train.get_global_step() / 200000)
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate*decay_rate)
      train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
      logging_hook = tf.train.LoggingTensorHook({"loss":loss, "steps":tf.train.get_global_step()}, every_n_iter=10)
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_chief_hooks=[logging_hook])

  # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

chart_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="E:/CNN_Model/Chinese_Character_Recognition_CNN/test_model2")

def predict_img(img_path):
  f=open('E:/毕业设计/数据集/dataset_HWDB1/char_dict','br')
  dict = pickle.load(f)
  images_pred,lables_pred = read_data(img_path)
  index_to_char = {value:key for key,value in dict.items()}
  classes = []
  for pred in chart_classifier.predict(input_fn=lambda:eval_input_fn(images_pred,lables_pred)):
    print(index_to_char[pred["classes"]])
    classes.append(pred["classes"])
    
  return classes

if __name__ == "__main__":
    # train model
    chart_classifier.train(input_fn=lambda:train_input_fn(images,labels), max_steps=200000)

    # evaluate model
    eval_results=chart_classifier.evaluate(input_fn=lambda:eval_input_fn(images_eval,labels_eval))
    print(eval_results)