import tensorflow as tf
from face_detect import find_faces
import cv2, sys
import numpy as np
from image_commons import nparray_as_image, draw_with_alpha
image_size = 64
emotions = ['neutral','fear', 'happy', 'sadness', 'surprise']


def _load_emoticons(emotions):
  return [nparray_as_image(cv2.imread('graphics/%s.png' % emotion, -1), mode=None) for emotion in emotions]

def get_faces(path):
  data = []
  frame = cv2.imread(path)
  normalized_faces = find_faces(frame)
  print(normalized_faces)
  for img, (x, y, w, h) in normalized_faces:
    img = cv2.resize(img[0], (image_size, image_size))
    img = np.reshape(img, (image_size*image_size))
    data.append(img)
  return data, normalized_faces

def draw_img(predictions, path, normalized_faces):
  emoticons = _load_emoticons(emotions)
  image = cv2.imread(path)
  for index, (img, (x, y, w, h)) in enumerate(normalized_faces):
    image_to_draw = emoticons[predictions[index]]
    emotion_img = draw_with_alpha(image, image_to_draw, (x, y, w, h))
    cv2.imwrite(str(emotions[predictions[index]]) + '_' + str(index) + '.png', emotion_img)


def get_emotion(sess, data):
  y_test=sess.run(index,feed_dict={x: data})
  y_conf=sess.run(conf,feed_dict={x: data})
  pred = np.asarray(y_test)[0]
  conf_value = y_conf.reshape(-1)[pred]
  print emotions[pred]
  return y_test

if __name__ == '__main__':
  path = sys.argv[1]
  tf.reset_default_graph()
  saver = tf.train.import_meta_graph("models/final.ckpt.meta")
  x = tf.get_default_graph().get_tensor_by_name("x:0")
  logits= tf.get_default_graph().get_tensor_by_name("output/BiasAdd:0")
  index=tf.argmax(tf.nn.softmax(logits),1)
  conf=tf.nn.softmax(logits)
  sess= tf.Session()
  saver.restore(sess, "models/final.ckpt")

  data, normalized_faces = get_faces(path)
  data = np.asarray(data)
  predictions = get_emotion(sess, data)

  draw_img(predictions, path, normalized_faces)
