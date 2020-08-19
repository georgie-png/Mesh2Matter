
from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
from matplotlib import pyplot as plt


IMG_WIDTH = 512
IMG_HEIGHT = 512

def set_WXH(width, height):
    
    IMG_WIDTH  = width
    IMG_HEIGHT = height
    
    

def load(image_file):
    
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)
  image = tf.cast(image, tf.float32)

  return image

def resize(input_image, height, width):
  input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  
  return input_image


def normalize(input_image):
  input_image = (input_image / 127.5) - 1
  
  return input_image

def load_image_test(image_file):
  input_image = load(image_file)
  input_image = resize(input_image,IMG_HEIGHT, IMG_WIDTH)
  print(input_image[200])
  input_image = normalize(input_image)
  print(input_image[200])
  
    
  return input_image