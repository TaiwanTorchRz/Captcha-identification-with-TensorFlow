import os,random
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def read_img(path):
    img_st = tf.io.read_file(path)
    img_decode = tf.image.decode_jpeg(img_st, channels=3)
    img_decode = tf.image.convert_image_dtype(img_decode, tf.float32)
    img_decode = tf.expand_dims(img_decode,axis=0)
    return img_decode

def read_img_detect(path):
    img_st = tf.io.read_file(path)
    img_decode = tf.image.decode_jpeg(img_st, channels=3)
    img_decode = tf.image.convert_image_dtype(img_decode, tf.float32)
    img_decode = tf.expand_dims(img_decode,axis=0)
    return img_decode

inputs = keras.Input(shape=(17,30,3))
x = layers.Conv2D(64,(3,3),input_shape=(17,30,3),activation='relu')(inputs)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(128,(3,3),input_shape=(17,30,3),activation='relu')(x)
x = layers.Conv2D(256,(3,3),input_shape=(17,30,3),activation='relu')(x)
x = layers.Conv2D(128,(3,3),input_shape=(17,30,3),activation='relu')(x)
#x = layers.Conv2D(64,(3,3),input_shape=(17,30,3),activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10,activation='softmax')(x)
model = keras.Model(inputs,outputs,name = 'model')


model.load_weights('best_model.h5')

while 1:
    os.system('cls')
    file = input('路徑：')#r'C:\Users\cliffsu\Desktop\test.jpg'

    img = read_img(file)

    cropped = tf.image.resize(tf.image.crop_to_bounding_box(img, 0,8,30,17), [17, 30])
    cropped2 = tf.image.resize(tf.image.crop_to_bounding_box(img, 0,24,30,17), [17, 30])
    cropped3 = tf.image.resize(tf.image.crop_to_bounding_box(img, 0,40,30,17), [17, 30])
    cropped4 = tf.image.resize(tf.image.crop_to_bounding_box(img, 0,56,30,17), [17, 30])


    #enc = tf.image.encode_jpeg(cropped)
    #tf.io.write_file('test_1.jpg',enc)

    #a = [0.9,0,0.1]
    #max(a) --> 0.9
    #a.index(max(a)) --> 0
    result1 = list(np.array(model.predict(cropped))[0])
    result1 = str(result1.index(max(result1)))
    result2 = list(np.array(model.predict(cropped2))[0])
    result2 = str(result2.index(max(result2)))
    result3 = list(np.array(model.predict(cropped3))[0])
    result3 = str(result3.index(max(result3)))
    result4 = list(np.array(model.predict(cropped4))[0])
    result4 = str(result4.index(max(result4)))
    os.system('cls')
    print(result1+result2+result3+result4)
    input('\n辨識完成：')
    
