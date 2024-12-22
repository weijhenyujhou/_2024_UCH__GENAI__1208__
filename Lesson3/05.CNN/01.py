#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "柯博文老師 Powen Ko, www.powenko.com"
import tensorflow as tf
import time

import numpy as np
import matplotlib.pyplot as plt

circle1=np.array([[1,1,1],
                 [1,0,1],
                 [1,1,1]])

cross1=np.array([[1,0,1],
                 [0,1,0],
                 [1,0,1]])


X = np.array([circle1,cross1])
Y = np.array([0,1])
print(X.shape)
X2 = X.reshape(2,3,3,1)
""" CNN
2 二張圖片
3 筆 列
3    行
1 顏色
"""
"""
2張圖片
1024x768
彩色 RGB
X.reshape(2,768,1024,3)
"""


print(X2.shape)
category=2
Y2 = tf.keras.utils.to_categorical(Y, category)

# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=3, #一張照片在切割成3張 建議256以上
                 kernel_size=(3, 3), #filter 3x3
                 padding="same",
                 activation='relu', #圖片激活函數 大多是用relu 最小值為0 最大值無限大 
                 input_shape=(3,3,1))) #實際X的資料(2,3,3,1) 刪除筆數填入
model.add(tf.keras.layers.Flatten()) # CNN資料轉成MLP
model.add(tf.keras.layers.Dense(units=10,
    activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=category,
    activation=tf.nn.softmax ))

model.summary()
