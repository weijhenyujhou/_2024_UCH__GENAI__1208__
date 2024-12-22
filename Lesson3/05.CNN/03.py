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
category=2
Y2 = tf.keras.utils.to_categorical(Y, category)

# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3),
                 padding="same",
                 activation='relu',
                 input_shape=(3,3,1)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=10,
    activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=category,
    activation=tf.nn.softmax ))

model.summary()

model.compile(optimizer='adam',
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=['accuracy'])
model.fit(X2,Y2,
          epochs=100)

#測試
score = model.evaluate(X2,Y2)
print("score:",score)

predict = model.predict(X2)
print("Ans:",np.argmax(predict[0]),np.argmax(predict[1]))

# predict2 = model.predict_classes(X2)
predict2=np.argmax(predict, axis=1)
print("predict_classes:",predict2)
print("y_test",Y)
