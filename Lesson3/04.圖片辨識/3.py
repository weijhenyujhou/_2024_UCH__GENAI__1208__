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

plt.subplot(2,2,1)
plt.imshow(circle1)

circle2=circle1.flatten()
circle2=circle1.reshape([9])
print(circle2)
plt.subplot(2,2,2)
plt.plot(circle2, 'ob',)

cross1=np.array([[1,0,1],
                 [0,1,0],
                 [1,0,1]])
plt.subplot(2,2,3)
plt.imshow(cross1)

cross2=cross1.reshape([9]) #調整到一維
print(cross2)
plt.subplot(2,2,4)
plt.plot(cross2, 'xb',)

#plt.show()


X = np.array([circle2,cross2])
Y = np.array([0,1])


category=2
dim=9

Y2=tf.keras.utils.to_categorical(Y, num_classes=(category))
# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=10,
    activation=tf.nn.relu,
    input_dim=dim))
model.add(tf.keras.layers.Dense(units=10,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=category,
    activation=tf.nn.softmax ))


model.compile(optimizer='adam',
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=['accuracy'])
model.fit(X, Y2,
          epochs=100)

#測試
score = model.evaluate(X,Y2, batch_size=16)
print("score:",score)

predict = model.predict(X)
print("Ans:",np.argmax(predict[0]),np.argmax(predict[1]))

# predict2 = model.predict_classes(X)
predict2 = np.argmax(predict, axis=1)  
print("predict_classes:",predict2)
print("y_test",Y)
