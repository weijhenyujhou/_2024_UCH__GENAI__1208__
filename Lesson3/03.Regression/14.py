#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "柯博文老師 Powen Ko, www.powenko.com"
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time

import pandas as pd

from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.shape)
print(y_train.shape)

classes = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
data = pd.DataFrame(x_train, columns=classes)
print(data.head())

data['MEDV'] = pd.Series(data=y_train)
print(data.head())
print(data.describe())  # get some basic stats on the dataset

import seaborn as sns

from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(320, activation='tanh', input_shape=[x_train.shape[1]]))
model.add(tf.keras.layers.Dense(640, activation='tanh'))
model.add(tf.keras.layers.Dense(640, activation='tanh'))
model.add(tf.keras.layers.Dense(1))
"""
try:
    with open('model2.h5', 'r') as load_weights:
        # 讀取模型權重
        model.load_weights("model2.h5")
except IOError:
    print("File not exists")
"""

learning_rate = 0.0001
opt1 = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
model.compile(loss='mse', optimizer=opt1, metrics=['mae'])
# 每隔20次 儲存一次模型
history1 = []
for step in range(40000):
    cost = model.train_on_batch(x_train, y_train)
    if step % 20 == 0:
        print("step{}   train cost{}".format(step, cost))
        # 保存模型架構
        with open("model_Boston.json", "w") as json_file:
            json_file.write(model.to_json())
        # 保存模型權重
        model.save_weights("mymodel_Boston.weights.h5")



# testing
print("start testing")
cost = model.evaluate(x_test, y_test)
print("test cost: {}".format(cost))

Y_pred2 = model.predict(x_test)  # Y predict

print(Y_pred2[:10])
print(y_test[:10])
