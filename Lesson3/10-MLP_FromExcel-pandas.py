#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "柯博文老師 Powen Ko, www.powenko.com"
# pip install pandas numpy


#from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf


import pandas as pd #讀取資料
import numpy as np


# Read the Excel file into a Pandas DataFrame
df = pd.read_excel("weather.xls")

# 选择DataFrame的第11和第12,4,9列，使用iloc索引
selected_columns = df.iloc[1:, [11, 12,4,9]]
# 转换选定的列为NumPy数组
X = selected_columns.values            #2D

selected_columns = df.iloc[1:, [20]]
# 转换选定的列为NumPy数组
Y = selected_columns.values.flatten()  # 1D




category=2
dim=X.shape[1]
x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size=0.2)
y_train2=tf.keras.utils.to_categorical(y_train, num_classes=(category))
y_test2=tf.keras.utils.to_categorical(y_test, num_classes=(category))

print("x_train[:4]",x_train[:4])
print("y_train[:4]",y_train[:4])
print("y_train2[:4]",y_train2[:4])

# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100,
    activation=tf.nn.relu,
    input_dim=dim))
model.add(tf.keras.layers.Dense(units=100,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=category,
    activation=tf.nn.softmax ))
model.compile(optimizer='adam',
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=['accuracy'])
model.fit(x_train, y_train2,
          epochs=2000,
          batch_size=64)

#測試
model.summary()

score = model.evaluate(x_test, y_test2, batch_size=64)
print("score:",score)

predict2 = model.predict_classes(x_test)
print("predict_classes:",predict2)
print("y_test",y_test[:])


