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

import os
if os.path.exists("model2.weights.h5"):
    """
    with open('model2.h5', 'r') as load_weights:
        # 讀取模型權重
        model.load_weights("model2.h5")
    """
    # 讀取模型權重
    model.load_weights("model2.weights.h5")           # 從 model.weights.h5 檔案讀取模型權重
else:
    print("File not exists")
learning_rate = 0.00001
opt1 = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
model.compile(
    #loss='mse'    
    loss=tf.keras.losses.MeanSquaredError(),  # 明確指定 MSE
              optimizer=opt1, metrics=['mae'])

# 添加回呼函數保存最佳權重
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="best_model.weights.h5",  # 保存的權重檔案名
    monitor="val_loss",               # 監控驗證損失
    save_best_only=True,               # 只保存最佳模型   //<---
    save_weights_only=True,            # 只保存權重
    verbose=1                          # 顯示保存資訊
)

# 訓練模型
history = model.fit(
    x_train, y_train,
    epochs=400,
    batch_size=200,
    validation_split=0.2,  # 添加驗證集
    callbacks=[checkpoint_callback]  # 添加回呼函數
)

# testing
print("start testing")
cost = model.evaluate(x_test, y_test)
print("test cost: {}".format(cost))

Y_pred2 = model.predict(x_test)  # Y predict

print(Y_pred2[:10])
print(y_test[:10])

import matplotlib.pyplot as plt

print(history.history.keys())
plt.plot(history.history['mae']) # mean_absolute_error
plt.title('Boston house price')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train mae'], loc='upper right')
plt.show()

"""
#保存模型架構
with open("model.json", "w") as json_file:
   json_file.write(model.to_json())
#保存模型權重
model.save_weights("model.h5")
"""
# 保存模型架構與權重
model.save('model2.h5')                           # 保存完整模型到檔案 model.h5

# 保存模型權重
model.save_weights("model2.weights.h5")           # 保存模型的權重到檔案 model.weights.h5


