#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "柯博文老師 Powen Ko, www.powenko.com"

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#from tensorflow.examples.tutorials.mnist
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
import tensorflow as tf

# 影像的類別數目
num_classes = 10

# 輸入的手寫影像解析度
img_rows= 28
img_cols= 28

# 載入資料（將資料打散，放入 train 與 test 資料集）
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()



print('x_train before reshape:', x_train.shape)  # (60000張, 28高, 28寬)
# 將原始資料轉為正確的影像排列方式
dim=img_rows*img_cols*1
x_train = x_train.reshape(x_train.shape[0], dim)
x_test = x_test.reshape(x_test.shape[0], dim)
print('x_train after reshape:', x_train.shape)   #  (60000, 784特徵數)

# 標準化輸入資料
print('x_train before div 255:',x_train[0][180:195])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train before div 255 ', x_train[0][180:195])


print('y_train shape:', y_train.shape)
print(y_train[:10])
# 將數字轉為 One-hot 向量
category=10
y_train2 = tf.keras.utils.to_categorical(y_train, category)
y_test2 = tf.keras.utils.to_categorical(y_test, category)
print("y_train2 to_categorical shape=",y_train2.shape)     #輸出 (60000, 10)
print(y_train2[:10])


# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=784,
    activation=tf.nn.relu,
    input_dim=dim))  # 784=28*28
model.add(tf.keras.layers.Dense(units=392,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=196,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=98,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=50,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=30,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=20,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=category,
    activation=tf.nn.softmax ))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss = tf.keras.losses.categorical_crossentropy,
    metrics = ['accuracy'])
# 設定模型的 Loss 函數、Optimizer 以及用來判斷模型好壞的依據（metrics）


# 顯示模型
model.summary()

import os
if os.path.exists("model.weights.h5"): 
    # 讀取模型權重
    model.load_weights("model.weights.h5")           # 從 model.weights.h5 檔案讀取模型權重
else:
    print("File not exists")



# 添加回呼函數保存最佳權重
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="model.weights.h5",  # 保存的權重檔案名
    monitor="val_loss",               # 監控驗證損失
    save_best_only=True,               # 只保存最佳模型   //<---
    save_weights_only=True,            # 只保存權重
    verbose=1                          # 顯示保存資訊
)


# 訓練模型
history=model.fit(x_train, y_train2,    #進行訓練的因和果的資料
          batch_size=1000,                              #設定每次訓練的筆數
          epochs=200,                       #設定訓練的次數，也就是機器學習的次數          
          validation_split=0.2,  # 添加驗證集
          callbacks=[checkpoint_callback],  # 添加回呼函數
          verbose=1)

#測試
score = model.evaluate(x_test, y_test2, batch_size=128)        # 計算測試正確率
print("score:",score)                                                                         #輸出測試正確率
predict = model.predict(x_test)                                                     #取得每一個結果的機率
print("Ans:",np.argmax(predict[0]),np.argmax(predict[1]),np.argmax(predict[2]),np.argmax(predict[3]))                                                                                    #取得預測答案1
# predict2 = model.predict_classes(x_test[:10])                                     #取得預測答案2
predict2 =np.argmax(predict,axis=1)                                                           #取得每一個結果的類別

print("predict_classes:",predict2[:10])                                                  #輸出預測答案2
print("y_test",y_test[:10])                                                                    #實際測試的果



