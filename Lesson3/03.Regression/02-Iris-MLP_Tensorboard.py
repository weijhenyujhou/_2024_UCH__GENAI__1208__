#!/usr/bin/env python                           # 設定為可執行的 Python 腳本
# -*- coding=utf-8 -*-                         # 設定檔案的編碼格式為 UTF-8
__author__ = "柯博文老師 Powen Ko, www.powenko.com" # 作者資訊

from sklearn import datasets                   # 引入 sklearn 套件的資料集模組
from sklearn.model_selection import train_test_split # 引入 sklearn 的資料分割模組
import tensorflow as tf                        # 引入 TensorFlow 套件
import numpy as np                             # 引入 NumPy 套件，用於數值運算
from tensorflow.keras.callbacks import TensorBoard # 引入 TensorBoard 回呼函數
from time import time                          # 引入 time 模組，用於計時

iris = datasets.load_iris()                   # 載入 Iris 資料集

category = 3                                  # 分類數量，Iris 資料集中有 3 種分類
dim = 4                                       # 特徵維度，Iris 資料集有 4 個特徵
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2) # 分割資料為訓練集和測試集，測試集佔 20%
y_train2 = tf.keras.utils.to_categorical(y_train, num_classes=(category)) # 將訓練標籤轉為 One-Hot 編碼
y_test2 = tf.keras.utils.to_categorical(y_test, num_classes=(category))   # 將測試標籤轉為 One-Hot 編碼

print("x_train[:4]", x_train[:4])             # 印出訓練集資料的前 4 筆
print("y_train[:4]", y_train[:4])             # 印出訓練集標籤的前 4 筆
print("y_train2[:4]", y_train2[:4])           # 印出轉換後的 One-Hot 編碼標籤前 4 筆

# 建立模型
model = tf.keras.models.Sequential()          # 建立 Sequential 模型
model.add(tf.keras.layers.Dense(units=10,     # 新增一個全連接層，10 個神經元
    activation=tf.nn.relu,                    # 激活函數使用 ReLU
    input_dim=dim))                           # 輸入層的特徵數目為 4
model.add(tf.keras.layers.Dense(units=10,     # 新增第二個全連接層，10 個神經元
    activation=tf.nn.relu))                   # 激活函數使用 ReLU
model.add(tf.keras.layers.Dense(units=category, # 新增輸出層，單元數等於分類數
    activation=tf.nn.softmax))                # 激活函數使用 Softmax
model.compile(optimizer=tf.keras.optimizers.Adam(), # 使用 Adam 優化器
    loss=tf.keras.losses.categorical_crossentropy,   # 損失函數為分類交叉熵
    metrics=['accuracy'])                    # 評估指標為準確率
tensorboard = TensorBoard(log_dir="logs")     # 設定 TensorBoard 日誌路徑為 logs
history = model.fit(x_train, y_train2,        # 訓練模型
    epochs=200, batch_size=128,               # 設定訓練的迭代次數和批量大小
    callbacks=[tensorboard],                  # 在每次 epoch 完成後記錄 TensorBoard 日誌
    verbose=1)                                # 設定輸出訓練過程的詳情

# 測試
score = model.evaluate(x_test, y_test2, batch_size=128) # 評估模型的測試集準確率與損失值
print("score:", score)                     # 印出測試結果的損失值和準確率

predict = model.predict(x_test)            # 對測試集進行預測
print("Ans:", np.argmax(predict[0]), np.argmax(predict[1]), np.argmax(predict[2]), np.argmax(predict[3])) # 印出前 4 筆預測的分類結果

# predict2 = model.predict_classes(x_test)   # 舊版本預測方法，已棄用
predict2 = np.argmax(predict, axis=1)       # 使用 argmax 獲得預測的分類結果
print("predict_classes:", predict2)         # 印出所有預測的分類結果
print("y_test", y_test[:])                  # 印出測試集的真實分類標籤