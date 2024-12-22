#!/usr/bin/env python                              # 設定腳本為可執行檔案
# -*- coding=utf-8 -*-                            # 設定檔案的編碼為 UTF-8
__author__ = "柯博文老師 Powen Ko, www.powenko.com" # 作者資訊

import tensorflow as tf                           # 引入 TensorFlow 套件
from sklearn.model_selection import train_test_split # 引入資料分割模組
import numpy as np                                # 引入 NumPy 套件，用於數值運算
import matplotlib.pyplot as plt                  # 引入 Matplotlib 套件，用於資料視覺化
import pandas as pd                               # 引入 Pandas 套件，用於資料處理
from tensorflow.keras.datasets import boston_housing # 引入波士頓房價資料集

# 載入波士頓房價資料集
(x_train, y_train), (x_test, y_test) = boston_housing.load_data() # 載入訓練集與測試集

# 印出訓練資料與標籤的形狀
print(x_train.shape)                             # 印出訓練資料的形狀 (訓練樣本數, 特徵數)
print(y_train.shape)                             # 印出訓練標籤的形狀 (訓練樣本數,)

# 建立 Pandas DataFrame 並新增列標籤
classes = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'] # 特徵名稱
data = pd.DataFrame(x_train, columns=classes)   # 建立 DataFrame 並加入列標籤
print(data.head())                              # 印出前 5 筆資料

# 新增房價 (MEDV) 資料列到 DataFrame
data['MEDV'] = pd.Series(data=y_train)          # 將房價資料加入 DataFrame
print(data.head())                              # 再次印出前 5 筆資料
print(data.describe())                          # 印出資料集的描述性統計資料

# 引入 Seaborn 用於視覺化資料 (此處未使用，但方便後續分析)
import seaborn as sns

# 資料標準化 (Min-Max Scaling)
from sklearn import preprocessing                # 引入 sklearn 的預處理模組
scaler = preprocessing.MinMaxScaler()            # 建立 Min-Max 標準化器
x_train = scaler.fit_transform(x_train)          # 標準化訓練資料
x_test = scaler.fit_transform(x_test)            # 標準化測試資料

# 建立神經網路模型
model = tf.keras.models.Sequential()             # 建立 Sequential 模型
model.add(tf.keras.layers.Dense(32, activation='tanh', input_shape=[x_train.shape[1]])) # 第一層，32 個神經元，激活函數 tanh
model.add(tf.keras.layers.Dense(64, activation='tanh')) # 第二層，64 個神經元，激活函數 tanh
model.add(tf.keras.layers.Dense(1))              # 輸出層，1 個輸出 (房價)

# 編譯模型
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),  # 明確指定 MSE
    optimizer='sgd', metrics=['mae']) # 損失函數為均方誤差 (MSE)，優化器為 SGD，評估指標為 MAE

# 訓練模型
history = model.fit(x_train, y_train,            # 訓練資料與標籤
                    epochs=40000,                # 設定訓練的迭代次數為 40000
                    batch_size=len(y_train))     # 批量大小設定為所有樣本數


# 保存模型架構與權重
#  tf.keras.models.save('model.h5')              # 這行為備註，原本用於保存模型
model.save('model.h5')                           # 保存完整模型到檔案 model.h5

##### 載入模型
model = tf.keras.models.load_model('model.h5')   # 載入已保存的模型


# 保存模型權重
model.save_weights("model.weights.h5")           # 保存模型的權重到檔案 model.weights.h5

# 讀取模型權重
model.load_weights("model.weights.h5")           # 從 model.weights.h5 檔案讀取模型權重


# 測試模型
print("start testing")                          # 開始測試
cost = model.evaluate(x_test, y_test)          # 評估模型在測試集上的損失與評估指標
print("test cost: {}".format(cost))            # 印出測試結果

# 預測測試集
Y_pred2 = model.predict(x_test)                # 預測測試集資料
print(Y_pred2[:10])                            # 印出前 10 筆預測結果
print(y_test[:10])                             # 印出前 10 筆實際標籤

# 視覺化訓練過程
print(history.history.keys())                  # 印出訓練歷史包含的關鍵字
plt.plot(history.history['mae'])               # 畫出 MAE 曲線
plt.title('Boston house price')                # 設定圖表標題
plt.ylabel('mae')                              # 設定 y 軸標籤
plt.xlabel('epoch')                            # 設定 x 軸標籤
plt.legend(['train mae'], loc='upper right')   # 加入圖例
plt.show()                                     # 顯示圖表