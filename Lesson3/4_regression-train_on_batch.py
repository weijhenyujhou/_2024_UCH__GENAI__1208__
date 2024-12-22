#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "柯博文老師 Powen Ko, www.powenko.com"

# 匯入必要的模組
from sklearn import datasets  # sklearn：一個機器學習模組（此處未使用）
from sklearn.model_selection import train_test_split  # 用於拆分資料集為訓練集和測試集
import tensorflow as tf  # TensorFlow：用於深度學習的框架
import numpy as np  # NumPy：用於數值計算的模組
import matplotlib.pyplot as plt  # Matplotlib：用於繪圖的模組

# 建立輸入和輸出資料集
X = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # 輸入數據
Y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # 輸出數據

# 拆分資料集為訓練集和測試集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)  # 將數據拆分為訓練集（80%）和測試集（20%）

# 建立神經網絡模型
model = tf.keras.models.Sequential()  # 使用序列式模型
model.add(tf.keras.layers.Dense(units=1, input_dim=1))  # 添加一個全連接層（單一神經元），輸入維度為1

# 編譯模型
model.compile(loss='mse', optimizer='sgd', metrics=['acc'])  # 使用均方誤差（mse）作為損失函數，隨機梯度下降（sgd）作為優化器，計算準確率（acc）

# model.fix()
# 開始訓練模型
for step in range(8000):  # epoch 訓練8000步
    cost = model.train_on_batch(x_train, y_train)  # 使用批次訓練資料進行一次模型更新，並返回當前批次的損失值和其他指標    
    if step % 200 == 0:  # 每200步打印一次資訊並繪圖
        W, b = model.layers[0].get_weights()  # 獲取模型的權重和偏置
        print("step{} Weights = {}, bias = {} train cost{}".format(step, W, b, cost))  # 打印當前步驟、權重、偏置和訓練損失

        plt.cla()  # 清除當前的圖表
        # 畫出數據點
        plt.scatter(X, Y)  # 繪製原始數據點的散點圖

        # 使用模型預測測試集數據
        X_test2 = np.array([0, 1])  # 測試集範圍
        Y_pred2 = model.predict(X_test2)  # 使用模型預測Y值
        plt.plot(X_test2, Y_pred2, "r-")  # 繪製預測線，用紅色表示

        # 在圖上顯示當前的訓練資訊
        plt.text(0, 1, 'epoch:%d ,W=%.4f ,b=%.4f ,cost=%.4f ' % (step, W, b, cost[0]),
                 fontdict={'size': 10, 'color': 'red'})  # 顯示步數、權重、偏置和損失值

        plt.pause(0.01)  # 暫停0.01秒，更新圖表
         
# 顯示最終的圖表
plt.show()  # 顯示圖表