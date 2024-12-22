#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "柯博文老師 Powen Ko, www.powenko.com"

# 匯入必要的模組
import tensorflow as tf  # TensorFlow：一個用於深度學習的框架
import numpy as np  # NumPy：用於數值計算的模組
import matplotlib.pyplot as plt  # Matplotlib：用於繪圖的模組
import time  # time：用於時間相關操作的模組

# 設定基本參數
dim = 5  # 維度數量
category = 1  # 類別數量，這裡未使用
num = 100  # 資料點的數量

# 產生等間距的數據，並將其重塑為 X 矩陣
a = np.linspace(0, 1, num * dim)  # 生成從0到1的等間距數列，共有num*dim個元素
X = np.reshape(a, (num, dim))  # 重塑為 num 行 dim 列的矩陣
Y = X.sum(axis=1)  # 對每一行的元素求和，得到對應的標籤 Y

# 顯示 X 和 Y 的前五個元素
print(X[:5])  # 打印前五個 X
print(Y[:5])  # 打印前五個 Y

# 建立訓練集和測試集
import sklearn.model_selection as sk  # 引入 sklearn 中的模型選擇模組
X_train, X_test, Y_train, Y_test = sk.train_test_split(X, Y, test_size=0.2, random_state=42)  # 將數據劃分為訓練集和測試集，測試集佔20%

# 建立神經網絡模型
model = tf.keras.models.Sequential()  # 使用序列式模型
model.add(tf.keras.layers.Dense(units=100, activation=tf.nn.tanh, input_dim=dim))  # 添加一個隱藏層，100個神經元，啟用函數為tanh，輸入維度為dim
model.add(tf.keras.layers.Dense(units=1))  # 添加一個輸出層，1個神經元
model.summary()  # 打印模型摘要

# 編譯模型
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])  # 使用均方誤差(mse)作為損失函數，隨機梯度下降(sgd)作為優化器，計算準確率

# 訓練模型
model.fit(X_test, Y_test, epochs=4000, batch_size=len(Y_test))  # 使用測試數據進行訓練，訓練4000個世代，批量大小為測試集大小

# 開始測試
print("start testing")  # 打印開始測試
cost = model.evaluate(X_test, Y_test)  # 評估模型在測試集上的表現
print("test cost: {}".format(cost))  # 打印測試損失值

# 預測測試集數據
Y_pred = model.predict(X_test)  # 預測測試集的 Y 值

# 準備繪圖
X2 = X[:, 2]  # 取 X 的第三列（索引為2）
print(X2[:5])  # 打印 X2 的前五個元素
X_test2 = X_test[:, 2]  # 取測試集 X 的第三列

# 畫出數據點和預測結果
plt.scatter(X2, Y, label='X,Y')  # 繪製 X2 與 Y 的散點圖
plt.scatter(X_test2, Y_test, label='X_test, Y_test')  # 繪製測試集的散點圖
plt.plot(X_test2, Y_pred, 'xr', label='pred')  # 繪製預測結果，用紅色叉表示

# 顯示圖表
plt.show()  # 顯示圖表
