#!/usr/bin/env python  # 指定Python直譯器的位置
# -*- coding=utf-8 -*-  # 指定檔案的編碼格式為UTF-8
__author__ = "柯博文老師 Powen Ko, www.powenko.com"  # 定義作者資訊

import tensorflow as tf  # 匯入TensorFlow庫
import numpy as np  # 匯入NumPy庫

# import tensorflow_datasets as tfds  # 匯入TensorFlow Datasets庫，用於處理標準化數據集
import matplotlib.pyplot as plt  # 匯入Matplotlib庫，用於繪圖

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()  # 載入CIFAR-10數據集，分為訓練集和測試集
print(x_train.shape)  # 輸出訓練數據的形狀
print(y_train.shape)  # 輸出訓練標籤的形狀
print(x_test.shape)  # 輸出測試數據的形狀
print(y_test.shape)  # 輸出測試標籤的形狀

# 顯示其中的圖形
num=0  # 初始化計數器num
for num in range(0, 36):  # 從0到35，顯示36張訓練圖片
   plt.subplot(6, 6, num+1)  # 創建6x6的子圖佈局，並選擇第num+1個位置
   plt.title('[%d]->%d' % (num, y_train[num]))  # 設定標題，顯示圖片編號和對應的標籤
   plt.imshow(x_train[num])  # 顯示第num張訓練圖片
plt.show()  # 顯示整個圖形

x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)  # 將訓練數據重塑為32x32x3的形狀（彩色圖像）
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)  # 將測試數據重塑為32x32x3的形狀（彩色圖像）
print(x_train.shape)  # 輸出重塑後訓練數據的形狀
print(x_test.shape)  # 輸出重塑後測試數據的形狀

x_train = x_train.astype('float32')  # 將訓練數據類型轉換為float32
x_test = x_test.astype('float32')  # 將測試數據類型轉換為float32
x_train /= 255  # 將訓練數據歸一化到0-1之間
x_test /= 255  # 將測試數據歸一化到0-1之間

# 將數字轉為 One-hot 向量
y_train2 = tf.keras.utils.to_categorical(y_train, 10)  # 將訓練標籤轉換為One-hot編碼
y_test2 = tf.keras.utils.to_categorical(y_test, 10)  # 將測試標籤轉換為One-hot編碼