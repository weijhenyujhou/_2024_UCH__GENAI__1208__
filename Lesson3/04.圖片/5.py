#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "柯博文老師 Powen Ko, www.powenko.com"

# 指定執行此腳本的 Python 解釋器位置和文件編碼為 UTF-8
# 作者信息

# 需要的 Python 模組安裝命令
# sudo apt-get install python-numpy python3-numpy python-matplotlib python3-matplotlib

# 匯入必要的模組
import os  # 用於與操作系統交互的模組
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 設置 TensorFlow 日誌等級為 2，以減少不必要的警告信息

# 從 TensorFlow 的 mnist 模組匯入數據加載功能（這一行應該是過時的註解）
#from tensorflow.examples.tutorials.mnist
# from tensorflow.examples.tutorials.mnist import input_data  # 這一行實際上可能無效，因為 TensorFlow 2.x 不再使用此方式

import tensorflow as tf  # 匯入 TensorFlow 模組
import matplotlib.pyplot as plt  # 匯入 Matplotlib，用於數據可視化 
 

# 載入 MNIST 資料集，並將資料分為訓練集和測試集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 打印訓練數據的形狀
print('x_train = ' + str(x_train.shape))  # x_train 應為 (60000, 28, 28)
print('y_train = ' + str(y_train.shape))  # y_train 應為 (60000,)

# 顯示資料內容的函數定義
def printMatrixE(a):
   rows = a.shape[0]  # 獲取行數
   cols = a.shape[1]  # 獲取列數
   for i in range(0, rows):  # 迭代每一行
      str1 = ""  # 初始化一個空字串，用於存儲當前行的內容
      for j in range(0, cols):  # 迭代每一列
         str1 = str1 + ("%3.0f " % a[i, j])  # 將當前元素格式化為整數並添加到字串
      print(str1)  # 打印當前行
   print("")  # 打印一個空行，分隔不同矩陣的顯示

# 打印第一張訓練圖片的像素矩陣
printMatrixE(x_train[0])  # 打印 x_train 的第一張圖片矩陣

# 打印第一張訓練圖片的標籤
print('y_train[0] = ' + str(y_train[0]))  # 打印 y_train 的第一個標籤