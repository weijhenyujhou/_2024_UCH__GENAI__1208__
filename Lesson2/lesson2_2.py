#!/usr/bin/env python  # 指定執行此腳本的 Python 解釋器路徑
# -*- coding=utf-8 -*-  # 設定檔案的文字編碼為 UTF-8
__author__ = "柯博文老師 Powen Ko, www.powenko.com"  # 作者資訊

import tensorflow as tf  # 匯入 TensorFlow 模組，用於建構深度學習模型
import numpy as np  # 匯入 NumPy 模組，用於數據操作和處理

# 生成隨機數據作為輸入 x1，大小為 500x1
x1 = np.random.random((500, 1))  
# 生成另一組隨機數據作為輸入 x2，並加上 1 使其與 x1 有明顯差異
x2 = np.random.random((500, 1)) + 1  
# 將 x1 和 x2 合併為一個訓練集 x_train
x_train = np.concatenate((x1, x2))  

# 創建對應於 x1 的標籤數據 y1，全為 0，長度為 500
y1 = np.zeros((500,), dtype=int)  
# 創建對應於 x2 的標籤數據 y2，全為 1，長度為 500
y2 = np.ones((500,), dtype=int)  
# 將 y1 和 y2 合併為一個標籤訓練集 y_train
y_train = np.concatenate((y1, y2))  

# 定義一個順序型模型（Sequential 模型）
model = tf.keras.models.Sequential([
    # 添加第一層 Dense（全連接層），輸入維度為 1，激活函數使用 ReLU
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_dim=1),
    # 添加第二層 Dense，輸出為 10 個節點，激活函數使用 ReLU
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    # 添加第三層 Dense，輸出為 2 個節點，激活函數使用 Softmax（用於分類）
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])