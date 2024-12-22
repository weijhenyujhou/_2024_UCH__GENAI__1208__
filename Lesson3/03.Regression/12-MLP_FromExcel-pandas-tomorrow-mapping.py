#!/usr/bin/env python                              # 設定腳本為可執行檔案
# -*- coding=utf-8 -*-                            # 設定檔案的編碼為 UTF-8
__author__ = "柯博文老師 Powen Ko, www.powenko.com" # 作者資訊
# pip install pandas numpy                         # 提醒使用者安裝必要的套件

from sklearn.model_selection import train_test_split # 引入資料分割模組
import tensorflow as tf                        # 引入 TensorFlow 套件
import numpy as np                             # 引入 NumPy 套件，用於數值運算
import pandas as pd                            # 引入 Pandas 套件，用於資料處理

# 讀取 Excel 檔案到 Pandas DataFrame
df = pd.read_excel("weather.xls")              # 載入名為 weather.xls 的 Excel 資料





# 選擇 DataFrame 的第 11、12、4 和 9 列作為特徵，使用 iloc 索引
selected_columns = df.iloc[1:, [0,1,2,3,4,6,9,10,11,12,13,14,15,16,17,18]] # 跳過第一列 (通常是標題列)，選擇指定的列

# 選擇 DataFrame 的第 20 列作為標籤
selected_columns = df.iloc[1:, [23]]           # 跳過第一列，選擇第 20 列
Y = selected_columns.values.flatten()          # 將選定的列轉為 NumPy 陣列並壓平成 1D


# 定義函數自動生成數字映射表
def generate_mapping(column):
    unique_values = column.dropna().unique()  # 去除 NaN 並取得唯一值
    mapping = {value: idx for idx, value in enumerate(unique_values)}
    return mapping


# 產生 RainToday 欄位的數字映射表
WindGustDir_mapping = generate_mapping(df['WindGustDir'])
# 將 RainToday 欄位映射為數字
df['WindGustDir_numeric'] = df['WindGustDir'].map(WindGustDir_mapping)
selected_columns['WindGustDir_numeric'] = df['WindGustDir_numeric'][1:]
X = selected_columns.values                    # 將選定的列轉為 NumPy 陣列 (2D)





# 設定分類數量與特徵維度
category = 2                                   # 分類數量 (二分類)
dim = X.shape[1]                               # 特徵維度 (取 X 的列數)

# 分割資料集為訓練集與測試集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) # 測試集比例為 20%
y_train2 = tf.keras.utils.to_categorical(y_train, num_classes=category) # 將訓練標籤轉為 One-Hot 編碼
y_test2 = tf.keras.utils.to_categorical(y_test, num_classes=category)   # 將測試標籤轉為 One-Hot 編碼

# 輸出部分訓練資料以檢查資料內容
print("x_train[:4]", x_train[:4])              # 印出訓練資料的前 4 筆
print("y_train[:4]", y_train[:4])              # 印出訓練標籤的前 4 筆
print("y_train2[:4]", y_train2[:4])            # 印出 One-Hot 編碼後的訓練標籤前 4 筆

# 建立模型
model = tf.keras.models.Sequential()           # 建立 Sequential 模型
model.add(tf.keras.layers.Dense(units=100,     # 新增全連接層，100 個神經元
                                activation=tf.nn.relu, # 激活函數使用 ReLU
                                input_dim=dim)) # 設定輸入層特徵數目為 dim
model.add(tf.keras.layers.Dense(units=100,     # 新增第二個全連接層，100 個神經元
                                activation=tf.nn.relu)) # 激活函數使用 ReLU
model.add(tf.keras.layers.Dense(units=category, # 新增輸出層，單元數等於分類數
                                activation=tf.nn.softmax)) # 激活函數使用 Softmax
model.compile(optimizer='adam',                # 使用 Adam 優化器
              loss=tf.keras.losses.categorical_crossentropy, # 損失函數為分類交叉熵
              metrics=['accuracy'])            # 評估指標為準確率

# 訓練模型
model.fit(x_train, y_train2,                   # 訓練資料與標籤
          epochs=1000,                        # 訓練的迭代次數設定為 20000
          batch_size=64)                       # 批量大小設定為 64

# 測試
model.summary()                                # 輸出模型的摘要

score = model.evaluate(x_test, y_test2, batch_size=64) # 評估模型在測試集上的表現
print("score:", score)                         # 印出測試結果的損失值與準確率

# 進行預測
predict=model.predict(x_test)                  # 對測試集進行預測
print("predict:", predict)                     # 印出所有預測的機率值
# predict2 = model.predict_classes(x_test)     # 預測測試資料的分類 (舊方法，已棄用)
predict2 = np.argmax(predict, axis=1)          # 使用 argmax 獲得預測的分類結果
print("predict_classes:", predict2)            # 印出所有預測的分類結果
print("y_test", y_test[:])                     # 印出測試集的真實分類標籤