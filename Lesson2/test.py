#!/usr/bin/env python  # 指定 Python 解釋器路徑
# -*- coding=utf-8 -*-  # 設定檔案編碼為 UTF-8
__author__ = "柯博文老師 Powen Ko, www.powenko.com"  # 作者資訊

import tensorflow as tf  # 匯入 TensorFlow，用於構建深度學習模型
import numpy as np  # 匯入 NumPy，用於數據操作

# 生成第一組隨機數據 x1，大小為 500x1
x1 = np.random.random((500, 1))  
# 生成第二組隨機數據 x2，大小為 500x1，並將其值偏移加 1
x2 = np.random.random((500, 1)) + 1  
# 將 x1 和 x2 合併為輸入數據 x_train
x_train = np.concatenate((x1, x2))  

# 生成對應 x1 的標籤 y1，全為 0，大小為 500
y1 = np.zeros((500,), dtype=int)  
# 生成對應 x2 的標籤 y2，全為 1，大小為 500
y2 = np.ones((500,), dtype=int)  
# 將 y1 和 y2 合併為完整的標籤數據 y_train
y_train = np.concatenate((y1, y2))  

# 定義一個順序型模型（Sequential 模型）
model = tf.keras.models.Sequential([
    # 第一層：全連接層（Dense），10 個神經元，激活函數為 ReLU，輸入維度為 1
    tf.keras.layers.Dense(units=10, activation=tf.nn.relu, input_dim=1),
    # 第二層：全連接層（Dense），10 個神經元，激活函數為 ReLU
    tf.keras.layers.Dense(units=10, activation=tf.nn.relu),
    # 第三層：全連接層（Dense），2 個輸出神經元（對應 2 個分類），激活函數為 Softmax
    tf.keras.layers.Dense(units=2, activation=tf.nn.softmax)
])

# 編譯模型，設定優化器、損失函數和評估指標
model.compile(
    optimizer='adam',  # 使用 Adam 優化器
    loss='sparse_categorical_crossentropy',  # 損失函數：稀疏分類交叉熵
    metrics=['accuracy']  # 評估指標：準確率
)

# 訓練模型
model.fit(
    x_train, y_train,  # 輸入數據和標籤
    epochs=20,  # 訓練 20 個迭代週期
    batch_size=128  # 每次批量大小為 128
)

# 測試模型
x_test = np.array([[0.22], [0.31], [1.22], [1.33]])  # 測試數據
y_test = np.array([0, 0, 1, 1])  # 測試標籤

# 評估模型性能
score = model.evaluate(x_test, y_test, batch_size=128)  
print("score:", score)  # 輸出評估分數（損失值和準確率）

# 預測測試數據
predict = model.predict(x_test)  
print("predict:", predict)  # 輸出預測結果（每類別的機率）
# 輸出每個數據點預測的類別
print("Ans:", np.argmax(predict[0]), np.argmax(predict[1]), np.argmax(predict[2]), np.argmax(predict[3]))  

# 使用 `predict_classes` 預測類別（已被棄用，建議使用 `np.argmax`）
# predict2 = model.predict_classes(x_test)  
predict2 = np.argmax(predict, axis=1)  # 找出每個數的類別（類別） 
print("predict_classes:", predict2)  # 輸出預測的類別
print("y_test", y_test[:])  # 輸出實際測試標籤