# -*- coding: utf-8 -*-
# 資料來源： https://medium.com/datadriveninvestor/generative-adversarial-network-gan-using-keras-ce1c05cfdfd3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm       # 顯示進度   50%|█████     | 5/10 [00:21<00:22,  4.55s/it]


for i in tqdm(range(10)):
    print(i)

# 讀取資料，並且標準化
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # 讀取手寫資料
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5    # 標準化 -1~1

    # convert shape of x_train from (60000, 28, 28) to (60000, 784)
    # 784 columns per row
    print("x_train.shape:",x_train.shape)
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2]*1) #60000,784)
    return (x_train, y_train, x_test, y_test)


(X_train, y_train, X_test, y_test) = load_data()     # 讀取資料，並且標準化 -1~1
print("X_train.shape:",X_train.shape)
