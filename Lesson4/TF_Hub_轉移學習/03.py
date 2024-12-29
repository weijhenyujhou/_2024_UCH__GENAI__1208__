#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "柯博文老師 Powen Ko, www.powenko.com"


import tensorflow_hub as hub
from tensorflow.keras import layers, models

# 加載一個預訓練模型
feature_extractor = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5",
                                   input_shape=(224, 224, 3),
                                   trainable=False)

# 添加全連接層
model = models.Sequential([
    feature_extractor,
    layers.Dense(10, activation='softmax')  # 假設有10個分類
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
model.fit(train_dataset, epochs=5)