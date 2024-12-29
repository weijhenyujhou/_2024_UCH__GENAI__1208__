#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "柯博文老師 Powen Ko, www.powenko.com"


import tensorflow as tf
import tensorflow_hub as hub
import os
from tensorflow.keras import layers, models

# 設定資料目錄路徑
data_dir = 'images'

# 使用 Keras 的 image_dataset_from_directory 函數載入資料
batch_size = 32
img_height = 224
img_width = 224

train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,  # 80% 訓練, 20% 驗證
    subset="training",
    seed=123,  # 固定隨機種子以確保結果一致
    image_size=(img_height, img_width),
    batch_size=batch_size
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# 顯示類別名稱
class_names = train_dataset.class_names
print(f"分類名稱: {class_names}")

# 標準化影像數據到 [0,1]
normalization_layer = layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

# 使用緩存以提高性能
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# 加載預訓練模型
feature_extractor = hub.KerasLayer(
    "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5",
    input_shape=(img_height, img_width, 3),
    trainable=False  # 凍結預訓練層
)

# 构建模型
model = models.Sequential([
    feature_extractor,
    layers.Dense(len(class_names), activation='softmax')  # 輸出層，根據類別數設置輸出維度
])

# 編譯模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 訓練模型
model.fit(train_dataset, validation_data=validation_dataset, epochs=1)

# 保存模型
model.save('my_tf_model')
print("模型訓練完成並已保存。")

# save weights
model.save_weights('my_tf_model_weights')
print("模型權重已保存。")