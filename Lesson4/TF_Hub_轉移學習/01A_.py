#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "柯博文老師 Powen Ko, www.powenko.com"

import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import urllib.request

# 加載模型
model = hub.load('https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5')



# 載入影像並處理
image = Image.open('1.jpg').resize((224, 224))  # 改變影像大小
image = np.array(image) / 255.0  # 正規化到 [0, 1]
image = np.expand_dims(image, axis=0)  # 增加批次維度

# 確保輸入的資料類型為 float32
image = tf.convert_to_tensor(image, dtype=tf.float32)

# 推論
predictions = model(image)  # 預測
predicted_class = tf.argmax(predictions[0])
print(f'Predicted class: {predicted_class.numpy()}')
predicted_index=predicted_class.numpy()


# 自動下載標籤文件
label_file_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
label_path = 'ImageNetLabels.txt'

if not os.path.exists(label_path):
    print("Downloading label file...")
    urllib.request.urlretrieve(label_file_url, label_path)
    print("Download complete.")

# 讀取標籤
with open(label_path, 'r') as f: # 讀取標籤
    labels = [line.strip() for line in f.readlines()] 
    predicted_label = labels[predicted_index]
    print(f'Predicted label: {predicted_label}')