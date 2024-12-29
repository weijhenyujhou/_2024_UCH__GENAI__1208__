#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "柯博文老師 Powen Ko, www.powenko.com"

import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import urllib.request

# 加載模型
model = hub.load(
    'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5')

# 自動下載標籤文件
label_file_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
label_path = 'ImageNetLabels.txt'

if not os.path.exists(label_path):
    print("Downloading label file...")
    urllib.request.urlretrieve(label_file_url, label_path)
    print("Download complete.")

# 讀取標籤
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# 處理影像的函數


def preprocess_frame(frame):
    # 調整大小為 224x224 並正規化
    frame_resized = cv2.resize(frame, (224, 224))
    frame_normalized = frame_resized / 255.0
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    return tf.convert_to_tensor(frame_expanded, dtype=tf.float32)


# 啟用 Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # 預處理當前幀
    input_tensor = preprocess_frame(frame)

    # 推論
    predictions = model(input_tensor)
    predicted_index = tf.argmax(predictions[0]).numpy()
    predicted_label = labels[predicted_index]

    # 在畫面上顯示分類結果
    output_frame = frame.copy()
    cv2.putText(output_frame, f"Predicted: {predicted_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 顯示畫面
    cv2.imshow("Webcam - Real-time Classification", output_frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# 釋放資源
cap.release()
cv2.destroyAllWindows()
