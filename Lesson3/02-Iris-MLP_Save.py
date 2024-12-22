#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "柯博文老師 Powen Ko, www.powenko.com"
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
import numpy as np

# 載入資料
iris = datasets.load_iris()

category = 3
dim = 4
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
y_train2 = tf.keras.utils.to_categorical(y_train, num_classes=category)
y_test2 = tf.keras.utils.to_categorical(y_test, num_classes=category)

# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=10, activation='relu', input_dim=dim))
model.add(tf.keras.layers.Dense(units=10, activation='relu'))
model.add(tf.keras.layers.Dense(units=category, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
tensorboard = TensorBoard(log_dir="logs")
history = model.fit(x_train, y_train2, epochs=1000, batch_size=128, callbacks=[tensorboard], verbose=1)

# 保存模型
#  tf.keras.models.save('model.h5')
model.save('model.h5')

##### 載入模型
model = tf.keras.models.load_model('model.h5')  # <---

#保存模型權重
model.save_weights("model.weights.h5")

# 讀取模型權重
model.load_weights("model.weights.h5")
 



#測試
score = model.evaluate(x_test, y_test2, batch_size=128)
print("score:",score)

predict = model.predict(x_test)
print("Ans:",np.argmax(predict[0]),np.argmax(predict[1]),np.argmax(predict[2]),np.argmax(predict[3]))

# predict2 = model.predict_classes(x_test)
predict2 = np.argmax(predict, axis=1)
print("predict_classes:",predict2)
print("y_test",y_test[:])



