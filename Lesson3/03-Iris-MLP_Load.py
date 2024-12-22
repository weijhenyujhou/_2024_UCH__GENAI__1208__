#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = "Powen Ko, www.powenko.com"

from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from time import time
from tensorflow.keras.models import model_from_json


iris = datasets.load_iris()

category=3
dim=4

x_train , x_test , y_train , y_test = train_test_split(iris.data,iris.target,test_size=0.2)
y_train2=tf.keras.utils.to_categorical(y_train, num_classes=(category))
y_test2=tf.keras.utils.to_categorical(y_test, num_classes=(category))

print("x_train[:4]",x_train[:4])
print("y_train[:4]",y_train[:4])
print("y_train2[:4]",y_train2[:4])

 

##### 載入模型
model = tf.keras.models.load_model('model.h5')


# 讀取模型權重
model.load_weights("model.weights.h5")
  
#不要忘記放入compile
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss = tf.keras.losses.categorical_crossentropy,
    metrics = ['accuracy'])



#測試
score = model.evaluate(x_test, y_test2, batch_size=128)
print("score:",score)

predict = model.predict(x_test)
print("Ans:",np.argmax(predict[0]),np.argmax(predict[1]),np.argmax(predict[2]),np.argmax(predict[3]))

# predict2 = model.predict_classes(x_test)
predict2 = np.argmax(predict, axis=1)
print("predict_classes:",predict2)
print("y_test",y_test[:])