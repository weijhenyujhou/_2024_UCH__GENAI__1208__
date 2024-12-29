#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "柯博文老師 Powen Ko, www.powenko.com"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os.path
from os import path

import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
print(tfds.list_builders())



mnist_train , info  = tfds.load(name="mnist", split=tfds.Split.TRAIN, batch_size=-1,with_info=True )
print(info)
mnist_test = tfds.load(name="mnist", split=tfds.Split.TEST, batch_size=-1)
mnist_train = tfds.as_numpy(mnist_train)
mnist_test = tfds.as_numpy(mnist_test)
x_train, y_train = mnist_train["image"], mnist_train["label"]
x_test, y_test = mnist_test["image"], mnist_test["label"]
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

print(x_train[0].shape)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train.astype(np.uint8)




# 顯示資料內容
def printMatrixE(a):
   rows = a.shape[0]
   cols = a.shape[1]
   for i in range(0,rows):
      str1=""
      for j in range(0,cols):
         str1=str1+("%3.0f " % a[i, j])
      print(str1)
   print("")

printMatrixE(x_train[0])
print('y_train[0] = ' + str(y_train[0]))

# 顯示其中的圖形
x_train = x_train.reshape(x_train.shape[0], 28, 28)
num=0
plt.title('x_train[%d]  Label: %d' % (num, y_train[num]))
plt.imshow(x_train[num], cmap=plt.get_cmap('gray_r'))
plt.show()



# 顯示其中的圖形
num=0
plt.figure()
for num in range(0,36):
   plt.subplot(6,6,num+1)
   plt.title('[%d]->%d'% (num, y_train[num]))
   plt.imshow(x_train[num], cmap=plt.get_cmap('gray_r'))
plt.show()

x_train = x_train.reshape(x_train.shape[0], 28,28, 1)
x_test = x_test.reshape(x_test.shape[0], 28,28, 1)
print(x_train.shape)
print(x_test.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


y_train2 = tf.keras.utils.to_categorical(y_train, 10)
y_test2 = tf.keras.utils.to_categorical(y_test, 10)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3),
                 padding="same",
                 activation='relu',
                 input_shape=(28,28,1)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=10,
    activation=tf.nn.softmax ))

model.summary()
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])


for step in range(100):
    cost = model.train_on_batch(x_train, y_train2)
    print("step{}   train cost{}".format(step, cost))

#測試
score = model.evaluate(x_test, y_test2, batch_size=128)
# 輸出結果
print("score:",score)

predict = model.predict(x_test)
print("Ans:",np.argmax(predict[0]),np.argmax(predict[1]),np.argmax(predict[2]),np.argmax(predict[3]))

predict2 = model.predict_classes(x_test)
print("predict_classes:",predict2)
print("y_test",y_test[:])



