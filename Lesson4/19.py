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

import tensorflow_datasets as tfds   # 1.3.2

import os
import os.path
from os import path

size = 32
max = 100000
category=101
if(path.exists("food101_x.csv") and path.exists("food101_y.csv")   ):
    print("Find food101_x.csv and food101_y.csv")
else:
    # batch_size=-1 to get the full dataset in NumPy arrays from the returned tf.Tensor object
    train , info  = tfds.load(name="food101", split=tfds.Split.TRAIN, batch_size=1,with_info=True )
    print(info)
    if path.exists("food101_x.csv"):
        os.remove("food101_x.csv")
    if path.exists("food101_y.csv"):
        os.remove("food101_y.csv")
    fx = open("food101_x.csv", "a+")
    fy = open("food101_y.csv", "a+")
    i=0
    for example in tfds.as_numpy(train):
        print("processing %d" % i)
        image = tf.image.resize(example['image'], (size, size))
        image=tfds.as_numpy(image)
        label= example['label']
        imageflatten=image.flatten()
        imageString = ','.join(['%d' % num for num in imageflatten])
        labelflatten=label.flatten()
        labelString = ','.join(['%d' % num for num in labelflatten])
        if i>0:
            fx.write("\r\n")
            fy.write(",")
        fx.write(imageString)
        fy.write(labelString)
        i=i+1
        if i>=max:
            break
    fx.close()
    fy.close()

    """
    print(x.shape)
    x=x.reshape(max,size*size*3)
    print(x.shape)
    np.savetxt('food101_x.csv', x.astype(int), fmt='%i',delimiter=',')
    np.savetxt('food101_y.csv', y.astype(int), fmt='%i', delimiter=',')
    x=x.reshape(max,size,size,3)
    """

if(path.exists("food101_x.csv") and path.exists("food101_y.csv")   ):
    # image=np.fromstring(image, sep=',')
    # label=np.fromstring(label, sep=',')
    x = np.loadtxt('food101_x.csv', delimiter=',')
    y = np.loadtxt('food101_y.csv', delimiter=',')
    print(x.shape)
    print(y.shape)
    x = x.reshape(x.shape[0], size , size, 3)
    y = y.reshape(y.shape[0],1)
    print(x.shape)
else:
    exit()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

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
         str1=str1+("%3.0f " % a[i, j,0])
      print(str1)
   print("")

printMatrixE(x_train[0])
print('y_train[0] = ' + str(y_train[0]))

# 顯示其中的圖形
# x_train = x_train.reshape(x_train.shape[0], size, size,3)

num=0
plt.title('x_train[%d]  Label: %d' % (num, y_train[num]))
plt.imshow(x_train[num], cmap=plt.get_cmap('gray_r'))
#plt.show()



# 顯示其中的圖形
num=0
plt.figure()
for num in range(0,36):
   plt.subplot(6,6,num+1)
   plt.title('[%d]->%d'% (num, y_train[num]))
   plt.imshow(x_train[num], cmap=plt.get_cmap('gray_r'))
#plt.show()

x_train = x_train.reshape(x_train.shape[0], size,size, 3)
x_test = x_test.reshape(x_test.shape[0], size,size,3)
print(x_train.shape)
print(x_test.shape)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


y_train2 = tf.keras.utils.to_categorical(y_train, category)
y_test2 = tf.keras.utils.to_categorical(y_test, category)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3),
                 padding="same",
                 activation='relu',
                 input_shape=(size,size,3)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=category,
    activation=tf.nn.softmax ))

model.summary()
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])


for step in range(100):
    cost = model.train_on_batch(x_train, y_train2)
    print("step{}   train cost{}".format(step, cost))





# 保存模型權重
model.save_weights("food101.weights.h5")
    
#測試
score = model.evaluate(x_test, y_test2, batch_size=128)
# 輸出結果
print("score:",score)

predict = model.predict(x_test)
print("Ans:",np.argmax(predict[0]),np.argmax(predict[1]),np.argmax(predict[2]),np.argmax(predict[3]))

predict2 = model.predict_classes(x_test)
print("predict_classes:",predict2)
print("y_test",y_test[:])
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

import tensorflow_datasets as tfds   # 1.3.2

import os
import os.path
from os import path

size = 32
max = 100000
category=101
if(path.exists("food101_x.csv") and path.exists("food101_y.csv")   ):
    print("Find food101_x.csv and food101_y.csv")
else:
    # batch_size=-1 to get the full dataset in NumPy arrays from the returned tf.Tensor object
    train , info  = tfds.load(name="food101", split=tfds.Split.TRAIN, batch_size=1,with_info=True )
    print(info)
    if path.exists("food101_x.csv"):
        os.remove("food101_x.csv")
    if path.exists("food101_y.csv"):
        os.remove("food101_y.csv")
    fx = open("food101_x.csv", "a+")
    fy = open("food101_y.csv", "a+")
    i=0
    for example in tfds.as_numpy(train):
        print("processing %d" % i)
        image = tf.image.resize(example['image'], (size, size))
        image=tfds.as_numpy(image)
        label= example['label']
        imageflatten=image.flatten()
        imageString = ','.join(['%d' % num for num in imageflatten])
        labelflatten=label.flatten()
        labelString = ','.join(['%d' % num for num in labelflatten])
        if i>0:
            fx.write("\r\n")
            fy.write(",")
        fx.write(imageString)
        fy.write(labelString)
        i=i+1
        if i>=max:
            break
    fx.close()
    fy.close()

    """
    print(x.shape)
    x=x.reshape(max,size*size*3)
    print(x.shape)
    np.savetxt('food101_x.csv', x.astype(int), fmt='%i',delimiter=',')
    np.savetxt('food101_y.csv', y.astype(int), fmt='%i', delimiter=',')
    x=x.reshape(max,size,size,3)
    """

if(path.exists("food101_x.csv") and path.exists("food101_y.csv")   ):
    # image=np.fromstring(image, sep=',')
    # label=np.fromstring(label, sep=',')
    x = np.loadtxt('food101_x.csv', delimiter=',')
    y = np.loadtxt('food101_y.csv', delimiter=',')
    print(x.shape)
    print(y.shape)
    x = x.reshape(x.shape[0], size , size, 3)
    y = y.reshape(y.shape[0],1)
    print(x.shape)
else:
    exit()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

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
         str1=str1+("%3.0f " % a[i, j,0])
      print(str1)
   print("")

printMatrixE(x_train[0])
print('y_train[0] = ' + str(y_train[0]))

# 顯示其中的圖形
# x_train = x_train.reshape(x_train.shape[0], size, size,3)

num=0
plt.title('x_train[%d]  Label: %d' % (num, y_train[num]))
plt.imshow(x_train[num], cmap=plt.get_cmap('gray_r'))
#plt.show()



# 顯示其中的圖形
num=0
plt.figure()
for num in range(0,36):
   plt.subplot(6,6,num+1)
   plt.title('[%d]->%d'% (num, y_train[num]))
   plt.imshow(x_train[num], cmap=plt.get_cmap('gray_r'))
#plt.show()

x_train = x_train.reshape(x_train.shape[0], size,size, 3)
x_test = x_test.reshape(x_test.shape[0], size,size,3)
print(x_train.shape)
print(x_test.shape)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


y_train2 = tf.keras.utils.to_categorical(y_train, category)
y_test2 = tf.keras.utils.to_categorical(y_test, category)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3),
                 padding="same",
                 activation='relu',
                 input_shape=(size,size,3)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=category,
    activation=tf.nn.softmax ))

model.summary()
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])


for step in range(100):
    cost = model.train_on_batch(x_train, y_train2)
    print("step{}   train cost{}".format(step, cost))





# 保存模型權重
model.save_weights("food101.weights.h5")
    
#測試
score = model.evaluate(x_test, y_test2, batch_size=128)
# 輸出結果
print("score:",score)

predict = model.predict(x_test)
print("Ans:",np.argmax(predict[0]),np.argmax(predict[1]),np.argmax(predict[2]),np.argmax(predict[3]))

predict2 = model.predict_classes(x_test)
print("predict_classes:",predict2)
print("y_test",y_test[:])
