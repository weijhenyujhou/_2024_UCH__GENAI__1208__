#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "柯博文老師 Powen Ko, www.powenko.com"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

X = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
Y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])



# 建立 trainig 與 testing data
import sklearn.model_selection as sk
X_train, X_test, Y_train, Y_test =sk.train_test_split(X,Y,test_size=0.2, random_state = 42)

# 建立 neural network from the first layer to last layer


model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Dense(units=1, activation=tf.nn.relu,input_dim=1))
model.add(tf.keras.layers.Dense(units=1,input_dim=1))
model.summary()
# 開始搭建 model
# mse = mean square error
# sgd = stochastic gradient descent
model.compile(loss='mse',optimizer='sgd',metrics=['accuracy'])

#model.compile(loss='mean_squared_error',
#              optimizer=tf.keras.optimizers.Adam(lr=0.1))


model.fit(X_test, Y_test,
          epochs=2000)


# testing
print("start testing")
cost = model.evaluate(X_test, Y_test)
print("test cost: {}".format(cost))
weights, biases = model.layers[0].get_weights()
print("Weights = {}, bias = {}".format(weights,biases))


# 印出測試的結果
Y_pred = model.predict(X_test)
# 畫出 data
plt.scatter(X,Y, label='X,Y')
plt.scatter(X_test, Y_test, label='X_test, Y_test')
plt.scatter(X_test, Y_pred, label='pred')
# 畫出 線
x2 = np.linspace(0,1,100)
print(biases[0])
print(weights[0])

y2 =(weights[0]*x2+biases[0])
plt.plot(x2, y2, '-r', label='weights')

plt.show()


