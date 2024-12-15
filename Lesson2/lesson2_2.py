#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "柯博文老師 Powen Ko, www.powenko.com"

from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np


iris = datasets.load_iris()

category=3
dim=4
x_train , x_test , y_train , y_test = train_test_split(iris.data,iris.target,test_size=0.2)
y_train2=tf.keras.utils.to_categorical(y_train, num_classes=(category))
y_test2=tf.keras.utils.to_categorical(y_test, num_classes=(category))

print("x_train[:4]",x_train[:4])
print("y_train[:4]",y_train[:4])
print("y_train2[:4]",y_train2[:4])