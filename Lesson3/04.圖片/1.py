#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "柯博文老師 Powen Ko, www.powenko.com"
import tensorflow as tf
import time

import numpy as np
import matplotlib.pyplot as plt
circle1=np.array([[1,1,1],
                 [1,0,1],
                 [1,1,1]])
plt.subplot(1,2,1)
plt.imshow(circle1)

cross1=np.array([[1,0,1],
                 [0,1,0],
                 [1,0,1]])
plt.subplot(1,2,2)
plt.imshow(cross1)
plt.show()


