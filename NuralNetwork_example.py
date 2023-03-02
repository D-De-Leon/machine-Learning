#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 23:43:39 2023

@author: daviddeleon
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

l0 = Dense(units = 1, input_shape = [1])
model = Sequential([l0])
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

size = 5

xs = [i for i in range(-1, size)]  # use a list comprehension to create the list

ys = [(2 * x) - 1 for x in xs]

print(xs)
print(ys)


# xs = np.array([-1.0 , 0.0 , 1.0 , 2.0 , 3.0 , 4.0])
# ys = np.array([-3.0, -1.0 , 1 , 3.0 , 5.0 , 7.0])

xs = np.array(xs)
ys = np.array(ys)

xs = xs.astype(float)
ys = ys.astype(float)

print(xs.dtype)

# xs = xs.reshape((size, 1))
# ys = ys.reshape((size, 1))


model.fit(xs,ys, epochs = 500)

print(model.predict([10.0]))
print("Here is what I learned: {}".format(l0.get_weights()))






