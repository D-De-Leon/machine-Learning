#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 22:18:45 2023

@author: daviddeleon
"""

import tensorflow as tf

# Custom callback to stop training when accuracy reaches 95%
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if(logs.get('accuracy') > 0.95):
            print("\nReached 95% accuracy so cancelling training")
            self.model.stop_training = True

# Create an instance of the custom callback
callbacks = myCallback()

# Load the Fashion-MNIST dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the training and test images
training_images = training_images / 255
test_images = test_images / 255

# Define the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)), # Flatten the 28x28 input images
    tf.keras.layers.Dense(128, activation=tf.nn.relu), # Add a dense layer with 128 nodes and ReLU activation
    tf.keras.layers.Dense(10, tf.nn.softmax) # Add an output layer with 10 nodes and softmax activation
    ])

# Compile the model with Adam optimizer and sparse categorical cross-entropy loss
model.compile(optimizer= 'adam',
              loss='sparse_categorical_crossentropy',
              metrics= ['accuracy'])

# Train the model for 50 epochs with the custom callback
model.fit(training_images,
          training_labels,
          epochs = 50,
          callbacks = [callbacks])

# Evaluate the model on the test set
model.evaluate(test_images, test_labels)

# Make predictions on the test set and print the classification and actual label of the first image
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])
