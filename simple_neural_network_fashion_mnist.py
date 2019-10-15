# Import required packages and functions and set the session seed
import numpy as np
np.random.seed(1234)
from tensorflow import set_random_seed
set_random_seed(1234)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.layers import Dropout, SpatialDropout2D
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras import optimizers

# Load the Fashion MNIST data from Keras
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the image data by dividing through the maximum pixel value (=255)
train_images = train_images / train_images.max()
test_images = test_images / test_images.max()

# Build a simple three-layer (1 hidden layer) model
# The input size is 28 x 28 pixels and is flattened to a vector of length 784
# The activation function is RELU (rectified linear unit) and performs the 
# multiplication of input and weights (plus bias)
# The output (softmax) layer returns probabilities for all ten classes
three_layer_model = Sequential()
three_layer_model.add(Flatten(input_shape = (28, 28)))
three_layer_model.add(Dense(128, activation = 'relu'))
three_layer_model.add(Dense(10, activation = 'softmax'))

# Compile the model with accuracy metric and adam optimizer
# Sparse categorical cross-entropy is the loss function for integer labels
# Fit the model using 70 percent of the data and 10 epochs
three_layer_model.compile(loss = 'sparse_categorical_crossentropy', 
                          optimizer = 'adam', metrics = ['accuracy'])
three_layer_model.fit(train_images, train_labels, epochs = 10, 
                      validation_split = 0.3, verbose = 2)

# Compute and print the test loss and accuracy
test_loss, test_acc = three_layer_model.evaluate(test_images, test_labels)
print("Model with three layers and ten epochs -- Test loss:", test_loss * 100)
print("Model with three layers and ten epochs -- Test accuracy:", test_acc * 100)

# Similarly as before, build a five-layer (3 hidden layers) model
five_layer_model = Sequential()
five_layer_model.add(Flatten(input_shape = (28, 28)))
five_layer_model.add(Dense(128, activation = 'relu'))
five_layer_model.add(Dense(128, activation = 'relu'))
five_layer_model.add(Dense(128, activation = 'relu'))
five_layer_model.add(Dense(10, activation = 'softmax'))

# Compile the model with accuracy metric and adam optimizer
# Fit the model using 70 percent of the data and 10 epochs
five_layer_model.compile(loss = 'sparse_categorical_crossentropy', 
                         optimizer = 'adam', metrics = ['accuracy'])
five_layer_model.fit(train_images, train_labels, epochs = 10, 
                     validation_split = 0.3, verbose = 2)

# Compute and print the test loss and accuracy
test_loss, test_acc = five_layer_model.evaluate(test_images, test_labels)
print("Model with five layers and ten epochs -- Test loss:", test_loss * 100)
print("Model with five layers and ten epochs -- Test accuracy:", test_acc * 100)

# Similarly as before, build a ten-layer (8 hidden layers) model
ten_layer_model = Sequential()
ten_layer_model.add(Flatten(input_shape = (28, 28)))
ten_layer_model.add(Dense(128, activation = 'relu'))
ten_layer_model.add(Dense(128, activation = 'relu'))
ten_layer_model.add(Dense(128, activation = 'relu'))
ten_layer_model.add(Dense(128, activation = 'relu'))
ten_layer_model.add(Dense(128, activation = 'relu'))
ten_layer_model.add(Dense(128, activation = 'relu'))
ten_layer_model.add(Dense(128, activation = 'relu'))
ten_layer_model.add(Dense(128, activation = 'relu'))
ten_layer_model.add(Dense(10, activation = 'softmax'))

# Compile the model with accuracy metric and adam optimizer
# Fit the model using 70 percent of the data and 10 epochs
ten_layer_model.compile(loss = 'sparse_categorical_crossentropy', 
                        optimizer = 'adam', metrics = ['accuracy'])
ten_layer_model.fit(train_images, train_labels, epochs = 10, 
                    validation_split = 0.3, verbose = 2)

# Compute and print the test loss and accuracy
test_loss, test_acc = ten_layer_model.evaluate(test_images, test_labels)
print("Model with ten layers and ten epochs -- Test loss:", test_loss * 100)
print("Model with ten layers and ten epochs -- Test accuracy:", test_acc * 100)

# Compile the model with accuracy metric and adam optimizer
# Fit the model using 70 percent of the data and 50 epochs
three_layer_model_50_epochs = three_layer_model.fit(train_images, train_labels, 
                                                  epochs = 50, validation_split = 0.3,
                                                  verbose = 2)

# Compute and print the test loss and accuracy
test_loss, test_acc = three_layer_model.evaluate(test_images, test_labels)
print("Model with three layers and fifty epochs -- Test loss:", test_loss * 100)
print("Model with three layers and fifty epochs -- Test accuracy:", test_acc * 100)

# Plot loss as function of epochs
plt.subplot(1, 2, 1)
plt.plot(three_layer_model_50_epochs.history['val_loss'], 'blue')
plt.plot(three_layer_model_50_epochs.history['loss'], 'red')
plt.legend(['Cross-validation', 'Training'], loc = 'upper left')
plt.ylabel('Loss')
plt.xlabel('Epoch')

# Plot accuracy as function of epochs
plt.subplot(1, 2, 2)
plt.plot(three_layer_model_50_epochs.history['val_acc'], 'blue')
plt.plot(three_layer_model_50_epochs.history['acc'], 'red')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.subplots_adjust(wspace = .35)

# Include plot title and show the plot
plt.suptitle('Model loss and accuracy over epochs for a three-layer neural network')
plt.show()

# Calculate and print predictions versus actual labels
predictions = three_layer_model.predict(test_images)
for i in range(10):
  print("Prediction " + str(i) + ": " + str(np.argmax(np.round(predictions[i]))))
  print("Actual " + str(i) + ": " + str(test_labels[i]))

# Reload the data for a convolutional neural network
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Reshape the data to the correct format (the last 1 stands for greyscale)
train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

# Convert the image data to numeric data and normalize them
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images = train_images / train_images.max()
test_images = test_images / test_images.max()

# One-hot encode the label data
# Convert every number to a vector of the length of the number of categories
# The vector has zero everywhere except a one on the position of the number it 
# represents. Example: 3 = [0 0 0 1 0 0 0 0 0 0]
train_labels_bin = to_categorical(train_labels)
test_labels_bin = to_categorical(test_labels)

# Build a convolutional neural network with two convolutional layers
conv_model = Sequential()
conv_model.add(Conv2D(128, (3, 3), input_shape = (28, 28, 1)))
conv_model.add(Activation('relu'))
conv_model.add(MaxPooling2D(pool_size = (2, 2)))
conv_model.add(Conv2D(128, (3, 3)))
conv_model.add(Activation('relu'))
conv_model.add(MaxPooling2D(pool_size = (2, 2)))
conv_model.add(Flatten())
conv_model.add(Dense(128))
conv_model.add(Dense(10))
conv_model.add(Activation('softmax'))

# Compile and fit the model with adam optimizer and accuracy metric
# Categorical cross-entropy is the loss function for one-hot encoded labels and
# batch size equal to the number of neurons in the convolutional layers and 10 epochs
conv_model.compile(loss = "categorical_crossentropy", 
                   optimizer = 'adam', metrics = ['accuracy'])
conv_model.fit(train_images, train_labels_bin, batch_size = 128, 
               epochs = 10, verbose = 2)

# Compute and print the test loss and accuracy
test_loss, test_acc = conv_model.evaluate(test_images, test_labels_bin)
print("Convolutional model ten epochs -- Test loss:", test_loss * 100)
print("Convolutional model ten epochs -- Test accuracy:", test_acc * 100)

# Build a convolutional neural network with two convolutional layers
# Decrease number of neurons and add dropout to reduce overfitting
conv_model_reduce_overfit = Sequential()
conv_model_reduce_overfit.add(Conv2D(64, (3, 3), input_shape = (28, 28, 1)))
conv_model_reduce_overfit.add(Activation('relu'))
conv_model_reduce_overfit.add(MaxPooling2D(pool_size = (2, 2)))
conv_model_reduce_overfit.add(Dropout(0.5))
conv_model_reduce_overfit.add(Conv2D(64, (3, 3)))
conv_model_reduce_overfit.add(SpatialDropout2D(0.5))
conv_model_reduce_overfit.add(Activation('relu'))
conv_model_reduce_overfit.add(MaxPooling2D(pool_size = (2, 2)))
conv_model_reduce_overfit.add(Flatten())
conv_model_reduce_overfit.add(Dense(64))
conv_model_reduce_overfit.add(Dropout(0.5))
conv_model_reduce_overfit.add(Dense(10))
conv_model_reduce_overfit.add(Activation('softmax'))

# Compile and fit the model with adam optimizer and accuracy metric
# Categorical cross-entropy is the loss function for one-hot encoded labels and
# batch size equal to the number of neurons in the convolutional layers and 10 epochs
# Add early stopping to avoid overfitting
conv_model_reduce_overfit.compile(loss = "categorical_crossentropy", 
                   optimizer = 'adam', metrics = ['accuracy'])
conv_callback = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 3)
conv_model_reduce_overfit.fit(train_images, train_labels_bin, validation_split = 0.3,
               epochs = 10, verbose = 2, callbacks = [conv_callback], batch_size = 64)

# Compute and print the test loss and accuracy
test_loss, test_acc = conv_model_reduce_overfit.evaluate(test_images, test_labels_bin)
print("Convolutional model ten epochs reduced overfit -- Test loss:", test_loss * 100)
print("Convolutional model ten epochs reduced overfit -- Test accuracy:", test_acc * 100)

# Calculate and print predictions versus actual labels
predictions = conv_model_reduce_overfit.predict(test_images)
for i in range(10):
  print("Prediction " + str(i) + ": " + str(np.argmax(np.round(predictions[i]))))
  print("Actual " + str(i) + ": " + str(test_labels[i]))


