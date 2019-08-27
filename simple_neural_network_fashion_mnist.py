# Load required packages
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras import models
from keras import layers
from keras import optimizers

# Load the Fashion MNIST data directly from the keras package and assign training and test data images and labels
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the data by dividing by maximum opacity of 255
train_images = train_images / 255.0
test_images = test_images / 255.0

# Develop a model with three layers (one input layer, one hidden layer and an output layer)
three_layer_model = Sequential()
three_layer_model.add(Flatten(input_shape = (28, 28)))
three_layer_model.add(Dense(128, activation = 'relu'))
three_layer_model.add(Dense(10, activation = 'softmax'))

# Compile and fit the model onto the training data
three_layer_model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
three_layer_model.fit(train_images, train_labels, epochs = 5, validation_split = 0.3)

# Compute the accuracy and loss of model performance on the test data
test_loss, test_acc = three_layer_model.evaluate(test_images, test_labels)
print("Model with three layers and five epochs -- Test loss:", test_loss * 100)
print("Model with three layers and five epochs -- Test accuracy:", test_acc * 100)

# Develop a model with five layers (one input layer, five hidden layers and an output layer)
five_layer_model = Sequential()
five_layer_model.add(Flatten(input_shape = (28, 28)))
five_layer_model.add(Dense(128, activation = 'relu'))
five_layer_model.add(Dense(128, activation = 'relu'))
five_layer_model.add(Dense(128, activation = 'relu'))
five_layer_model.add(Dense(10, activation = 'softmax'))

# Compile and fit the model onto the training data
five_layer_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
five_layer_model.fit(train_images, train_labels, epochs = 5, validation_split = 0.3)

# Compute the accuracy and loss of model performance on the test data
test_loss, test_acc = five_layer_model.evaluate(test_images, test_labels)
print("Model with five layers and five epochs -- Test loss:", test_loss * 100)
print("Model with five layers and five epochs -- Test accuracy:", test_acc * 100)

# Develop a model with ten layers (one input layer, eight hidden layers and an output layer)
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

# Compile and fit the model onto the training data
ten_layer_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
ten_layer_model.fit(train_images, train_labels, epochs = 5, validation_split = 0.3)

# Compute the accuracy and loss of model performance on the test data
test_loss, test_acc = ten_layer_model.evaluate(test_images, test_labels)
print("Model with ten layers and five epochs -- Test loss:", test_loss * 100)
print("Model with ten layers and five epochs -- Test accuracy:", test_acc * 100)

# Rerun the five layer model but now with 50 instead of 5 epochs
five_layer_model_50_epochs = five_layer_model.fit(train_images, train_labels, epochs = 50, validation_split = 0.3)

# Compute the accuracy and loss of model performance on the test data
test_loss, test_acc = five_layer_model.evaluate(test_images, test_labels)
print("Model with five layers and fifty epochs -- Test loss:", test_loss * 100)
print("Model with five layers and fifty epochs -- Test accuracy:", test_acc * 100)

# Compute the predictions from the five layer model with 50 epochs
predictions = five_layer_model.predict(test_images)
majority_vote = dict()
for i in range(len(predictions)):
    majority_vote[i] = np.argmax(predictions[i])
print(majority_vote)

# Plot accuracy and loss as functions of the number of epochs to better understand the loss/accuracy trade-off
plt.subplot(1, 2, 1)
plt.plot(five_layer_model_50_epochs.history['val_loss'], 'blue')
plt.plot(five_layer_model_50_epochs.history['loss'], 'red')
plt.legend(['Cross-validation data', 'Training data'], loc = 'upper left')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Model loss and accuracy over epochs for a six-layer neural network')

plt.subplot(1, 2, 2)
plt.plot(five_layer_model_50_epochs.history['val_acc'], 'blue')
plt.plot(five_layer_model_50_epochs.history['acc'], 'red')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.subplots_adjust(wspace = .35)

plt.show()

print(five_layer_model_50_epochs.history['val_acc'])
print(five_layer_model_50_epochs.history['val_loss'])
