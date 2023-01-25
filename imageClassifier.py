import tensorflow as tf
import numpy as np
from tensorflow import keras

import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Here images are represented as numbers ranging from 0 to 255, label is just a number representing the type of image
# print(train_labels[0])
# print(train_images[0])

# cmap, vmin and vmax values are used to represent the image in gray scale
# plt.imshow(train_images[0], cmap='gray', vmin=0, vmax=255)
# plt.show()

# Define our neural network structure
# It's a good practice to define the input and the output layer first then the hidden layer/s
model = keras.Sequential([
    # input layer (28x28) image, Flatten: means we will have a single column of nodes representing each pixel
    # so in this case we will have 784 rows or pixels as 28 times 28 equals 784
    keras.layers.Flatten(input_shape=(28, 28)),

    # hidden layer: here we select an arbitrary number of nodes, it doesn't have to be power of 2
    # basically this number is selected so that our neural net find the hidden patterns in the input given
    # more no. nodes and more no. hidden layers might result in an effective model but may be time-consuming
    # earlier sigmoid fn was used but now relu is used which returns the value or 0, basically removes all -ve values
    keras.layers.Dense(units=128, activation=tf.nn.relu),

    # output layer: will consist of 10 nodes representing each label from 0-9 or each item
    # The ouput layer wll return some probab will corresponds to how close an image is to the item
    # whichever node has the higher probab will be chosen as the label for that particular input image
    keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

# Compile the model, getting our model ready for training
# loss function tells how wrong our model is predicting and based ton hat new weights are assigned to the synapses'
# optimizer comes in when our model is super wrong, it basically makes changes to the weights which will help the model
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy')

# train the model
# epoch means running the above line 5 times, so we are basically assigning new weights like 5 times
model.fit(train_images, train_labels, epochs=5)

plt.imshow(test_images[3], cmap='gray', vmin=0, vmax=255)
plt.show()

# Test the model
test_loss = model.evaluate(test_images, test_labels)

predictions = model.predict(test_images)

print(list(predictions[3]).index(max(predictions[3])))


print("code ran successfully")