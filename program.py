import numpy as np
import matplotlib.pyplot as plt
from network import Network
import random

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']
    validation_images = data['validation_images']
    validation_labels = data['validation_labels']

# # Format each data set as tuples (x, y) where x is an array containing
# # the brightness values of the image and y is an array of length 10 
# # where 1.0 represents the index of the number the image represents
# training_data = [(x, y) for x, y in zip(training_images, training_labels)]
# test_data = [(x, y) for x, y in zip(test_images, test_labels)]
# validation_data = [(x, y) for x, y in zip(validation_images, validation_labels)]

# Reformat arrays to work with matrix system in network
training_images = np.array(np.reshape(training_images, (len(training_images), 784)))
training_labels = np.array(np.reshape(training_labels, (len(training_labels), 10)))
test_images = np.array(np.reshape(test_images, (len(test_images), 784)))
test_labels = np.array(np.reshape(test_labels, (len(test_labels), 10)))
validation_labels = np.array(np.reshape(validation_labels, (len(test_labels), 10)))
validation_images = np.array(np.reshape(validation_images, (len(validation_images), 784)))

# Train network
net = Network([784, 30, 10], training_images, training_labels)
net.SGD(30, 1.0, 10, test_images=test_images, test_labels=test_labels)

# Make random guesses to demonstrate accuracy until cancelled
while True:
    i = random.randint(0, len(validation_images) - 1)
    plt.imshow(validation_images[i].reshape(28, 28), cmap='gray')
    plt.show()
    print("Actual: " + str(np.argmax(validation_labels[i])))
    print("Guess: " + str(np.argmax(net.feedforward(validation_images[i]))))