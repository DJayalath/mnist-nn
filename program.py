import numpy as np
import matplotlib.pyplot as plt
from network import Network

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

net = Network([784, 30, 10], training_images)
net.feedforward()
# net.SGD(training_data, 30, 10, 0.5, test_data=test_data)