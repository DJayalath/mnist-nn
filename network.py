import numpy as np
import random

class Network(object):

    # 'sizes' is a list containing the number of neurons
    # in each layer of the network. This is used to initialize
    # the network. e.g [5, 3, 2] is a network with 3 layers where
    # the input layer has 5 neurons, the single hidden layer has 
    # 3 neurons and the output layer has 2 neurons.
    def __init__(self, sizes, training_images):

        self.num_layers = len(sizes)
        self.sizes = sizes

        # Create matrix of hidden weights
        self.Wh = np.random.randn(sizes[0], sizes[1])

        # Create matrix of hidden biases
        self.Bh = np.random.randn(1, sizes[1])

        # Create matrix of output weights
        self.Wo = np.random.randn(sizes[1], sizes[2])

        # Create matrix of output biases
        self.Bo = np.random.randn(1, sizes[2])

        # Create matrix of input data
        # 50000 rows, 784 columns
        self.X = np.array(np.reshape(training_images, (50000, 784)))
    
    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def feedforward(self):
        H = self.sigmoid(np.dot(self.X, self.Wh) + self.Bh)
        O = self.sigmoid(np.dot(H, self.Wo) + self.Bo)
        print(np.argmax(O[0]))