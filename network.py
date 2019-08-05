import numpy as np
import random

class Network(object):

    # 'sizes' is a list containing the number of neurons
    # in each layer of the network. This is used to initialize
    # the network. e.g [5, 3, 2] is a network with 3 layers where
    # the input layer has 5 neurons, the single hidden layer has 
    # 3 neurons and the output layer has 2 neurons.
    def __init__(self, sizes, training_images, training_labels):

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

        # WARNING: self.X needs to be changed to only hold
        # images in the current mini batch!
        # Create matrix of input data
        # 50000 rows, 784 columns
        self.training_images = np.array(np.reshape(training_images, (50000, 784)))
        # WARNING: self.Y needs to be changed to only
        # hold labels for the current mini batch!
        self.training_labels = np.array(np.reshape(training_labels, (len(training_labels), 10)))
        # np.array(np.reshape(training_labels, (len(training_labels, 10))))
    
    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    # Derivative of the activation function
    @staticmethod
    def sigmoid_derivative(z):
        return Network.sigmoid(z) * (1 - Network.sigmoid(z))
    
    def feedforward(self, X):
        H = self.sigmoid(np.dot(X, self.Wh) + self.Bh)
        return self.sigmoid(np.dot(H, self.Wo) + self.Bo)
    
    # Stochastic gradient descent
    # This is the outer-loop stepping through
    # epochs and splitting batches
    def SGD(self, epochs, eta, mini_batch_size, test_images = None, test_labels = None):

        # test_images = np.array(np.reshape(test_images, (len(test_images), 784)))
        # print("Epoch {0}: {1} / {2}".format(0, self.evaluate(test_images, test_labels), len(test_images)))

        if test_labels.any():
            n_test = len(test_images)
            test_labels = np.array(np.reshape(test_labels, (len(test_labels), 10)))
            test_images = np.array(np.reshape(test_images, (len(test_images), 784)))

        n = len(self.training_images)

        for j in range(epochs):

            # Shuffle data
            self.training_images, self.training_labels = unison_shuffled_copies(self.training_images, self.training_labels)

            # Split into mini batches
            mini_batches = [
                self.training_images[k: k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            mini_batch_labels = [
                self.training_labels[k: k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for batch, labels in zip(mini_batches, mini_batch_labels):
                self.backpropogate(batch, labels, eta)
            
            if test_labels.any():
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_images, test_labels), n_test))
            else:
                print ("Epoch {0} complete".format(j))

            

            

    # WARNING: calculates over entire mini batch simultaneously!
    def backpropogate(self, X, Y, eta):

        ## Feedforward, storing Z-values and activations

        # Compute hidden Z-values and activations
        Zh = np.dot(X, self.Wh) + self.Bh
        Ah = self.sigmoid(Zh)

        # Compute output Z-values and activations
        Zo = np.dot(Ah, self.Wo) + self.Bo
        Ao = self.sigmoid(Zo)

        # Compute output layer error
        Eo = (Ao - Y) * self.sigmoid_derivative(Zo)

        # Computer hidden layer error
        Eh = Eo * self.Wo * self.sigmoid_derivative(Zh)
        # np.dot(Eo @ self.Wo, self.sigmoid_derivative(Zh))

        # Compute derivative of cost with respect to weight
        nabla_w_o = np.dot(Eo, Ah)
        nabla_w_h = np.dot(Eh, X)

        # Set derivative of cost with respect to bias
        nabla_b_o = Eo # ???
        nabla_b_h = Eh

        self.Wo -= eta * nabla_w_o
        self.Wh -= eta * nabla_w_h

        self.Bo -= eta * nabla_b_o
        self.Bh -= eta * nabla_b_h
    
    def evaluate(self, X, Y):
        test_results = self.feedforward(X)

        total = 0
        for i in range(len(test_results)):
            if np.argmax(test_results[i]) == np.argmax(Y[i]):
                total += 1


        return total

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]