import numpy as np
import random
from graph import Graph

class Network(object):

    # 'sizes' is a list containing the number of neurons
    # in each layer of the network. This is used to initialize
    # the network. e.g [5, 3, 2] is a network with 3 layers where
    # the input layer has 5 neurons, the single hidden layer has 
    # 3 neurons and the output layer has 2 neurons.
    def __init__(self, sizes, training_images, training_labels, test_images, test_labels, validation_images, validation_labels):

        self.num_layers = len(sizes)
        assert self.num_layers == 3, "Network only supports a single hidden layer!"
        self.sizes = sizes

        # Create matrix of hidden weights
        self.Wh = np.random.randn(sizes[0], sizes[1]) * np.sqrt(2.0/sizes[0])

        # Create matrix of hidden biases
        self.Bh = np.full((1, sizes[1]), 0.1)

        # Create matrix of output weights
        self.Wo = np.random.randn(sizes[1], sizes[2]) * np.sqrt(2.0/sizes[1])

        # Create matrix of output biases
        self.Bo = np.full((1, sizes[2]), 0.1)

        # Create matrices from input data
        self.training_images = training_images
        self.training_labels = training_labels
        self.test_images = test_images
        self.test_labels = test_labels
        self.validation_images = validation_images
        self.validation_labels = validation_labels
    
    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    # Derivative of the activation function
    @staticmethod
    def sigmoid_derivative(z):
        return Network.sigmoid(z) * (1.0 - Network.sigmoid(z))
    
    def feedforward(self, X):
        # Input layer --> Hidden layer
        H = self.sigmoid(X @ self.Wh + self.Bh)
        # Hidden layer --> Output layer
        return self.sigmoid(H @ self.Wo + self.Bo)
    
    # Stochastic gradient descent
    # This is the outer-loop stepping through
    # epochs and splitting batches
    def SGD(self, epochs, eta, mini_batch_size, lmbda = None, monitor_text = False, monitor_cost = False, monitor_eval_accuracy = False, monitor_train_accuracy = False):
        
        if monitor_cost or monitor_eval_accuracy or monitor_train_accuracy:
            graph = Graph()
            if monitor_cost:
                graph.add_line("cost")
                graph.add_axes_labels("Epochs", "Cost")
            if monitor_eval_accuracy or monitor_train_accuracy:
                graph.add_axes_labels("Epochs", "Accuracy")
                if monitor_eval_accuracy:
                    graph.add_line("eval_acc")
                if monitor_train_accuracy:
                    graph.add_line("train_acc", color="blue")

        n = len(self.training_images)
        n_test = len(self.test_images)

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
                self.backpropogate(batch, labels, mini_batch_size, eta, lmbda)

            if monitor_cost or monitor_eval_accuracy or monitor_train_accuracy:
                if monitor_cost:
                    graph.add_data("cost", j, self.total_cost(self.training_images, self.training_labels, lmbda))
                if monitor_eval_accuracy:
                    self.test_images, self.test_labels = unison_shuffled_copies(self.test_images, self.test_labels)
                    graph.add_data("eval_acc", j, self.evaluate(self.test_images[:100], self.test_labels[:100]))
                if monitor_train_accuracy:
                    graph.add_data("train_acc", j, self.evaluate(self.training_images[:100], self.training_labels[:100]))
                
                graph.update()
            
            if monitor_text:
                print("Epoch {0}: {1} / {2}".format(j + 1, self.evaluate(self.test_images, self.test_labels), n_test))
            else:
                print ("Epoch {0} complete".format(j + 1))
        
        if monitor_cost or monitor_eval_accuracy or monitor_train_accuracy:
            graph.ioff()


    def backpropogate(self, X, Y, batch_size, eta, lmbda=None):

        # NOTE * is element-wise multiplication (Hadamard product)
        # NOTE @ is matrix multiplication
        # NOTE np.dot on matrices is equivalent to @

        ## Feedforward, storing Z-values and activations

        # Compute hidden Z-values and activations
        Zh = X @ self.Wh + self.Bh
        Ah = self.sigmoid(Zh)

        # Compute output Z-values and activations
        Zo = Ah @ self.Wo + self.Bo
        Ao = self.sigmoid(Zo)

        # Compute output layer error
        Eo = (Ao - Y) * self.sigmoid_derivative(Zo)

        # Computer hidden layer error
        Eh = (Eo @ self.Wo.T) * self.sigmoid_derivative(Zh)

        # L2 regularization scaling factor
        if lmbda:
            sf = (1.0 - (eta * lmbda) / eta)
        else:
            sf = 1.0
        
        # Compute eta / m for efficiency
        ebs = eta / batch_size

        # Apply weight update rule
        self.Wo = sf * self.Wo - ebs * (Ah.T @ Eo)
        self.Wh = sf * self.Wh - ebs * (X.T @ Eh)

        # Apply bias update rule
        self.Bo -= ebs * np.sum(Eo, axis=0)
        self.Bh -= ebs * np.sum(Eh, axis=0)
    
    # Feedforward input and count how many correct answers were found
    def evaluate(self, X, Y):
        return sum(np.argmax(x) == np.argmax(y) for x, y in zip(self.feedforward(X), Y))

    # Calculate cross-entropy cost
    def total_cost(self, X, Y, lmbda):

        A = self.feedforward(X)
        intermediate = np.nan_to_num(Y * np.log(A) + (1.0 - Y) * np.log(1.0 - A))
        C_0 = (-1.0 / len(X)) * np.sum(intermediate.flatten())
        if lmbda:
            return C_0 + (lmbda / (2 * len(X))) * sum(w ** 2 for w in np.append(self.Wh, self.Wo).flatten())
        else:
            return C_0

# Shuffle two arrays in unison and return copies
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]