# MNIST-NN
##### Shallow multilayer perceptron neural network for classifiying hand-written digits using the MNIST dataset. 
This project is for learning purposes only and is an attempt at implementing [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) and converting it into a fully matrix form
## Features
- From first principles (no helper libraries)
- Cross-entropy cost function
- Optimized weight initialisation
- Stochastic gradient descent (using mini batches)
- Fully vectorised (elements in mini batch processed simultaneously in matrix)
- L2 regularization
- Dynamic updating cost vs epoch graph
## Cost vs Epochs
![Alt text](CostvEpochs.png?raw=true "Cost vs Epochs")
## Feedforward Randomly Selected Images in Test Set
##### Original Image | Prediction | Confidence
![Alt text](Example.png?raw=true "Random Feedforward")

Best classification accuracy (after 30 epochs): 96.08%
## Architecture
- Learning Rate: 0.9329610181548651
- Layers: (784, 30, 10)
- Training epochs: 15
- Mini batch size: 20
