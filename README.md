# MNIST-NN
##### Deep multilayer perceptron neural network for classifiying hand-written digits using the MNIST dataset. 
This project is for learning purposes only and is an attempt at implementing, improving and converting the MLP example in [Neural Networks and Deep Learning (Michael Nielsen)](http://neuralnetworksanddeeplearning.com/) into a fully matrix form. **This network is written "from scratch"** i.e. no deep learning helper libraries (e.g. Keras, TensorFlow) were used. Mini batch processing is fully vectorized.
## Architecture (Best classification accuracy: 98.30%)
- Layers: (784, 512, 512, 10)
- Learning Rate: 0.2
- Training Epochs: 35
- Optimizer: Stochastic Gradient Descent
- Mini batch size: 128
- Cost function: Cross Entropy
- Activation function: ReLU
- Weight initialization: He
- Bias initialization: 0.1
- L2 regularization: None
- Dropout: None
## Usage
1. *(Optional)* Set desired hyper-parameters in `program.py`
2. Run `python3 program.py`
## Cost vs Epochs
![Alt text](CostvEpochs.png?raw=true "Cost vs Epochs")
## Example From Test Set
##### Original Image | Prediction | Confidence
![Alt text](Example.png?raw=true "Random Feedforward")
