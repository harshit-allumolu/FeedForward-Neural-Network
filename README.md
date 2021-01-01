# FeedForward-Neural-Network
Implementation of a basic feedforward neural network in python from scratch

# FFNN.py
The source code for object oriented implementation of a feedforward neural network. 
There are 2 classes - (i) Layer and (ii) FeedForward Neural Network

# Layer
Initialises a layer in a general neural network with required variables and details. It contains the data related to that layer like input shape, number of nodes, weights etc.

# FeedForwardNeuralNetwork
Initialises a network
Facilitates addition of layers
Sigmoid activation function is used in all layers
If number of target classes > 2, then softmax activation function is used in the output layer

# xor.py
A sample to show the usage of FFNN.py in training a model for implementation of XOR logic gate
