"""
    Author : Allumolu Harshit (<allumoluharshit@gmail.com>)
    National Institute of Technology, Warangal.
    
    Summary : 
    
        Feed Forward Neural Network :-
                    
            -- Activations supported - sigmoid and softmax(only output layer)
            -- Limit on number of layer - No limit
            -- Must specify number of neurons and number of inputs to a layer.
            -- Limit on dimensions - No limit
            -- Expects input in the shape - (n_examples,n_features)
            -- Expects training output vector as a one hot encoded vector (only needed when number of output units > 1).
            -- However class labels can be given to test the neural network (no need of one hot encoding).
"""

import numpy as np

class Layer():
    """
        Class name : Layer
        Purpose : To create a layer object that can be used
                  in a feed forward neural network
    """
    def __init__(self,n_input,n_neurons):
        """
            Method name : __init__()
            Purpose : Initializing all the required variables
            Arguments :
                - n_neurons : Number of neurons in the layer
                - n_input : Number of neurons in the previous layer 
                            (or) input layer (for first hidden layer)
        """
        self.input_shape = n_input
        self.output_shape = n_neurons
        self.dimensions = (self.input_shape,self.output_shape)
        self.weights = np.random.randn(self.input_shape,self.output_shape)
        self.bias = np.random.randn(1,self.output_shape)
        self.output = list()
        self.delta = list()
        self.activation = ''


class FeedForwardNeuralNetwork():
    """
        Class name : FeedForwardNeuralNetwork
        Purpose : To create a new NN which is feedforward and 
                facilitating training and testing
    """
    def __init__(self):
        """
            Method name : __init__()
            Purpose : Initializing a neural network
        """
        self.network = list()
        self.n_layers = 0
        self.cost_function = 'lms'  #default loss function for binary classification
    
    
    def add(self,layer):
        """
            Method name : add
            Purpose : To facilitate adding layers to a network
            Arguments :
                - Layer object
        """
        self.network.append(layer)  #append the layer object at the end
        self.n_layers += 1          #increment number of layers

    
    def compile(self):
        """
            Method name : compile
            Purpose : To check whether the neural network is ready
                    for training or not
            Returns : True if fine, else False
        """
        self.network[0].activation = 'sigmoid'  #default activation function
        for i in range(1,self.n_layers):
            if self.network[i].input_shape != self.network[i-1].output_shape:
                return False
            self.network[i].activation = 'sigmoid'
            if i == self.n_layers-1:
                if self.network[i].output_shape > 2:
                    self.network[i].activation = 'softmax'  #if multiclass classification - change activation
                    self.cost_function = 'ace'  #if multiclass classification - change loss function
        return True
    
    
    def summary(self,print_weights=False):
        """
            Method name : summary
            Purpose : Prints all the parameters of the neural network
            Arguments :
                - print_weights : False (default) if made True then all weights are printed
        """
        print('\n***** Summary *****\n')
        print('Number of hidden layers = {}'.format(self.n_layers-1))
        print('\nDimensions - ')
        for i in range(self.n_layers):
            if i == self.n_layers-1:
                print('\n{}. Output layer - '.format(i+1))
            else:
                print('\n{}. Hidden layer - '.format(i+1))
            print('Shape of weights - {}'.format(self.network[i].dimensions))
            if print_weights:
                print(self.network[i].weights)
            print('Shape of bias - {}'.format((1,self.network[i].output_shape)))
            if print_weights:
                print(self.network[i].bias)
        print('\n*******************\n')
    

    def sigmoid(self,x):
        """
            Method name : sigmoid
            Purpose : To calculate sigmoid activation
            Arguments :
                - x : single value or a vector
            Returns : 
                - sigmoid(x)
        """
        return 1 / (1 + np.exp(-x))     #sigmoid activation function
    

    def sigmoid_derivative(self,y):
        """
            Method name : sigmoid_derivative
            Purpose : To calculate the value of sigmoid derivate
            Arguments : 
                - y : output from the final layer
            Returns :
                - y * (1 - y)
        """
        return y * (1 - y)      #derivative of sigmoid activation function
    
    
    def softmax(self,x):
        """
            Method name : softmax
            Purpose : To calculate softmax activation
            Arguments :
                - x : single value or a vector
            Returns : 
                - softmax(x)
        """
        exp = np.exp(x)
        return exp / np.sum(exp,axis=1,keepdims=True)   #softmax activation
    
    
    def compute_cost(self,a,y):
        """
            Method name : compute_cost
            Purpose : To compute the cost of the network
            Arguments :
                - y : Expected output (one hot encoded)
                - a : Actual output
            Returns : 
                - Respective cost
        """
        if self.cost_function == 'lms':
            return np.sum(np.square(a - y)) / 2
        elif self.cost_function == 'ace':
            logs = np.log(a)
            return -np.sum(y * logs)/a.shape[0]
    
    
    def forward_propagate(self,row):
        """
            Method name : forward_propagate
            Purpose : To take input through the network and get the output
            Arguments :
                - row : A row or list of rows in x_train
        """
        for i in range(self.n_layers):
            if i == 0:
                self.network[i].output = self.sigmoid(np.dot(row,self.network[i].weights) + self.network[i].bias)
            else:
                self.network[i].output = np.dot(self.network[i-1].output,self.network[i].weights) + self.network[i].bias
                if self.network[i].activation == 'sigmoid':
                    self.network[i].output = self.sigmoid(self.network[i].output)
                elif self.network[i].activation == 'softmax':
                    self.network[i].output = self.softmax(self.network[i].output)
    
    
    def backward_propagate(self,a,y):
        """
            Method name : backward_propagate
            Purpose : To calculate the delta backwards through the network
            Arguments : 
                - y : Expected output (one hot encoded)
                - a : Actual output
        """
        for i in reversed(range(self.n_layers)):
            if i == self.n_layers - 1:
                self.network[i].delta = (a - y) / a.shape[0]
                if self.network[i].activation == 'sigmoid':
                    self.network[i].delta *= self.sigmoid_derivative(a) * a.shape[0]
            else:
                self.network[i].delta = self.sigmoid_derivative(self.network[i].output)
                self.network[i].delta *= np.dot(self.network[i+1].delta,self.network[i+1].weights.T)
    
    
    def update_weights(self,row,l_rate):
        """
            Method name : update_weights
            Purpose : To update weights of all the layers after backpropagation
            Arguments : 
                - row : A row or list of rows in x_train
                - l_rate : Learning rate
        """
        for i in range(self.n_layers):
            if i == 0:
                self.network[i].weights -= (l_rate * np.dot(row.T,self.network[i].delta))
            else:
                self.network[i].weights -= (l_rate * np.dot(self.network[i-1].output.T,self.network[i].delta))
            self.network[i].bias -= (l_rate * np.sum(self.network[i].delta,axis=0))

    
    def fit(self,x_train,y_train,l_rate=0.1,n_epochs=500,batch_size=32):
        """
            Method name : fit
            Purpose : To train the neural network on given data
            Arguments :
                - x_train : Training features
                - y_train : Training output (one hot encoded labels)
                - l_rate : Learning rate
                - n_epochs : Number of epochs
                - batch_size : training batch size
            Returns :
                - epochs : Number of epochs training is done
                - errors : A list of errors
        """
        if not self.compile():
            print('!!!!! ERROR !!!!!')
            return
        m = x_train.shape[0]
        errors = list()
        for i in range(n_epochs):
            error = 0
            for j in range(0,m,batch_size):
                batch = min(batch_size,m-j)
                row = x_train[j:j+batch]
                y = y_train[j:j+batch]
                self.forward_propagate(row)     #forward propagate the input
                a = self.network[-1].output
                e = self.compute_cost(a,y)
                error += e
                self.backward_propagate(a,y)    #backward propagate the error
                self.update_weights(row,l_rate) #update the weights
            if i > 0 and error > errors[-1]:
                n_epochs = i
                break
            else:
                errors.append(error)
            if i==0 or (i+1)%(n_epochs/5) == 0:
                print('Error after epoch {} - {}'.format(i+1,error))
        return errors, n_epochs

    
    def predict(self,x):
        """
            Method name : predict
            Purpose : Given a row, it predicts the class label
            Arguments :
                - x : input row
            Returns : 
                - class label
        """
        self.forward_propagate(x)
        a = self.network[-1].output
        if self.network[-1].output_shape == 1:
            if a[0][0] > 0.5:
                return 1
            else:
                return 0
        else:
            return np.argmax(a[0])

    
    def test(self,x_test,y_test):
        """
            Method name : test
            Purpose : To calculate accuracy of the neural network
                      for a given test dataset
            Arguments : 
                - x_test : test input features
                - y_test : test_output labels
            Returns :
                - Accuracy
        """
        n = y_test.shape[0]
        accuracy = 0
        for i,x in enumerate(x_test):
            if self.predict([x]) == y_test[i][0]:
                accuracy += 1
        accuracy = (accuracy/n)*100
        accuracy = round(accuracy,2)
        return accuracy
