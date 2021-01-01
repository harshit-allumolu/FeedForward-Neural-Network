import numpy as np
import matplotlib.pyplot as plt
from FFNN import Layer, FeedForwardNeuralNetwork

xor_gate = np.array([
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
])

def main():
    x_train = xor_gate[:,:2]        #extract training features
    y_train = xor_gate[:,2:3]       #extract training outputs

    model = FeedForwardNeuralNetwork()      #initialize a nn
    model.add(Layer(n_input=2,n_neurons=2)) #hidden layer
    model.add(Layer(n_input=2,n_neurons=1)) #output layer
    ret = model.compile()                   #compile the model
    if not ret:
        print("!!!!! ERROR IN NEURAL NETWORK !!!!!")
        return
    
    #print initial weights
    print("\nInitial Weights - ")
    model.summary(print_weights=True)

    #train the model
    l_rate = float(input('Learning rate = '))
    n_epochs = int(input('Number of epochs = '))
    errors, epochs = model.fit(x_train,y_train,l_rate=l_rate,n_epochs=n_epochs,batch_size=1)
    
    #print final weights
    print("\nFinal Weights - ")
    model.summary(print_weights=True)

    #plot the data
    plt.title('Error vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.plot([i for i in range(epochs)],errors)
    plt.show()


if __name__ == "__main__":
    main()
