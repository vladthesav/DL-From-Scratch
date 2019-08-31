# DL-From-Scratch
Implementation of neural network in pure numpy.

# Fully Connected Network
The layer class is an interface to hide all the linear algebra from the user. It consists of a weight matrix (nxm numpy array), 
a bias vector (mx1 numpy array), and an actication (defined in Activations.py - using dictionary to store actication functions and their derivatives). This also includes helper methods and variables for intermediate results to do backpropation.

Use the Layer class to define each layer as follows:
    
    Layer(input_dim, output_dim, activation)
    e.g. Layer(500, 300, 'relu')
  
To constuct your network define all layers and put them in an array (in order).
    
    example:
    L1 = Layer(500, 300, 'relu')
    L2 = Layer(300, 50, 'relu')
    L3 = Layer(50, 10, 'softmax')
    
    layers = [L1, L2, L3]
    
    net = Network(layers)
    
    
Once you have your layers defined you can do the forward pass by using the forward method.
  
    example:
    output = net.forward(input)
    
Now you probably would want to train your network. So far I have only implemented stochastic gradient descent.
To get your gradient use the grad method as follows
    
    x = input
    y = output
    grad = net.grad(y,x)
    
 For now you will have to manually loop through and update all parameters in the network, I will work on this soon.
 
    lr = learning_rate
    
    for G, l in zip(grad, net.layers):
      #gradient of cost with respect to weight of this layer
      dC_dW = G[0]
      
      #gradient of cost with respect to bias of this layer
      dC_db = G[1]
      
      #awesome now let's update our parameters for this layer
      l.W -= lr*dC_dW
      l.b -= lr*dC_db
      
      
      
 Todo:
 - [x] Implement fully connected layer
 - [] Make better interface for training network
 - [] Get backpropagation to work for Convolutional layers
 - [] Make Convolutional layers to work with fully connected layers
 - [] Create dataloader class for user to test network on dataset
 - [] Implement more loss functions and optiizers (momentum, adam, nesterov, ect.)
      
      
    
