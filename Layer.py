import numpy as np

#cool let's define our activation functions
sigmoid = lambda z : 1/(1+np.exp(-z))
sigmoid_grad = lambda z : z*(1-z)

relu = lambda z: (z>0)*Z
relu_grad = lambda z : (z>0)*1

softmax = lambda z: np.exp(z)/np.sum(np.exp(z))
softmax_grad = lambda z: z*(1-Z)

cross_entropy_loss = lambda y, y_hat: -np.sum(y*np.log(y_hat+.00001)+(1.00001-y)*np.log(1.00001 - y_hat))
cross_entropy_grad = lambda y, y_hat: -y*(y_hat+.000001)**-1 + (1-y)*(1.000001-y_hat)**-1

activation_dict = {"sigmoid":[sigmoid, sigmoid_grad], "relu":[relu, relu_grad], "softmax":[softmax, softmax_grad]}

class Layer():
  def __init__(self, inputs, outputs, activation):
    self.W_shape = (outputs, inputs)
    self.b_shape = (outputs)
    
    self.W = np.random.random(self.W_shape)/100
    self.b = np.random.random(self.b_shape)/100
    
    self.activation = activation_dict[activation][0]
    self.activation_grad = activation_dict[activation][1]

    #this comes in handy later
    self.prev = None
    self.next = None

  def forward(self, z):
    return sigmoid(np.dot(self.W, z)+self.b)

  def delta(self, delta_prev, L_prev, z):
    return np.dot(L_prev.W.T, delta_prev)*self.activation_grad(z)

  def grad(self, delta, a):
    dW = np.dot(a, delta)
    db = delta
    return [dW, db]
