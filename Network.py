import numpy as np
from Layer import *
from Activations import *


cross_entropy_loss = lambda y, y_hat: -np.sum(y*np.log(y_hat+.00001)+(1.00001-y)*np.log(1.00001 - y_hat))
cross_entropy_grad = lambda y, y_hat: y*(y_hat+.000001)**-1 - (1-y)*(1.000001-y_hat)**-1

class Network():
  def __init__(self, layers):
    self.input_layer = layers[0]
    self.output_layer = layers[-1]
    self.layers=layers

    i=1
    while i <len(layers)-1:
      if layers[i].W.shape[0] == layers[i-1].W.shape[1]:
        layers[i-1].next = layers[i]
      i+=1

  def forward(self,z):
    o = z
    for l in self.layers:
      o = l.forward(o)
    return o

  def grad(self, y, z):
    """takes label and input and computes gradient of loss function"""
    outputs = []
    o=z
    for l in self.layers:
      o=l.forward(o)
      outputs.append(o)

    #first the loss grad
    dC = cross_entropy_grad(y, o)

    #now to get the deltas
    deltas = []
    l = len(self.layers)-1
    last = l
    while l>=0:
      if l == last:
        deltas.append(dC*sigmoid_grad(outputs[l]))
      else:
        deltas.append(self.layers[l].delta(deltas[-1], self.layers[l+1], outputs[l]))
      l-=1
    deltas = list(reversed(deltas))
    #awesome now let's get the grads
    grad = []
    l = len(self.layers)-1
    last = l
    while l>=0:
      grad.append(self.layers[l].grad(deltas[l], outputs[l]))
      l-=1
    return list(reversed(grad))
