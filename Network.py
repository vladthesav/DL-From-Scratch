import numpy as np
from Layer import *

#cool let's define our activation functions
sigmoid = lambda z : 1/(1+np.exp(-z))
sigmoid_grad = lambda z : z*(1-z)

relu = lambda z: (z>0)*Z
relu_grad = lambda z : (z>0)*1

softmax = lambda z: np.exp(z)/np.sum(np.exp(z))
softmax_grad = lambda z: z*(1-Z)

cross_entropy_loss = lambda y, y_hat: -np.sum(y*np.log(y_hat+.00001)+(1.00001-y)*np.log(1.00001 - y_hat))
cross_entropy_grad = lambda y, y_hat: -y*(y_hat+.000001)**-1 + (1-y)*(1.000001-y_hat)**-1
