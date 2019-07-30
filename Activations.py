import numpy as np

sigmoid = lambda z : 1/(1+np.exp(-z))
sigmoid_grad = lambda z : z*(1-z)

relu = lambda z: (z>0)*z
relu_grad = lambda z : (z>0)*1

softmax = lambda z: np.exp(z)/np.sum(np.exp(z))
softmax_grad = lambda z: z*(1-z)

tanh = lambda z : np.tanh(z)
tanh_grad = lambda z : 1-z*z

activation_dict = {"sigmoid":[sigmoid, sigmoid_grad], "relu":[relu, relu_grad],
                   "softmax":[softmax, softmax_grad], "tanh":[tanh, tanh_grad]}

