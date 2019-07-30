import numpy as np
from Activations import *

def conv(img, ker,bias, s=1):
  (k_h, k_w) = ker.shape # filter dimensions
  im_h, im_w = img.shape # image dimensions

  out_dim = int((im_h - k_h)/s)+1


  out = np.zeros((out_dim,out_dim))

  curr_y = out_y = 0
  while curr_y + k_h <= im_h:
    curr_x = out_x = 0
    while curr_x + k_w <= im_w:
      out[out_y, out_x] = np.sum(ker * img[curr_y:curr_y+k_h, curr_x:curr_x+k_w])+bias
      curr_x += s
      out_x += 1
    curr_y += s
    out_y += 1
  return out

def maxpool(X, k, s=2):
  x_h, x_w = X.shape
  
  out_dim = int((x_h-k)/s)+1
  out = np.zeros((out_dim, out_dim))

  curr_y = out_y = 0
  while curr_y +k <= x_h:
    curr_x=out_x = 0
    while curr_x <= k <= x_w:
      out[out_y, out_x] = np.max(X[curr_y:curr_y+k, curr_x:curr_x+k])
      curr_x += s
      out_x += 1
    curr_y += s
    out_y += 1
  return out

class ConvLayer():
    
  def __init__(self, input_dim, output_c, kernel_size, activation):
    self.kernels = np.random.random((output_c, kernel_size, kernel_size))
    self.biases = np.random.random((output_c))
    
    self.input_dim = input_dim
    self.kernel_size = kernel_size
    
    self.output_c = output_c
    self.activation = activation_dict[activation][0]
    self.activation_grad = activation_dict[activation][1]

  def conv(self, z):
    
    out = []
    for k, ker in enumerate(self.kernels):
      out.append(conv(z, ker, self.biases[k]))
    return np.array(out)

  def activate(self, z):
      return self.activation(z)
    
  def downsample(self, z):
    out = []
    for i,x in enumerate(z):
      out.append(maxpool(x,self.kernel_size))
    return np.array(out)

  def forward(self, x):
    z = F.activate(F.conv(x))
    return F.downsample(z)

  #this returns the flattened feature map to feed into a fully connected layer
  def flatten(self, z):
    return z.reshape((z.size,1)) 



F = ConvLayer(50, 10, 3, "relu")
x = np.random.random((50,50))
z = F.forward(x)


