import numpy as np

class Activation:
  def function(self, x):
    pass
    
  def derivative(self, x):
    pass

class ReLU(Activation):
  def function(x):
    return np.maximum(0, x)

  def derivative(x):
    return x > 0