# demonstration of example layer
#
# a = np.array([[2, 1, 2, 3, 3],   
#              [2, 1, 1, 3, 1],
#              [2, 2, 1, 2, 1],
#              [2, 3, 3, 2, 2],
#              [1, 1, 2, 1, 3]])
# ^ weights and biases connecting 5 imput neurons and 5 output neurons
# a row represents the connections between one output neuron and all imput neurons
#
# b = np.array([2, 1, 1, 1, 1])
# ^ activations in the imput layer
#
# c = np.array([-10, -10, -10, -10, -10])
# ^ biases for output neurons
#
# def ReLU(x):
#  return np.maximum(0, x)
#
# output = ReLU(a.dot(b) + c)
# ^ activations in the output layer excluding biases

import numpy as np
import copy
from activation import *
from learn_algo import *

# high level class
class Network:
  def __init__(self, layers = []):
    self.layers = layers
    self.learn_algo = None # e.g back_prop defined in learn_algos
    self.loss_function = None # function used to quanitfy network performance defined in learn_algos

# function to add layer to network
  def add(self, layer):
    self.layers.append(layer)

# function to run network on a numpy vector
  def run(self, data):
    output = data
    for i in self.layers:
      output = i.forward_prop(output) # forward_prop is defined in the layer subclass used
    return output

# data and labels are lists of lists
  def create_population(self, population_size):#, learn_algo, loss_function): # runs and optimizes population of networks for on batch of data
    population = []
    layers = []
    Net = None

    for i in range(population_size):
      layers = []
      for j in self.layers:
        layer = copy.deepcopy(j)
        layer.rand_init()
        layers.append(layer)
      population.append(Network(layers))
    return(population)
        
  def train_reinforcement(self, data, labels, population):
    costs = []
    for i in population:
      local_cost = []
      for input, label in zip(data, labels):
        local_cost.append(cost(i.run(input), label))
        #print('input', input, 'output', i.run(input), 'label', label)
        #print('local cost', local_cost)
      costs.append(np.mean(local_cost))

    print("minimum cost", min(costs), "average cost", np.average(costs))
    return genetic_algorithm(population, costs)
      
# base class for all layer types
class Layer:
  def __init__(self):
    self.input_size = None
    self.output_size = None
    self.input = None
    self.output = None
    self.weights = None
    self.bias = None

# These functions where specificly for implementing a genetic algorithm. I couldn't figure out how to fit it into one function along with back prop, so I just deal with the weights and biases outside of the layer class.    
  def get_weights(self):
    return self.weights

  def rand_init(self):
    self.weights = np.random.rand(self.output_size, self.input_size) - 0.5
    self.bias = np.random.rand(1, self.output_size) - 0.5

  def get_bias(self):
    return self.bias

  def update_weights(self, weights):
    self.weights = weights

  def update_bias(self, bias):
    self.bias = bias

# place holders for the functions in the specific layers 
  def forward_prop(self, input):
    raise NotImplementedError
    
  def back_prop(self, loss, learning_rate):
    raise NotImplementedError

# sub class of Layer
# this is the most basic layer type similar to the keras dense layer
class Dense(Layer):
# input and output size correspond to number of nurons and activation is the function applied to the output
  def __init__(self, input_size, output_size, activation):
    # here I initilize all the values as random numbers between -.5 and .5
    self.weights = np.random.rand(output_size, input_size) - 0.5
    self.bias = np.random.rand(1, output_size) - 0.5
    self.activation = activation
    self.input_size = input_size
    self.output_size = output_size
    
# uses weights and biases to calculate the inputs for the next layer
  def forward_prop(self, input_data):
    self.input_data = input_data
    self.output = np.dot(self.weights, self.input_data) # computing the pruduct of the weights for each output neuron
    self.output = self.output + self.bias # adding the bias
    self.output = self.activation.function(self.output) # applying the activation function
    self.output = np.reshape(self.output, self.output_size)
    return self.output