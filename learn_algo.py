import numpy as np
import copy
from network import *

# loss functions
def cost(output, labels): # difference between output values an expected values. Only used if you have labeled data
  result = 0
  for x, y in zip(output, labels):
    result += (x - y) ** 2
  return(result / len(output))
  
def fitness(cost): # fitness increases as the network inproves unlike cost
  return (1 / cost)
#learn algorithms

def mutate(list): # used in cross to add "mutations" to a network layer
  mutation_rate = .05 # controlles how often mutations occur
  mutation_amount = .05 # controlles amount of mutation applied
  new_list = []
  for i in list:
    r = np.random.rand()
    if r < mutation_rate:
      new_list.append(i + (np.random.randn() * mutation_amount)) # i use randn here because it's convenient and more natural, if that makes sense
    else:
      new_list.append(i)
  return new_list
      
def cross(mating_list): # takes in a mating_list and outputs a new population
  #break list into mating_pairs
  new_population = []
  for i in mating_list:
    net1 = i[0]
    net2 = i[1]
    #break networks into layers
    for layer_1, layer_2 in zip(net1.layers, net2.layers):
      #flatten layers so they can be crossed
      layer_weights_1 = layer_1.get_weights()
      layer_weights_2 = layer_2.get_weights()
      weights_1 = layer_weights_1.flatten()
      weights_2 = layer_weights_2.flatten()
      crossover_rate = .5 # how likely "genes" will swapped
      new_weights_1 = []
      new_weights_2 = []
      #break weights into pairs
      for x, y in zip(weights_1, weights_2):
        r = np.random.rand() # this determines if a gene will cross over
        if r < crossover_rate:
          new_weights_1.append(y) # swap the gene pairs
          new_weights_2.append(x)

        else:
          new_weights_1.append(x)
          new_weights_2.append(y)
    
      layer_1.update_weights(np.array(mutate(new_weights_1)).reshape(layer_weights_1.shape)) # mutate weights, then convert them to theie original shape
      layer_2.update_weights(np.array(mutate(new_weights_2)).reshape(layer_weights_2.shape))
      # repeat the same for biases
      layer_bias_1 = layer_1.get_bias()
      layer_bias_2 = layer_2.get_bias()
      bias_1 = layer_bias_1.flatten()
      bias_2 = layer_bias_2.flatten()
    
      crossover_rate = .5 # how likely "genes" will swapped
      new_bias_1 = []
      new_bias_2 = []
      #break weights into pairs
      for x, y in zip(bias_1, bias_2):
        r = np.random.rand() # this determines if a gene will cross over
        if r < crossover_rate:
          new_bias_1.append(y) # swap the gene pairs
          new_bias_2.append(x)

        else:
          new_bias_1.append(x)
          new_bias_2.append(y)
          
      layer_1.update_bias(np.array(mutate(new_bias_1)).reshape(layer_bias_1.shape))
      layer_2.update_bias(np.array(mutate(new_bias_2)).reshape(layer_bias_2.shape))

    new_population.append(copy.deepcopy(net1))
    new_population.append(copy.deepcopy(net2))
    
  return(new_population)
  
      
  #break layers into weights and biases
  #cross weights and biases individually
  #mutate(weights)
  #mutate(biases)
  

def genetic_algorithm(population, cost): # creats new population based on cost values
  fitness_list = []
  for i in cost:
    fitness_list.append(fitness(i))

  mating_list = [] #list of lists representing the pairs of networks to be crossed over
  for i in range(int(len(population) / 2)):
    mating_pair = []
    for i in  range(2):
      r = np.random.rand() * sum(fitness_list) # creates a random number used for pool selection
      index = 0
      
      while r > 0: # explanation of algorithm: https://www.youtube.com/watch?v=ETphJASzYes&t=525s
        r -= fitness_list[index]
        index += 1
      index -= 1
      mating_pair.append(population[index])
    mating_list.append(mating_pair)
  
  return cross(mating_list)
  