# -*- coding: utf-8 -*-
"""
Created on Sat May 15 11:23:18 2021

@author: doist
"""

import random
import copy
from tqdm import tqdm

class wolf:
    def __init__(self, fitness, dim, lower_bound, upper_bound, seed, data, true_labels):
        self.rnd = random.Random(seed)
        self.weights = [0.0 for i in range(dim)]
        
        for i in range(dim):
            self.weights[i] = ((upper_bound - lower_bound) * self.rnd.random() + lower_bound)
            
        self.fitness = fitness(self.weights, data, true_labels)
        
def gwo(fitness, max_iter, n, dim, lower_bound, upper_bound, data, true_labels):
    GWO_summary = {}
    
    rnd = random.Random(0)
    
    # create n random wolves
    population = [wolf(fitness, dim, lower_bound, upper_bound, i, data, true_labels) for i in range(n)]
    
    # Sort population based on fitness
    population = sorted(population, key = lambda temp: temp.fitness, reverse=False)
    
    # Determine alpha, beta and gamma
    alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[:3])
    
    # Main loop of GWO
    k = 0
    while k < max_iter:
        # print iteration number and best fitness value so far
        if k % 1 == 0 and k > 1:
            print('Iter = ' + str(k) + " best fitness = %.3f" % alpha_wolf.fitness)
            print('Best weights = ', alpha_wolf.weights)
        
        # Add info to PSO summary
        GWO_summary[k] = {}
        GWO_summary[k]['best_solution'] = alpha_wolf.weights
        GWO_summary[k]['fitness_best_solution'] = alpha_wolf.fitness
        GWO_summary[k]['population'] = population
            
        # linearly decreased from 2 to 0
        a = 2*(1-k/max_iter)
        
        # Updating each population member with the help of best three members
        print('\nUpdating population in iteration ' + str(k) + '\n')
        for i in range(n):
            A1, A2, A3 = a*(2*rnd.random()-1), a*(2*rnd.random()-1), a*(2*rnd.random()-1)
            C1, C2, C3 = 2*rnd.random(), 2*rnd.random(), 2*rnd.random()
            
            w1 = [0.0 for i in range(dim)]
            w2 = [0.0 for i in range(dim)]
            w3 = [0.0 for i in range(dim)]
            w_new = [0.0 for i in range(dim)]
            for j in range(dim):
                w1[j] = alpha_wolf.weights[j] - A1 * abs(
                  C1 - alpha_wolf.weights[j] - population[i].weights[j])
                w2[j] = beta_wolf.weights[j] - A2 * abs(
                  C2 -  beta_wolf.weights[j] - population[i].weights[j])
                w3[j] = gamma_wolf.weights[j] - A3 * abs(
                  C3 - gamma_wolf.weights[j] - population[i].weights[j])
                w_new[j]+= w1[j] + w2[j] + w3[j]
              
            for j in range(dim):
                w_new[j]/=3.0
                
            # Check bounds:
            for l in range(dim):
                if w_new[l] < lower_bound:
                    w_new[l] = lower_bound
                if  w_new[l] > upper_bound:
                    w_new[l] = upper_bound
              
            # fitness calculation of new solution
            fnew = fitness(w_new, data, true_labels)
  
            # greedy selection
            if fnew < population[i].fitness:
                population[i].weights = w_new
                population[i].fitness = fnew
                
        # On the basis of fitness values of wolves 
        # sort the population in asc order
        population = sorted(population, key = lambda temp: temp.fitness)
  
        # best 3 solutions will be called as 
        # alpha, beta and gama
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])
            
        k += 1
        
    return alpha_wolf, GWO_summary
            
            
            
            
            
            
            
            
        