# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:04:31 2021

@author: doist
"""

from __future__ import division
import random

class Particle:
    def __init__(self, init_weights, num_dimensions):
        self.weights_i = []           # particle weights
        self.velocity_i = []          # particle velocity
        self.weights_best_i = []          # best weights individual
        self.ndcg_best_i = 0.0         # best error individual
        self.ndcg_i = 0.0              # error individual
        
        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.weights_i.append(init_weights[i])
            
    def evaluate(self, fitness, data, true_labels):
        #Evaluate current fitness
        self.ndcg_i = fitness(self.weights_i, data, true_labels)
        
        # Check if current weights are the best for individual particle
        if self.ndcg_i < self.ndcg_best_i or self.ndcg_best_i == -1:
            self.weights_best_i = self.weights_i
            
    def update_velocity(self, weights_best_group, num_dimensions):
        '''
        Update new particle velocity
        '''
        w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=2        # social constant

        for i in range(0,num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.weights_best_i[i]-self.weights_i[i])
            vel_social=c2*r2*(weights_best_group[i]-self.weights_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social
                
    def update_weights(self, bounds, num_dimensions):
        '''
        Update particle weights based on new velocity updates
        '''
        for i in range(0,num_dimensions):
            self.weights_i[i]=self.weights_i[i]+self.velocity_i[i]

            # adjust maximum weights if necessary
            if self.weights_i[i]>bounds[i][1]:
                self.weights_i[i]=bounds[i][1]

            # adjust minimum weights if neseccary
            if self.weights_i[i] < bounds[i][0]:
                self.weights_i[i]=bounds[i][0]
                
class PSO():
    def particle_swarm_optimization(fitness, init_weights, bounds, num_particles, maxiter, data, true_labels):
        num_dimensions = len(init_weights)
        
        ndcg_best_group = 0.0
        weights_best_group = []
        
        PSO_summary = {}
        
        # Establish swarm
        print('\nInitializing swarm\n')
        swarm = []
        for i in range(0, num_particles):
            swarm.append(Particle(init_weights, num_dimensions))
            
        # Begin optimization loop
        print('\nBegin Particle Swarm Optimization')
        i = 0
        while i < maxiter:
            print('\nIteration = ', i)
            print('\nBest solution so far = ', weights_best_group)
            print('\nNDCG best solution so far = ', ndcg_best_group)
            
            # Add info to PSO summary
            PSO_summary[i] = {}
            PSO_summary[i]['best_solution'] = weights_best_group
            PSO_summary[i]['fitness_best_solution'] = ndcg_best_group
            PSO_summary[i]['swarm'] = swarm
            
            for j in range(0,num_particles):
                swarm[j].evaluate(fitness, data, true_labels)
                
                # Determine if current particle is the best of the group
                if swarm[j].ndcg_i < ndcg_best_group or ndcg_best_group == -1:
                    weights_best_group = list(swarm[j].weights_i)
                    ndcg_best_group = float(swarm[j].ndcg_i)
                
            # Update swarm velocities and weights
            for j in range(0, num_particles):
                swarm[j].update_velocity(weights_best_group, num_dimensions)
                swarm[j].update_weights(bounds, num_dimensions)
            
            i += 1
            
        print('\nPSO finished\n')
        print('BEST SOLUTION WITH WEIGHTS: ',weights_best_group)
        print('\nNDCG BEST SOLUTION = ',ndcg_best_group)
        
        return PSO_summary
        
            