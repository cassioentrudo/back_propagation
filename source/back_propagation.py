#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 10:20:30 2019

@author: cassio
"""

import numpy as np


def sigmoid(z):
    sigmoide = 1.0 / (1.0 + np.exp(-z))
    return sigmoide


def propagation(network, instances):
    print("\n\n############################# [back_propagation] propagation #############################")
    

    for instance in instances:
        print("instance:", instance)
        atribbutes = instance.split(";")        
        
        inputs = []
        inputs.append(atribbutes[0])
        
        outputs = []
        outputs .append(atribbutes[1])
        
    
        # para cada camada, calcula a e z
        for i in range(len(network.layers)-1):
            print("\n\n########################################################################")
            print("layer:", network.layers[i])            
            print("[back_propagation] Numero de entradas:", network.layers[i+1]) 
            print("[back_propagation] Numero de camadas:", network.layers[i]) 
            
            
            #criando vetor a
            v_a = np.zeros((int(network.layers[i])+1, 1), dtype=np.float64)
            v_a[0] = 1 #bias
            _i = 0
            for inp in inputs:
                print("inp",inp)
                v_a[_i + 1] = inp
                _i+=1
                
            print("v_a=", v_a)
                
            
            #vetor com pesos
            v_weights = np.zeros((int(network.layers[i+1]), int(network.layers[i]) + 1), dtype=np.float64)
            print("v_weights", v_weights)
#           
            print("network.weights", network.weights)
            
            weights = []
            for line_weights in network.weights.split(';'):
                print("line_weights", line_weights)
                for n in line_weights.split(','):
                    #weights.append(n))
                    print("n",n)
                        
            
            for l in range(int(network.layers[i+1])):
                for j in range(int(network.layers[i]) + 1):
                    print("weights[l])=", weights[l])
                    v_weights[i][j] = float(weights[l])
#                
#                
#            v_weights = str(network.weights).split(';')
#            print("new_v_weights", v_weights)
                
        
            #vetor intermediário z
            #v_z = 
        