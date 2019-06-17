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
    print("[back_propagation] propagation")
    num_inputs = network.layers[0] #numero de entradas
    print("[back_propagation] Numero de entradas:", num_inputs)
    
#    for weight in networ.weights:
        
    
    
    for instance in instances:
        print("instance:", instance)
        atribbutes = instance.split(";")        
        print("input_and_output:", atribbutes)
        
        
        inputs = []
        inputs.append(atribbutes[0])
        print("inputs:", inputs)
        
        
        outputs = []
        outputs .append(atribbutes[1])
        print("outputs:", outputs)

    
        # para cada camada, calcula a e z
        for layer in network.layers:
            print("layer:", layer)
            
            #criando vetor a
            a = np.zeros((int(layer) + 1, 1), dtype=np.float64)
            a[0] = 1 #bias
          
            
            print(inputs)
            i = 0
            for inp in inputs:
                print("inp",inp)
                a[i + 1] = inp
                i+=1
                
        
        
        
        
#    
#    network.a = np.zeros((int(num_inputs) + 1, 1), dtype=np.float64) #cria vetor a com
#    network.a[0] = 1; #bias
#    
#    for instance in instances:
#        print("instance:", instance)
#        
#        input_output = instance.split(';')
#        inpu = input_output.split(',')[0]

        
        
#        for i in range(len(input_output)):
#            print(network.a[i+1],inputs[i])
#            network.a[i+1] = input_outputß[i]
    


        