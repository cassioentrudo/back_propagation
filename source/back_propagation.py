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

def error_j(fx, expected, size_dataset):
    y = np.array(expected)
    j_vector = -y * np.log(fx) - (np.ones(len(y)) - y) * np.log(np.ones(len(y)) - fx)
    j = np.sum(j_vector)
    #j /= size_dataset ????
    return j

def inputs_propagation(network, instance, inputs):
    for i in range(len(network.layers_size)-1):
        print("\n\n########################################################################")
        print("layer:", i+1)            
        print("quantidade de neuronios=", network.layers_size[i]) 
                        
        #criando vetor a
        a = np.zeros((int(network.layers_size[i])+1, 1), dtype=np.float64)
        a[0] = 1 #bias

        for j in range(len(inputs)):
            print("input=", round(inputs[j], 5))
            a[j + 1] = round(inputs[j],5)
        print("vetor_a=", a)               
            
        #vetor com pesos
        layer = network.layers[i]
        print("layer=", layer)
        #print("layer_T=", np.transpose(layer))
        z = np.dot(layer, a)
        print("vetor_z=", z)

        new_inputs = sigmoid(z)
        #print("out", new_inputs)
        inputs = new_inputs
            
    return new_inputs
 
def backPropagation(network, fx, y):
    
    delta_y = np.array()
    delta_k = np.array()
    D  = np.array()    
    aux = np.array()    
    for i in range(len(fx)):
        delta_y = fx[i] - y[i]
    
    for k in range(network.layers_size, 1, -1):
        
        aux = np.multiply( np.dot(np.transpose(network.layers[k]), delta_y[k + 1]), network.a[k]) 
        delta_k = np.multiply( aux, (1 - network.a[k]))
        
        delta_y[k] = delta_k[1:]
    
    for j in range(network.layers_size, 0, -1):
        D[j] += np.dot(delta_y[j + 1], np.transpose(network.a[j]))
        

    return delta_y, D
    

def regularization(network, D): 
    
    P  = np.array()
    
    for k in range(network.layers_size, 0, -1):
        layersReg = np.matrix(network.layers[k - 1][:])
        layersReg[:, 0] = np.zeros(len(layersReg[0]))
        P[k] = np.multiply( network.lmbda, layersReg)
        D[k] = (1.0 / len(table)) * (D[k] + P[k])
        
        

    
    
def update_layers(alpha, network , D):
    for k in range(network.layers_size, 0, -1):
        network.layers[k] -= np.multiply(alpha, D[k])
     
     
      

def execute(network, instances):
    for instance in instances:
        inputs_outputs = instance.split(";")        
        
        inputs = []
        for inpts in inputs_outputs[0].split(','):
            print("inpts:", inpts)
            inputs.append(float(inpts))

        outputs = []
        for outpts in inputs_outputs[1].split(','):
            print("outpts:", outpts)
            outputs.append(float(outpts))
        
        fx = inputs_propagation(network, instance, inputs)
        
        print("saida predita=", fx)
        print("saida esperada=", outputs)
        
        error = error_j(fx, outputs, len(instances))
        print("erro J calculado=", error)
        