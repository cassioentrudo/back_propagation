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

def error_j(fx, expected):
    y = np.array(expected)
    j_vector = -y * np.log(fx) - (np.ones(len(y)) - y) * np.log(np.ones(len(y)) - fx)
    j = np.sum(j_vector)
    return j

def propagation(network, instance, inputs):
    for i in range(len(network.layers_size)-1):
        print("\n\n########################################################################")
        print("layer:", i+1)            
        print("quantidade de neuronios=", network.layers_size[i]) 
                        
        #criando vetor a
        a = np.zeros((int(network.layers_size[i])+1, 1), dtype=np.float64)
        a[0] = 1 #bias

        for j in range(len(inputs)):
            print("input=",inputs[j])
            a[j + 1] = inputs[j]
        print("vetor_a=", a)               
            
        #vetor com pesos
        layer = network.layers[i]
        #print("layer=", layer)
        #print("layer_T=", np.transpose(layer))
        z = np.dot(layer, a)
        print("vetor_z=", z)

        new_inputs = sigmoid(z)
        #print("out", new_inputs)
        inputs = new_inputs
            
    return new_inputs
        

def execute(network, instances):
    for instance in instances:
        print("instance:", instance)
        inputs_outputs = instance.split(";")        
        
        inputs = []
        for inpts in inputs_outputs[0].split(','):
            inputs.append(float(inpts))

        outputs = []
        for outpts in inputs_outputs[1].split(','):
            outputs.append(float(outpts))
        
        fx = propagation(network, instance, inputs)
        
        print("saida predita=", fx)
        print("saida esperada=", outputs)
        
        error = error_j(fx, outputs)
        print("erro J calculado=", error)
        