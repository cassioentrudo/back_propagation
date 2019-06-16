#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 20:47:11 2019

@author: cassio
"""

class Network:
    def __init__(self, lmbda, layers, weights):
        self.lmbda = lmbda
        self.layers = layers
        self.weights = weights
        self.a = []
        self.f = []
        

#class Neuron:
#    def __init__(self):
#        self.weights = 0
#
#class Layer:
#    def __init__(self, num_neurons):
#        print("[Layer] Criando Layer com", num_neurons, "neuronios")
#        self.neurons = []
#        
#        for i in range(int(num_neurons)):
#            print("[Layer] Criando neuronio")
#            neuron = Neuron()
#            self.AddNeuron(neuron)
#     
#    def AddNeuron(self, neuron):
#        self.neurons.append(neuron)
#        
#    def GetTotalNeurons(self):
#        total_neurons =  self.neurons.count()
#        return total_neurons
