#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 20:47:11 2019

@author: cassio
"""

class Network:
    def __init__(self, lmbda, layers, weights):
        #fator de regularização
        self.lmbda = lmbda 
        
        #camadas
        self.layers = layers
        
        #ativações
        self.a = []
        
        #pesos
        self.weights = []
        
        #vetor intermediario z
        self.z = []
