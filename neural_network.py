#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 20:47:11 2019

@author: cassio
"""

class Neural_Network:
    def __init__(self, lmbda, layers_size, layers):
        #fator de regularização
        self.lmbda = lmbda 
        
        #camadas
        self.layers_size = layers_size
        
        #ativações
        self.a = []
        
        #layer propriamente dito, com neurônios e pesos
        self.layers = layers
        
        #vetor intermediario z
        self.z = []
        
    def PrintNetwork(self):
        #print("*******NODO*******")
        #print("lmbda =", self.lmbda)
        #print("layers_size =", self.layers_size)
        #print("a =", self.a)
        #print("z =", self.z)
        #print("********************")
        return 0