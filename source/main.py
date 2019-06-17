#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 20:41:09 2019

@author: cassio
"""

import sys
import back_propagation

from network import Network
  
def initialize_network():
    print("[main] Inicializando rede")
    network_file = sys.argv[1] #arquivo que define a estrutura da rede
    initial_weights_file = sys.argv[2] #arquivo indicado pesos inciais
    dataset_file = sys.argv[3] #arquivo com conjunto de treinamento
    
    #Leitura do arquivo network_file (estrutura da rede, número de camadas, quantidade de neurônios, etc)
    f = open(network_file, "r")
    network_file_lines = f.read().splitlines()
    #primeira linha é o fator de regularização
    network_lambda = float(network_file_lines[0])
    
    #cria cada linha sendo uma camada e o valor da linha sendo a quantidade de neurônios
    layers = []
    for neurons in network_file_lines[1:]: 
        print("[main] camada com", neurons, "neuronio")
        layers.append(neurons)

    #Leitura do arquivo initial_weights_file (pesos iniciais)
    weights = []
    f = open(initial_weights_file, "r")
    initial_weights_file_lines = f.read().splitlines()

    for weight in initial_weights_file_lines:
        print("[main] peso", weight)
        weights.append(weight)
        
    instances = []
    f = open(dataset_file, "r")
    dataset_file_lines = f.read().splitlines()
    for instance in dataset_file_lines:
        print("[main] instance:", instance)
        instances.append(instance)
    
    
    print("[main] Fator de regularizacao:", network_lambda)
    print("[main] Quantidade de camadas:", len(layers))
    
    #estrutura geral da rede
    network = Network(network_lambda, layers, weights)
    
    #chama algoritmo de bajpropagation passando a rede e as instancias de treinamento
    back_propagation.propagation(network, instances)
    
        
initialize_network()