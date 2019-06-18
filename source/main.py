#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 20:41:09 2019c

@author: cassio
"""

import sys
import k_folds
from DadosTreinamento import table
import back_propagation
from network import Network
import numpy as np

numFolds=10

#def main():
#    folds = k_folds.k_folding(table, numFolds, table.columns[table.columns.size-1])
#    initialize_network()
  
def initialize_network():
    print("[main] Inicializando rede")
    network_file = sys.argv[1] #arquivo que define a estrutura da rede
    initial_weights_file = sys.argv[2] #arquivo indicado pesos inciais
    dataset_file = sys.argv[3] #arquivo com conjunto de treinamento
    
    #Leitura do arquivo network_file (estrutura da rede, número de camadas, quantidade de neurônios, etc)
    f = open(network_file, "r")
    network_file_lines = f.readlines()
    #primeira linha é o fator de regularização
    network_lambda = float(network_file_lines[0])
    
    #ada linha sendo uma camada e o valor da linha sendo a quantidade de neurônios
    layers_size = []
    for neurons in network_file_lines[1:]: 
        print("[main] camada com", neurons, "neuronio")
        layers_size.append(int(neurons))

    #Leitura do arquivo initial_weights_file (pesos iniciais)
    f = open(initial_weights_file, "r")
    initial_weights_file_lines = f.readlines()

    layers = [] # camadas
    for line in initial_weights_file_lines:
        neurons = line.split(';')
        v_neurons = []
        for neuron in neurons:
            weights = neuron.split(',')
            v_weights = []
            for weight in weights: #pesos de cada neurônio
                v_weights.append(float(weight))
            v_neurons.append(v_weights)
        layers.append(np.array(v_neurons)) #cada camada tem seus neurônios que contém seus pesos
        
        
    instances = []
    f = open(dataset_file, "r")
    dataset_file_lines = f.readlines()
    for instance in dataset_file_lines:
        print("[main] instance:", instance)
        instances.append(instance)
    
    
    print("[main] Fator de regularizacao:", network_lambda)
    print("[main] Quantidade de camadas:", len(layers))
    
    #estrutura geral da rede
    network = Network(network_lambda, layers_size, layers)
    
    #chama algoritmo de bajpropagation passando a rede e as instancias de treinamento
    back_propagation.execute(network, instances)
    
        
#if __name__ == "__main__":
#    main()
    
initialize_network()