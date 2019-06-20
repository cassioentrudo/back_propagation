#!/usr/bin/env python2
# encoding: utf-8
"""
Created on Tue Jun 11 20:41:09 2019c

@author: cassio
"""
import sys
import random
import numpy as np
import k_folds
import back_propagation

from DadosTreinamento import table
from DadosTreinamento import neural_network_structure
from neural_network import Neural_Network


numFolds=10

#def main():
#    folds = k_folds.k_folding(table, numFolds, table.columns[table.columns.size-1])
#    initialize_network()
  
def initialize_network_for_validation(network_file_lines, initial_weights_file_lines, dataset_file_lines, isTest):
    #print("[main] Inicializando rede")
    
    print("network_file_lines", network_file_lines)
    print("initial_weights_file_lines", initial_weights_file_lines)
    #print("dataset_file_lines", dataset_file_lines)

    #primeira linha é o fator de regularização
    network_lambda = float(network_file_lines[0])   
    #ada linha sendo uma camada e o valor da linha sendo a quantidade de neurônios
    layers_size = []
    for neurons in network_file_lines[1:]: 
        print("[main] camada com", neurons, "neuronio")
        layers_size.append(int(neurons))


    layers = [] # camadas
    
    # faz a leitura dos pesos no arquivo de pesos iniciais passados por linha de comando
    if(len(initial_weights_file_lines) > 0):
        print("initial weghts vector is not null")
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
    else:
        #cria pesos inicias randomicamente entre -1 e 1
        print("initial weights vector is null")
        print("layer_sizes", layers_size)
        
        for i, layer in enumerate(layers_size[:-1]):
            v_neurons = []
            for i in range(layers_size[i+1]):
                weights_v = []
                for y in range(layer + 1): #bias
                    weights_v.append(random.triangular(-1, 1, 0))
                v_neurons.append(weights_v)
            layers.append(np.array(v_neurons))
          
    instances = []
    for instance in dataset_file_lines:
        instances.append(instance)
    
    print("[main] Fator de regularizacao:", network_lambda)
    print("[main] Quantidade de camadas:", len(layers))
    
    #estrutura geral da rede
    neural_network = Neural_Network(network_lambda, layers_size, layers)
    
    #chama algoritmo de bajpropagation passando a rede e as instancias de treinamento
    back_propagation.execute(neural_network, dataset_file_lines, isTest)
    

attributes_command_line = sys.argv

#carrega a estrutura da rede, pesos e data set dos argumentos passados
if(len(attributes_command_line) > 1):
    network_file = sys.argv[1] #arquivo que define a estrutura da rede
    initial_weights_file = sys.argv[2] #arquivo indicado pesos inciais
    dataset_file = sys.argv[3] #arquivo com conjunto de treinamento
    
    f = open(network_file, "r")
    network_file_lines = f.readlines()
    
    f = open(initial_weights_file, "r")
    initial_weights_file_lines = f.readlines()
        
    f = open(dataset_file, "r")
    dataset_file_lines = f.readlines()
    
    initialize_network_for_validation(network_file_lines, initial_weights_file_lines, dataset_file_lines, True)
else:
    empty_initial_weights = []
    
    print("neural_network_structure=", neural_network_structure)
    print("empty_initial_weights=", empty_initial_weights)
    
    
    dataset = table.values.tolist()
    fixed_dataset = []
    
    #print("dataset", dataset)
    for i in range(len(dataset)):
        fixed_dataset.append(str(dataset[i])[2:-2])
    
    initialize_network_for_validation(neural_network_structure, empty_initial_weights, fixed_dataset, False)
