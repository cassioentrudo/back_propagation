#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 20:41:09 2019

@author: cassio
"""

import sys
import k_folds
from DadosTreinamento import table
from network import Network

numFolds=10

def main():
    folds = k_folds.k_folding(table, numFolds, table.columns[table.columns.size-1])
    #initialize_network()
  
def initialize_network():
    print("[initialize_network]")
    network_file = sys.argv[1] #arquivo que define a estrutura da rede
    initial_weights_file = sys.argv[2] #arquivo indicado pesos inciais
    #dataset_file = sys.argv[3] #arquivo com conjunto de treinamento
    
    #Leitura do arquivo network_file (estrutura da rede, número de camadas, quantidade de neurônios, etc)
    f = open(network_file, "r")
    network_file_lines = f.read().splitlines()
    #primeira linha é o fator de regularização
    network_lambda = float(network_file_lines[0])
    
    #pega as linhas que correspondems as camadas
    layers = []
    for layer in range(len(network_file_lines)-1): 
        layers.append(network_file_lines[layer + 1] )

    #Leitura do arquivo initial_weights_file (pesos iniciais)
    weights = []
    f = open(initial_weights_file, "r")
    initial_weights_file_lines = []
    initial_weights_file_lines = f.read().splitlines()
    initial_weights_file_lines = initial_weights_file_lines.split(';')

    index = 0
    for w in initial_weights_file_lines:
        print("layer[", index, "] = ", w)
        layers[index] = w
        index += 1

    
    
    print("[Main] Fator de regularizacao:", network_lambda)
    print("[Main] Quantidade de camadas:", len(layers))

    
    
    #estrutura geral da rede
    network = Network(network_lambda, layers, weights)
        

if __name__ == "__main__":
    main()