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
import pandas as pd

from data_training import table
from neural_network import Neural_Network


numFolds=10
alpha=0.2


def categoricVotation(network, testFold, targetFeature, testDataset):
    answers = []
    correct = []
    possibleAnswers = []
    for i in range(len(testFold)):
        if (not(testFold[testFold.columns.values[targetFeature]][i] in possibleAnswers)):
            possibleAnswers.extend([testFold[testFold.columns.values[targetFeature]][i]])
        correct.extend([testFold[testFold.columns.values[targetFeature]][i]])
    #print("[VOTATION] testFold", testFold)
   # print("[VOTATION] len(testFold)=", len(testFold))
    for i in range(len(testFold)):
        #print("[VOTATION] tree=", tree)
        errorReg, network, fx = initialize_network_for_validation([0], [0], [testDataset[i]], False, network)
        minDif=100
        minAnswer=100
        for j in range(len(possibleAnswers)):
            if (abs(fx-possibleAnswers[j])<minDif):
                minDif=abs(fx-possibleAnswers[j])
                minAnswer=possibleAnswers[j]
        answers.extend([minAnswer])
    print("Resultado: ", answers)
    print("Corretos: ", correct)
    return answers,correct

def initialize_network_for_validation(network_file_lines, initial_weights_file_lines, dataset_file_lines, isTest, network = None):
    #print("[main] Inicializando rede")
    
    #print("network_file_lines", network_file_lines)
    #print("initial_weights_file_lines", initial_weights_file_lines)
    #print("dataset_file_lines", dataset_file_lines)
    if (network == None):
        #primeira linha é o fator de regularização
        network_lambda = float(network_file_lines[0])
        #ada linha sendo uma camada e o valor da linha sendo a quantidade de neurônios
        layers_size = []
        for neurons in network_file_lines[1:]: 
            #print("[main] camada com", neurons, "neuronio")
            layers_size.append(int(neurons))
    
    
        layers = [] # camadas
        
        # faz a leitura dos pesos no arquivo de pesos iniciais passados por linha de comando
        if(len(initial_weights_file_lines) > 0):
            #print("initial weghts vector is not null")
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
            #print("initial weights vector is null")
            #print("layer_sizes", layers_size)
            
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
        
        #print("[main] Fator de regularizacao:", network_lambda)
        #print("[main] Quantidade de camadas:", len(layers))
        
        #estrutura geral da rede
        neural_network = Neural_Network(network_lambda, layers_size, layers)
        
        networkPlus = Neural_Network(network_lambda, layers_size, layers)
        networkMinus = Neural_Network(network_lambda, layers_size, layers)
       # err = back_propagation.gradient_verification(network, dataset_file_lines, False, alpha,networkPlus, networkMinus,  0.000001)
        
        
        
    
        #chama algoritmo de bajpropagation passando a rede e as instancias de treinamento
        errorReg, network, fx = back_propagation.execute(neural_network, dataset_file_lines, isTest, alpha)
        
    else:
        errorReg, network, fx = back_propagation.execute(network, dataset_file_lines, isTest, alpha)
    
    return errorReg, network, fx


def save_results(neural_network_structure, total_rights, total_wrongs):
    full_string = "Quantidade de Folds: "
    full_string += str(numFolds)
    full_string += '\n\n'
    full_string += "Alpha: "
    full_string += str(alpha)
    full_string += '\n\n'
    full_string += "Fator de Regularização : "
    full_string += str(neural_network_structure[0])
    full_string += '\n\n'
    full_string += "Quantidade de entradas : "
    full_string += str(neural_network_structure[1])
    full_string += '\n\n'
    full_string += "Quantidade de saídas: "
    full_string += str(neural_network_structure[-1])
    full_string += '\n\n'
    full_string += "Quantidade de camadas ocultas: "
    full_string += str(len(neural_network_structure) - 3)
    full_string += '\n\n'
    
    
    for i in range(2, len(neural_network_structure)-1, 1):
        full_string += "Camada oculta. Neurônios : "
        full_string += str(neural_network_structure[i])
        full_string += '\n'
    
    full_string += '\n'
    full_string += "Acertos: "
    full_string += str(total_rights)
    full_string += '\n\n'
    full_string += "Erros: "
    full_string += str(total_wrongs)
    
    f= open("Results.txt","w+")
    f.write(full_string)


def main():
    folds = k_folds.k_folding(table, numFolds, table.columns[table.columns.size-1])
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
    
        #print("empty_initial_weights=", empty_initial_weights)
        
        total_rights = 0
        total_wrongs = 0
        
        for i in range(len(folds)):
            testFold=folds[i]
            numVector = list(range(len(testFold)))
            testFold.index=numVector
            datasets=pd.DataFrame();
            for j in range(len(folds)):
                if (j!=i):
                    datasets = datasets.append(folds[j])
            dataset = datasets.values.tolist()
            testset = testFold.values.tolist()
            fixed_dataset = []
            test_dataset = []
    
            #print("dataset", dataset)
            for k in range(len(dataset)):
                fixed_dataset.append(str(dataset[k])[1:-1])
            
            for k in range(len(testFold)):
                test_dataset.append(str(testset[k])[1:-1])
                
            neural_network_structure = [0.250, table.columns.size -1, 40, 40, 1 ]
            print("Executing with fold number: ", i, " and neural_network_structure=", neural_network_structure)
            errorReg,network,fx = initialize_network_for_validation(neural_network_structure, empty_initial_weights, fixed_dataset, False)
            difError=errorReg
            while (abs(difError)>0.0001):
                difError=errorReg
                network.a=[]
                network.z=[]
                errorReg,network,fx = initialize_network_for_validation(neural_network_structure, empty_initial_weights, fixed_dataset, False, network)
                difError-=errorReg
                #print("Fx: ", fx)
                print("difError: ", difError)
            answers,correct = categoricVotation(network,testFold,table.columns.size -1, test_dataset)
            right = 0
            wrong = 0
            for j in range(len(answers)):
                if(answers[j]==correct[j]):
                    right += 1
                else:
                    wrong += 1
            print("Acertos: ", right, "Erros: ", wrong )
            total_rights += right
            total_wrongs += wrong
            
        save_results(neural_network_structure, total_rights, total_wrongs)
            
            

main()