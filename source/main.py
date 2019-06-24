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
alpha=0.4
numAlpha = 3
numFR = 3
numCamadas = 3
numParada = 3
conParada = 0.00015
tamCamadas = 3
initFR=0.250


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
    #print("Resultado: ", answers)
    #print("Corretos: ", correct)
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
        networkClean = Neural_Network(network_lambda, layers_size, layers)
        err = back_propagation.gradient_verification(network, dataset_file_lines, isTest, alpha,networkPlus, networkMinus,networkClean,  0.000001)
        
        
        
    
        #chama algoritmo de bajpropagation passando a rede e as instancias de treinamento
        errorReg, network, fx = back_propagation.execute(neural_network, dataset_file_lines, isTest, alpha)
        
    else:
        errorReg, network, fx = back_propagation.execute(network, dataset_file_lines, isTest, alpha)
    
    return errorReg, network, fx


def save_results(neural_network_structure, vp, vn, fp, fn, usedAlpha, conParada):
    full_string = "Condição de parada difJ < "
    full_string += str(conParada)
    full_string += '\n'
    full_string += "Alpha: "
    full_string += str(usedAlpha)
    full_string += '\n'
    full_string += "Fator de Regularização : "
    full_string += str(neural_network_structure[0])
    full_string += '\n'
    full_string += "Quantidade de entradas : "
    full_string += str(neural_network_structure[1])
    full_string += '\n'
    full_string += "Quantidade de saídas: "
    full_string += str(neural_network_structure[-1])
    full_string += '\n'
    full_string += "Quantidade de camadas ocultas: "
    full_string += str(len(neural_network_structure) - 3)
    full_string += '\n'
    
    
    for i in range(2, len(neural_network_structure)-1, 1):
        full_string += "Camada oculta. Neurônios : "
        full_string += str(neural_network_structure[i])
        full_string += '\n'
    
    full_string += "VP: "
    full_string += str(vp)
    full_string += '\n'
    full_string += "VN: "
    full_string += str(vn)
    full_string += '\n'
    full_string += "FP: "
    full_string += str(fp)
    full_string += '\n'
    full_string += "FN: "
    full_string += str(fn)
    full_string += '\n\n'
    
    
    
    with open("Results.txt","a+") as f:
        f.write(full_string)


def execute_once(neural_network_structure, alpha, folds, condParada):
    vp = 0
    vn = 0
    fp = 0
    fn = 0
    for i in range(len(folds)):
        empty_initial_weights = []
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
            
        
        print("Executing with fold number: ", i, " and neural_network_structure=", neural_network_structure)
        errorReg,network,fx = initialize_network_for_validation(neural_network_structure, empty_initial_weights, fixed_dataset, False)
        negPosError = 0
        lastDifError = 0
        difError=errorReg
        while (abs(difError)>condParada and negPosError < 6):
            difError=errorReg
            network.a=[]
            network.z=[]
            errorReg,network,fx = initialize_network_for_validation(neural_network_structure, empty_initial_weights, fixed_dataset, False, network)
            difError-=errorReg
            if ((lastDifError > 0 and difError < 0) or (lastDifError < 0 and difError > 0)):
                negPosError += 1
            else:
                negPosError = 0
            lastDifError = difError
            #print("Fx: ", fx)
            print("difError: ", difError)
        answers,correct = categoricVotation(network,testFold,table.columns.size -1, test_dataset)
        for j in range(len(answers)):
            if(answers[j]==correct[j]):
                if(answers[j]==0):
                    vn += 1
                else:
                    vp += 1
            else:
                if(answers[j]==0):
                    fn += 1
                else:
                    fp += 1
    return vp, vn, fp, fn

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
    
        #print("empty_initial_weights=", empty_initial_weights)
        for k in range(1,numParada+1):
            for i in range(1,numAlpha+1):
                for j in range(1,numFR+1):
                    for l in range(1,tamCamadas+1):
                        for m in range (1,numCamadas+1):
                        
                            #neural_network_structure = [0.250, table.columns.size -1, 40, 40, 1 ]
                            neural_network_structure = []
                            neural_network_structure.extend([initFR*j])
                            neural_network_structure.extend([table.columns.size -1])
                            for n in range (1,m+1):
                                neural_network_structure.extend([l*table.columns.size -1])
                            neural_network_structure.extend([1])
                            vp, vn, fp, fn = execute_once(neural_network_structure, i*alpha, folds, conParada/k)
                            save_results(neural_network_structure, vp, vn, fp, fn, i*alpha, conParada/k)
            
            

main()