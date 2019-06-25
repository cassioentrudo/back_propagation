#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 10:20:30 2019

@author: cassio
"""
import copy
import numpy as np
import re

D = []

def sigmoid(z):
    sigmoide = 1.0 / (1.0 + np.exp(-z))
    return sigmoide

def error_j(fx, expected, size_dataset):
    #print ("fx: ", fx, "expected: ", expected, "size_dataset: ", size_dataset)
    y = np.array(expected, dtype=np.float64)
    #print("YYY", y)
    j_vector = -y * np.log(fx) - (np.ones(y.size) - y) * np.log(np.ones(y.size) - fx)
     #np.dot(-y,  np.log(fx)) - np.dot((np.ones(y.size) - y) , np.log(np.ones(y.size) - fx))
    j = np.sum(j_vector)
    #j /= size_dataset
    return j

def inputs_propagation(network, instance, inputs):
    auxa = []
    auxz = []
    #criando vetor a
    a = np.zeros((int(network.layers_size[0])+1, 1)).ravel()
    a[0] = 1 #bias

        
    for j in range(len(inputs)):

        a[j + 1] = np.round(inputs[j],5)
    auxa.append(a)
    for i in range(len(network.layers_size)-1):
        layer = network.layers[i] #.astype
        z = np.dot(layer, a)
        auxz.append(z)

        a = sigmoid(z)
        
        if(i!=len(network.layers_size)-2):
            a = np.insert(a, 0, 1)       
        
        auxa.append(a)

        inputs = a
    network.a.append(auxa)
    network.z.append(auxz)
    
    return inputs
 
def backPropagation(network, fx, y, inst):
    #print("len(network.layers)-1:=", len(network.layers)-1)
    delta_y = []
    container = []
    delta_k = []
    D  = []
    aux = []
    for i in range(len(fx)): 
        if (isinstance(y,list)):
            container.extend([np.around(fx[i] - y[i],5)])
        else:
            container.extend([np.around(fx[i] - y,5)])
    #print("container: ", container)
    delta_y.append(container)
    
    #delta_y = np.reshape(delta_y,len(delta_y[-1]),1)
    for k in range(len(network.layers_size)-1, 1, -1):
        delt = delta_y[-1]
       # print("delt: ", delt)
        transposta = np.transpose(network.layers[k-1])
       # print("transposta: ", transposta)
        aux = np.dot(transposta, delt)
        aux = np.multiply( aux, network.a[inst][k-1]) 
        aux = np.multiply( aux, (1 - network.a[inst][k-1]))
       # print("aux: ", aux)
        for j in range(1,len(aux)):
            auxj=aux[j]
            delta_k.extend([np.around(auxj,5)])
        aux = []  #limpa
        delta_y.append(delta_k)
        delta_k = []        #limpa
    
    for j in range(len(network.layers_size)-1, 0, -1):
        #np.insert(D,j, D[j] + (delta_y[(len(network.layers_size)-1)-j]*np.transpose(network.a[j])))
        transposta = np.transpose(network.a[inst][j-1])
        delta = np.array(delta_y[len(network.layers_size)-1-j])
        delta = np.reshape(delta,(len(delta),1))
        if(transposta.ndim>1):
            if(transposta.shape[1]>transposta.shape[0]):
                transposta = np.reshape(transposta,(1,transposta.shape[1]))
            else:
                transposta = np.reshape(transposta,(1,transposta.shape[0]))
        else:
            transposta = np.reshape(transposta,(1,len(transposta)))
        #print("D= ", delta_y[(len(network.layers_size)-1)-j], "\n transposta da a: ", transposta)
        D.append(np.dot(delta, transposta))
        
    return delta_y, D
    

def regularization(network, D, instances): 
    
    P  = []
    
    for k in range(len(network.layers_size)-1, 0, -1):
        layersReg = np.matrix(network.layers[k - 1][:])
        layersReg[:, 0] = np.zeros(len(layersReg[0]))
        layersReg=network.lmbda*layersReg
        P.append(layersReg)
    
    
    norm = []
    for k in range(len(P)):
        norm.extend([0])
        
    for inst in range(len(instances)):
        for k in range(len(P)):
            norm[len(P)-1-k] = D[inst][k] + norm[len(P)-1-k]
            
    for k in range(len(P)):
        #aux = ([(1.0 / len(instances)) * (norm[len(P)-1-k] + P[k])])
        aux = np.ravel([(1.0 / len(instances)) * (norm[len(P)-1-k] + P[k])])
        norm[len(P)-1-k] =  aux
            
            #if (norm[len(P)-1-k]==0):
            #    aux = ([(1.0 / len(instances)) * (D[inst][k] + P[k] + norm[len(P)-1-k])])
            #else:
            #    aux = ([(1.0 / len(instances)) * (D[inst][k] + P[k] + norm[len(P)-1-k][0])])
            #norm[len(P)-1-k] =  aux
        
    return norm

    
    
def update_layers(alpha, network , D):
    for k in range(len(network.layers_size)-1, 0, -1):
        network.layers[k-1] = network.layers[k-1] - alpha*D[k-1][0]
     
     
def calculateS(network):
    S=0
    for layer in network.layers:
        for i in range(len(layer)):
            for k in range(len(layer[i])):
                aux = layer[i,k]
                S += aux ** 2
    return S



def gradient_verification(network, instances, isTest, alpha, networkPlus, networkMinus, networkClean,  E):
    error, networkClean, fx, D = execute(networkClean, instances, isTest, alpha)
    networkPlus = copy.deepcopy(networkClean) 
    networkMinus = copy.deepcopy(networkClean)
    
    full_string = ""
    for l in range(len(D)):
        for c in range(len(D[l])):
            g = round(D[l][c], 5)
            full_string += str(g)
            full_string += '\t'
        full_string += '\n'
    
    full_string += '\n'
            
    print(full_string)
    
    for j in range(len(networkPlus.layers)):

        gradError = 0


        for i in range(len(networkPlus.layers[j])):
            for k in range(len(networkPlus.layers[j][i])):
                networkPlus.layers[j][i][k] += E
                errorPlus, networkPlus, fxPlus, Derror1 = execute(networkPlus, instances, isTest, alpha)
                networkMinus.layers[j][i][k] -= E

                errorMinus, networkMinus, fxMinus, Derror2  = execute(networkMinus, instances, isTest, alpha)
                gradError += (errorPlus - errorMinus) / (2 * E)

        print("Theta: ", j," erro", gradError)
        full_string += "Theta: "
        full_string += str(j)
        full_string += " Erro: "
        full_string += str(gradError)
        full_string += '\n'

        
    f = open("verificacao_numerica_gradiente.txt","w+")
    f.write(full_string)

def execute(network, instances, isTest, alpha):
    D=[]
    inst = -1
    error=0
    
    for instance in instances:
        inputs = []
        outputs = []
        
        inst = inst +1
        
        #print("isTest=", isTest)
                       
        if(isTest):
            inputs_outputs = instance.split(";")    
            
            for inpts in inputs_outputs[0].split(','):
                #print("inpts:", inpts)
                inputs.append(float(inpts))

        
            for outpts in inputs_outputs[1].split(','):
                #print("outpts:", outpts)
                outputs.append(float(outpts))
        else:
            inputs_outputs = instance.split(",")   
            for i in range(len(inputs_outputs) - 1):
                #print(i, "inpts:", inputs_outputs[i])
                inputs.append(float(inputs_outputs[i]))

        
            outputs = float(inputs_outputs[-1])
            #print("outpts:", outputs)
        
        #print("Chamando Inputs_Propagation")
        fx = np.round(inputs_propagation(network, instance, inputs),5)
        
        #print("saida predita=", fx)
        #print("saida esperada=", outputs)
        
        error += error_j(fx, outputs, len(instances))
        
        #network.PrintNetwork()
        delta_y, aux = backPropagation(network, fx, outputs, inst)
        D.append(aux)
        
    D = regularization(network,D,instances)
    
    J = error/len(instances)
    S = calculateS(network)
    S *= network.lmbda/(len(instances)*2)
    errorReg = J+S
    #print("Erro regularizado: ", errorReg)
    
    if(len(instances)>=1):
        update_layers(alpha, network, D)
            
    return errorReg, network, fx, D