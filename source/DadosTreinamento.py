# -- coding: utf-8 --
import pandas as pd


isNumeric = True
#tablePath = "..\data\wine.data"
#tablePath = "..\data\ionosphere.data"


tablePath = "../data/dataset_1.txt"
#tablePath = "..\data\dataset_2.data"

#vetor que contem a estrutura da rede (camadas e neuronios por camada)
neural_network_structure = [0.000, 1, 2, 1 ]

def DataRead(str1):
    if(isNumeric==True):    
        dataTable = pd.read_csv("%s" % str1,header=None, sep="/s*/,",  engine='python') #PARA ATRIBUTOS NUMÉRICOS
    else:
        dataTable = pd.read_csv("%s" % str1, sep="\s*\;",  engine='python')
    return dataTable

def normalizeTable(table, minNumber, maxNumber):
    for i in table.columns:
        if (i != table.columns[table.columns.size-1]):
            maxVal = table[i].max()
            minVal = table[i].min()
            if (maxVal != minVal):
                table[i] = minNumber + (table[i]-minVal)*(maxNumber-minNumber)/(maxVal-minVal)
            else:
                table[i]=0
    return table

table = DataRead(tablePath)

if (isNumeric == True):
    names = {}
    for x in range(len(table.columns)):
        number = x+65
        if (number>90):
            number = number+6
        names[x]=chr(number)
    table = table.rename(columns=names)

if(tablePath == "wdbc.data"):
    table = table.drop(columns="A")
    aux = table["B"]
    table = table.drop(columns="B")
    table["result"]=aux

if (tablePath == "wine.data"):
    aux = table["A"]
    table = table.drop(columns="A")
    for x in range(len(aux)):
       aux.iloc[x]=names.get(aux.iloc[x])
    table["result"]=aux
    
normalizeTable(table,-1,1)

    
    