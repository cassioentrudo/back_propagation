# -*- coding: utf-8 -*- 
import pandas as pd


#%%

def k_folding(kTable, kN, kTarget):
    
    totalSize = len(kTable)
    foldSize = totalSize//kN
    folds = []
    count=0
    if (totalSize%kN>0):
        foldSize = foldSize + 1
    orderedTable = kTable.sort_values(kTarget)
    stratified = pd.DataFrame();
    for x in range(kN):
        for y in range (0, totalSize, kN):
            if (y+x<totalSize):
                #print(y+x)
                pdobject = pd.DataFrame(orderedTable.iloc[y+x])
                pdobject = pdobject.transpose()
                stratified = stratified.append(pdobject)
    x=0
    while(x<totalSize):
        count = count + 1
        if (count<=totalSize%kN):
            folds.append(stratified[x:x+foldSize])
            x = x + foldSize
        else:
            folds.append(stratified[x:x+foldSize])
            x = x + foldSize
    return folds

#%%

#n=4
#target= table.columns[table.columns.size-1]
#fold = k_folding(table, n, target)