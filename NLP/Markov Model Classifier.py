# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 19:26:08 2022
Markov Model Classifier
@author: kholm
"""

from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np


f = open(r"Frost.txt", "r",encoding="utf-8")
fil = f.read().splitlines()
x= list(filter(None,[x for x in fil if x != 'â€‰']))


X_trainfrost, X_testfrost = train_test_split( x, test_size=0.2)



f = open(r"Poe.txt", "r",encoding="utf-8")
fil = f.read().splitlines()
x= list(filter(None,[x for x in fil if x != 'â€‰']))



X_trainpoe, X_testpoe = train_test_split( x, test_size=0.2)




#preprocess
def preprocess(lines):
    
    def def_value():
        return 0
    
    
    cnt1 = defaultdict(def_value)
    lookups = defaultdict(def_value)
    clean = []
    ind = 0
    #remove special characters
    for l in lines:
        templ = l.replace(",","").replace("!","").replace(".","").replace("'","").replace('"',"").replace('>',"") \
                    .replace(':',"").replace('?',"").replace('-',"").replace('(',"").replace(')',"").replace(';',"").lower()      
        lst = templ.split()
        
        clean.append(lst)
        for i in lst:
            cnt1[i] += 1
            if cnt1[i] == 1:
                lookups[i] = ind
                ind += 1
    
    
    
    #add unknown space
    cnt1["unk"] = 1
    lookups["unk"] = ind 

    #calc log priors
    tot = np.sum(list(cnt1.values()))
    
    for k,v in cnt1.items():
      
        cnt1[k] /= tot
        cnt1[k] = np.log(cnt1[k])
    
    
    return clean,cnt1,lookups
    

cleanpoetrain,pipoe,lookuppoe = preprocess(X_trainpoe)
cleanfrosttrain,pifrost,lookupfrost = preprocess(X_trainfrost)
priorpoe = len(X_trainpoe)/(len(X_trainpoe) + len(X_trainfrost))
priorfrost  = len(X_trainfrost)/(len(X_trainpoe) + len(X_trainfrost))


def train(data,lookup):
    A = np.ones((len(lookup),len(lookup)))
    
    for ln in data:
        i = 1
        while i < len(ln):
            
            frm = lookup[ln[i-1]]
            to = lookup[ln[i]]
            A[frm,to] += 1
            
         
            i += 1
        
        
    logA = np.log(A/np.sum(A,axis=1))

    return logA
    
    
Amatpoe = train(cleanpoetrain,lookuppoe)
Amatfrost = train(cleanfrosttrain,lookupfrost)


#calculate posterior prob
def calc_posterior(A,pi,prior,seq,lookup):
    
    templ = seq.replace(",","").replace("!","").replace(".","").replace("'","").replace('"',"").replace('>',"") \
                    .replace(':',"").replace('?',"").replace('-',"").replace('(',"").replace(')',"").replace(';',"").lower()      
    ln = templ.split()
    
    p = 0
    

    #add prior prob

    p += pi.get(ln[0],pi[lookup["unk"]]) 
    
    
    i = 1
    #add up probability
    while i < len(ln):
        
        
        frm = lookup.get(ln[i-1],lookup["unk"])
        to = lookup.get(ln[i],lookup["unk"])
    
        p += A[frm,to]
        
        i += 1
        
    
    #add author prior
    p += prior

    return p
    


#calculate training accuracy
correct = 0
for j in X_trainfrost:
  
    if len(j) > 1:
        p1 = calc_posterior(Amatpoe,pipoe,priorpoe,j,lookuppoe)
        p2 = calc_posterior(Amatfrost,pifrost,priorfrost,j,lookupfrost)
        if p1 < p2:
            correct += 1
        
print("TRAINING ACCURACY: ")
print("FROST: {:.2f}".format(correct/len(X_trainfrost)))
        
        
correct = 0
for j in X_trainpoe:
  
    if len(j) > 1:
        p1 = calc_posterior(Amatpoe,pipoe,priorpoe,j,lookuppoe)
        p2 = calc_posterior(Amatfrost,pifrost,priorfrost,j,lookupfrost)
        if p1 > p2:
            correct += 1
                
        
print("POE: {:.2f}".format(correct/len(X_trainpoe)))       


#calculate TESTING accuracy
correct = 0
for j in X_testfrost:
  
    if len(j) > 1:
      
        p1 = calc_posterior(Amatpoe,pipoe,priorpoe,j,lookuppoe)
        p2 = calc_posterior(Amatfrost,pifrost,priorfrost,j,lookupfrost)
        if p1 < p2:
            correct += 1
        
print()
print("TESTING ACCURACY: ")
print("FROST: {:.2f}".format(correct/len(X_testfrost)))
        
        
correct = 0
for j in X_testpoe:
  
    if len(j) > 1:
        p1 = calc_posterior(Amatpoe,pipoe,priorpoe,j,lookuppoe)
        p2 = calc_posterior(Amatfrost,pifrost,priorfrost,j,lookupfrost)
        if p1 > p2:
            correct += 1
                
        
print("POE: {:.2f}".format(correct/len(X_testpoe)))       

