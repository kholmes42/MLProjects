
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 19:26:08 2022
Markov Model GENERATOR using 2nd order transitions with python dictionaries
@author: kholm
"""


from collections import defaultdict
import numpy as np







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
                    .replace(':',"").replace('?',"").replace('-',"").replace('(',"").replace(')',"").replace(';',"").replace('_',"").lower()      
        lst = templ.split()
        
        clean.append(lst)
       
        cnt1[lst[0]] += 1
        if cnt1[lst[0]] == 1:
            lookups[lst[0]] = ind
            ind += 1
    
    
    
    #add unknown space
    cnt1["<END>"] = 1
    lookups["<END>"] = ind 

    #calc log priors
    tot = np.sum(list(cnt1.values()))
    
    for k,v in cnt1.items():
      
        cnt1[k] /= tot
        cnt1[k] = cnt1[k]
    
    
    return clean,cnt1,lookups
    





def train(data,lookup):
      
    def def_value():
        return 0
    
    def def_value2():
        return defaultdict(def_value)
    

    A1 = {key: defaultdict(def_value) for key in lookup.keys()}
    A2 = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))

    #create A1 for first order    
    for ln in data:
        
        if len(ln) > 1:
            A1[ln[0]][ln[1]] += 1
        
    #normalize prob
    for w1 in A1.keys():
        tot = sum(A1[w1].values())
        for w2 in A1[w1].keys():
          
            A1[w1][w2] /= tot
            

    #create A2 for 2nd order
    for ln in data:
       
       if len(ln) > 1:
           wrd = 2
           while wrd < len(ln):
           
               
               A2[ln[wrd-2]][ln[wrd-1]][ln[wrd]] += 1
               wrd += 1
           #add end stop
           A2[ln[wrd-2]][ln[wrd-1]]["<END>"] += 1 


    #normalize prob
    for w1 in A2.keys():
        for w2 in A2[w1].keys():
        
            tot = sum(A2[w1][w2].values())
            for w3 in A2[w1][w2].keys():
             
                A2[w1][w2][w3] /= tot

  
    return A1,A2

    



def generate(priors,A1,A2,lookup,lines=4):
    
    poem = ""
    
    for i in range(0,lines):
        
        wrd = np.random.choice(list(priors.keys()),p=list(priors.values()))
      
        cnt = 1
        line = []
        line.append(wrd)
        while wrd != "<END>":
  
      
            #pull from A1
            if cnt == 1:
                
                wrd = np.random.choice(list(A1[wrd].keys()),p=list(A1[wrd].values()))
                line.append(wrd)
            #pull from A2    
            else:
                
                prevprevwrd = line[cnt-2]
                prevwrd = line[cnt-1]
            
                wrd = np.random.choice(list(A2[prevprevwrd][prevwrd].keys()),p=list(A2[prevprevwrd][prevwrd].values()))
                line.append(wrd)
                
            cnt += 1
           
        poem += "\r\n" + " ".join(line)[:-6] + "."
    
    
    return poem

#read data
f = open(r"Frost.txt", "r",encoding="utf-8")
fil = f.read().splitlines()
x= list(filter(None,[x for x in fil if x != 'â€‰']))

#clean data
clean,priors,looks = preprocess(x)

#train and generate data
a1,a2 = train(clean,looks)
pom = generate(priors,a1,a2,looks)
print("A Frost poem by a computer:")
print("----------------------------")
print(pom)
