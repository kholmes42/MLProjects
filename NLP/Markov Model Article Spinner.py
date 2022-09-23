# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 20:09:09 2022
Markov Model Article Spinner
@author: kholm
"""

import pandas as pd
import numpy as np


from collections import defaultdict
import nltk as nl

from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok


#train markov model (only 2nd order required)
def train(data):
      
    def def_value():
        return 0
    
    def def_value2():
        return defaultdict(def_value)
    

    A2 = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))


    #create A2 for 2nd order
    for ln in data:
      
       if len(ln) > 1:
           wrd = 1
           while wrd < len(ln)-1:
               
               A2[ln[wrd-1]][ln[wrd+1]][ln[wrd]] += 1
               wrd += 1

    #normalize prob
    for w1 in A2.keys():
        for w2 in A2[w1].keys():
        
            tot = sum(A2[w1][w2].values())
            for w3 in A2[w1][w2].keys():
             
                A2[w1][w2][w3] /= tot

  
    return A2


#this function replaces random words in each sentence according to a trained Markov Model (assumes corpus matches)
def spin_article(txt,prob,wrdpersent = 1):
    spun = ""
    sentences = nl.tokenize.sent_tokenize(txt)
    detokenizer = Detok()
    for s in sentences:
        new = nl.tokenize.word_tokenize(s)
        if len(new)-3 > wrdpersent and len(new) > 3:
            indtoreplace = np.random.randint(1,len(new)-3,wrdpersent)
            #replace words
            for i in indtoreplace:
         
                prevwrd = new[i-1].lower()
                nextwrd = new[i+1].lower()

                lookupnewword = np.random.choice(list(prob[prevwrd][nextwrd].keys()),p=list(prob[prevwrd][nextwrd].values()))
                
                new[i] = lookupnewword
              
        newsentence = detokenizer.detokenize(new)
        spun += newsentence
    return spun
    
    
    


df = pd.read_csv(r"bbc_text_cls.csv")
bizarticles = df[df["labels"] == "business"]
allwords = []
for i in range(0,bizarticles.shape[0]):
    #token into words
    words = nl.tokenize.word_tokenize(bizarticles["text"][i].lower())
    #arrange into usable corpus
    allwords.append(words)
    
    
#train Markov model
a2 = train(allwords)



#spin the articles
bizarticles["spun"] = bizarticles["text"].apply(spin_article,prob=a2)
