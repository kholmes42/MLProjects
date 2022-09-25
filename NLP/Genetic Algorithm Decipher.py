# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 15:44:01 2022
Genetic Algorithm Dencryption Cipher
@author: kholm
"""


import numpy as np
import re
import string
import random
import seaborn as sns
from collections import defaultdict
import nltk as nl




#read data
f = open(r"mobydick.txt", "r",encoding="utf-8")
fil = f.read()


trainingtext = nl.tokenize.sent_tokenize(fil.lower())



#go through corpus and calculate letter Markov Probs
def train_letter_prob(txt):
    
    def def_value():
        return 1
    
    
    alphabet = list(string.ascii_lowercase)

    priors = dict(zip(alphabet, [1]*26))
    A1 = defaultdict(lambda: defaultdict(def_value))
                     
    #loop through each sentence
    for sent in txt:
        
        #loop through each word
        words = nl.tokenize.word_tokenize(sent)
        for w in words:
            
            #clean word
            fixedwrd = re.sub(r'[^a-zA-Z]','',w)
            
            #loop through each word
            if len(fixedwrd) >= 1:
                priors[fixedwrd[0]] += 1
                
                cnt = 1
                for l in fixedwrd[1:]:
                    currlett = fixedwrd[cnt]
                    A1[fixedwrd[cnt-1]][currlett] += 1
                    
                    
                    cnt += 1
                
                
    #normalize to log odds
    tot = sum(list(priors.values())) +26
    for k in priors.keys():

        priors[k] = np.log(priors[k]/tot)
    
    
    for k in A1.keys():
        tot = sum(list(A1[k].values()))
        for k1 in A1[k].keys():

            A1[k][k1] = np.log(A1[k][k1]/tot)
    
    

    return priors,A1

p,a1 = train_letter_prob(trainingtext)


#calculate probability of word
def get_wrd_prob(wrd,pr,a1):
    prob = 0
    #add starting letter prob

    prob += pr[wrd[0]]
    
    #loop through remaining
    cnt = 1
    for i in wrd[1:]:
        prevlett = wrd[cnt - 1]
   
        #only add probabilities we have seen before
        if a1[prevlett].get(wrd[cnt],None) is not None:
       
            prob += a1[prevlett][wrd[cnt]]   
        #add large negative prob
        else:
            prob += -16
        cnt += 1
        
    return prob
    



#calculate probability of sentence
def get_sentence_prob(txt,pr,a1):
    sentprob = 0

    #loop through words in text
    for wrd in txt:
        #clean word
  
        fixedwrd = re.sub(r'[^a-zA-Z]','',wrd)
        if len(fixedwrd) > 0:
            sentprob += get_wrd_prob(fixedwrd,pr,a1)
            
    
    return sentprob



#score text passage
def score_text(txt,pr,a1):
    
    
    lowertext = nl.tokenize.sent_tokenize(txt.lower())
    score = 0
    #loop through each sentence
    for sent in lowertext:
        
        #loop through each word
        words = nl.tokenize.word_tokenize(sent)
        score += get_sentence_prob(words,pr,a1)

    return score
    



def offspring(parents,numchild=3):
    newDNA = []

    #for all parents
    for p in parents:
        #create n children

        for c in range(0,numchild):
            newchild = p.copy()
            r1 = np.random.randint(0,26) +97
            r2 = np.random.randint(0,26) +97
            #cant swap same element
            while r2 == r1:
                r2 = np.random.randint(0,26) +97
            
            #mutate elements

            newchild[chr(r1)] = p[chr(r2)]
            newchild[chr(r2)] = p[chr(r1)]
            
            newDNA.append(newchild)    
        
    newDNA.extend(parents)
 
    return newDNA



def encrpyt_func(txt,encryption):
    
    i = 0
    encryptedtext = ""
    while i < len(txt):
        if txt[i] in encryption:
            encryptedtext += encryption[txt[i]]
            
        else:
            encryptedtext += txt[i]
        i += 1
        
    return encryptedtext



def decrypt_func(txt,decryption):
    
    i = 0
 
    decryptedtext = ""
    while i < len(txt):
   
        if txt[i] in decryption:
    
            decryptedtext += decryption[txt[i]]
        else:
            decryptedtext += txt[i]
   
        i += 1
        
    return decryptedtext
    
    






#creating true mapping encryption of characters
alph  = list(string.ascii_lowercase)
alph2  = list(string.ascii_lowercase) 
random.shuffle(alph2)
encrypt = dict(zip(alph, alph2))

#text to check
originaltext = """I then lounged down the street and found, as I expected, that there
was a mews in a lane which runs down by one wall of the garden. I lent
the ostlers a hand in rubbing down their horses, and received in
exchange twopence, a glass of half-and-half, two fills of shag tobacco,
and as much information as I could desire about Miss Adler, to say
nothing of half a dozen other people in the neighbourhood in whom I was
not in the least interested, but whose biographies I was compelled to
listen to."""

encrpytedtext = encrpyt_func(originaltext.lower(),encrypt)


avgscore = []
#create random 20 DNA
dna_pool = []

#create 20 children
for i in range(0,20):
    alph3  = list(string.ascii_lowercase) 
    random.shuffle(alph3)
    decrypt = dict(zip(alph3.copy(), alph))
    dna_pool.append(decrypt)


keep = 5
iters =400
for i in range(0,iters):
    # create children
    if i > 0:
        print("iter {}".format(i))
        dna_pool = offspring(dna_pool,3)
        

    
    scores = [score_text(decrypt_func(encrpytedtext,prnt),p,a1) for prnt in dna_pool]    
    avgscore.append(np.mean(scores))
    topkeep = np.argsort(-np.array(scores))[0:keep]

    dna_pool = np.array(dna_pool)[topkeep]
 
    
 
#get top decryption
translatedtext = decrypt_func(encrpytedtext,dna_pool[0])
print("BEST DECRYPTION")
print(translatedtext)
print()
print("ACTUAL TEXT")
print( decrypt_func(encrpytedtext,dict((v,k) for k,v in encrypt.items())))


bestkey = dna_pool[0]
print()
print("ENCRYPTED TEXT")
print(encrpytedtext)


sns.lineplot(x=np.arange(0,iters),y=avgscore)
