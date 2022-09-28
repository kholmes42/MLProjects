# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 18:10:44 2022
EXTRACTIVE TEXT SUMMARIZATION USING TF-IDF w Text Rank Scoring
@author: kholm
"""


import pandas as pd
import numpy as np


import nltk as nl

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



df = pd.read_csv(r"bbc_text_cls.csv")



# function to take article and write extractive summary
def extract_summarize_text(txt,top=1):
    
    #create model
    tfidf = TfidfVectorizer(norm="l1",stop_words="english")
    
  
    #token into sentences
    sentences = nl.tokenize.sent_tokenize(txt)
    
    #score each setence
    vects = tfidf.fit_transform(sentences)
    scores = cosine_similarity(vects)
    uni = np.full((scores.shape[0], scores.shape[0]), 1/scores.shape[0]) 
    
    

    #smooth matrix
    textrankmatrix = 0.0001*uni + (1-0.0001)*scores
    textrankmatrix /= np.sum(textrankmatrix,axis=1)[:,np.newaxis]
    
    
    #use text rank algo (find eigenvector with eigenval = 1)
    vals,vecs = np.linalg.eig(textrankmatrix.T)
    z = vecs[:,0]/np.sum(vecs[:,0])
    rnks = np.argsort(-z)[0:top]
    sent = np.array(sentences)
    
    return  sent[rnks]



df["Summary"] = df["text"].apply(extract_summarize_text,top=5)
