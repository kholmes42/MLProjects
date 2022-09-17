# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 16:11:50 2022
EXTRACTIVE TEXT SUMMARIZATION USING TF-IDF
@author: kholm
"""


import pandas as pd
import numpy as np


import nltk as nl


from sklearn.feature_extraction.text import TfidfVectorizer




df = pd.read_csv(r"bbc_text_cls.csv")



# function to take article and write extractive summary
def extract_summarize_text(txt,top=1):
    
    #create model
    tfidf = TfidfVectorizer(norm="l1",stop_words="english")
    
  
    #token into sentences
    sentences = nl.tokenize.sent_tokenize(txt)
    
    #score each setence
    vects = tfidf.fit_transform(sentences)
    scores = vects.sum(axis=1) / vects.astype(bool).sum(axis=1)

    
    rnks = np.array(np.array(scores.argsort(axis=0))[::-1][:top]).flatten()

    rnks = list(rnks)
    sent = np.array(sentences)
    
    
    
    return  sent[rnks]


df["summary"] = df["text"].apply(extract_summarize_text,top=5)



