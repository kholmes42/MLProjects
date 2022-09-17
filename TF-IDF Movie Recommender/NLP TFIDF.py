
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 14:58:40 2022
MOVIE RECOMMENDER USING TF IDF
@author: kholm
"""


import pandas as pd
import json as js

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv(r"tmdb_5000_movies.csv")



dfnew = df.dropna(subset=['overview'])



data = dfnew["overview"]
"""
OVERVIEW ATTEMPT
"""

#train TFIDF
tfidf = TfidfVectorizer(stop_words="english",strip_accents="ascii")
matdata = tfidf.fit_transform(data)


#movie to compare
checkmovie = "Batman v Superman: Dawn of Justice"
tgt = dfnew[dfnew["original_title"] == checkmovie].index 


#compute pairwise cosine sim
rankings = cosine_similarity(matdata, matdata[tgt,:])
#rank and sort
rank = pd.Series(rankings.ravel())
rank.sort_values(inplace=True,ascending=False)

#get top 5

print()
print("USING OVERVIEW")
print("COSINE SIMILAR MOVIES to {}".format(checkmovie))
print(dfnew.iloc[rank.iloc[1:6].index,:]["original_title"])
print(rank.iloc[1:6])

def convertjson(lst):
    
    newstr = ""
    dic = js.loads(lst)
    for i in dic:
        newstr += " " + i.get("name",None).replace(" ","")
        
  
    return newstr


"""
KEYWORD ATTEMPT
"""
pd.options.mode.chained_assignment = None  # default='warn'

#TRY WITH KEYWORDS ONLY
dfnew["genres2"] = dfnew["genres"].apply(convertjson)
dfnew["keywords2"] = dfnew["keywords"].apply(convertjson)

data2 = dfnew["genres2"] + " " + dfnew["keywords2"]
tfidf = TfidfVectorizer(max_features=2000)
matdata = tfidf.fit_transform(data2)


#movie to compare
tgt = dfnew[dfnew["original_title"] == checkmovie].index 


#compute pairwise cosine sim
rankings = cosine_similarity(matdata, matdata[tgt,:])
#rank and sort
rank2 = pd.Series(rankings.ravel())
rank2.sort_values(inplace=True,ascending=False)

#get top 5

print()
print("USING KEYWORDS")
print("COSINE SIMILAR MOVIES to {}".format(checkmovie))
print(dfnew.iloc[rank2.iloc[1:6].index,:]["original_title"])
print(rank2.iloc[1:6])