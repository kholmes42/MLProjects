# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 14:23:34 2022
WORD EMBEDDING CHECK
@author: kholm
"""

import pandas as pd
import csv
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv(r"glove.6B\glove.6B.50d.txt"
                 ,quoting=csv.QUOTE_NONE,header=None,sep= " ",encoding='utf-8',index_col=0)



print(df.head())
print(df.shape)



def get_most_similar(search,embed,top=1):
    
    
    if (isinstance(search, pd.Series)):
        search = search.to_numpy()
    
    scores = cosine_similarity(embed, search.reshape(1,search.shape[0]))
    
    ranks = np.argsort(-scores,axis=0)
    
    
    return embed.index[ranks[1:top+1][:,0]].values



def get_analogy(embed,froms,to,sim):
    
    frm = embed.loc[froms.lower()]
    ts = embed.loc[to.lower()]
    sm = embed.loc[sim.lower()]
    
    similarembed = frm - ts + sm
    similarembed = similarembed.to_numpy()
    
    

    return get_most_similar(similarembed, embed)


f = "king"
t = "man"
eq = "woman"

out = get_analogy(df, f, t, eq)[0]
print()
print("{} is to {} as {} is to {}.".format(f,t,out,eq))



f = "december"
t = "november"
eq = "june"

out = get_analogy(df, f, t, eq)[0]
print()
print("{} is to {} as {} is to {}.".format(f,t,out,eq))


lookupword = df.loc["king"]
sims = get_most_similar(lookupword,df,top=5)
print()
print("{} is most similar to {}.".format(lookupword.name,sims))

