# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 15:08:33 2022
LSA vis SVD
@author: kholm
"""



from nltk.corpus import stopwords


from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer 
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

stops = set(stopwords.words('english'))
stops = stops.union(set(["etext","series","edition","approach","printed","guide","third","fourth","volume"]))


#read data
f = open(r"booktitles.txt", "r",encoding="utf-8")
fil = f.read()

#each line (title) is document
trainingtext = fil.lower().split('\n')
lemmatizer = WordNetLemmatizer()

newtraintext = []

#lemmatize words
for l in trainingtext:
    new = l.split(" ")
    newtitle = ""
    for w in new:
        new_w = lemmatizer.lemmatize(w)
        if len(new_w) > 3:
            newtitle += new_w + " "
        
    newtraintext.append(newtitle.rstrip())


cntvect = CountVectorizer(binary=True,stop_words=stops)

doc_term_matrix = cntvect.fit_transform(trainingtext)

term_doct = doc_term_matrix.T

svd = TruncatedSVD(n_components=2, random_state=42)

transformeddata = svd.fit_transform(term_doct)




pio.renderers.default='browser'
plt.figure()
fig = px.scatter(x=transformeddata[:,0],y=transformeddata[:,1],text = cntvect.get_feature_names(),size_max=60)
fig.update_traces(textposition="top center")
fig.show()
