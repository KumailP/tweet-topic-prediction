# -*- coding: utf-8 -*-

import pandas as pd
import sys
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import nltk

# Load spreadsheet with tweets of all users
df = pd.read_excel('Twitter_timeline.xlsx', None)

# Combine all users and drop null values
all_df = []
for key in df.keys():
    all_df.append(df[key])
data_concatenated = pd.concat(all_df,axis=0,ignore_index=True, sort=False)

data_concatenated.dropna(subset=['text'], inplace=True)
data_concatenated.dropna(subset=['tags'], inplace=True)

# Only select text and tags and save in np array
data_concatenated = data_concatenated[['text', 'tags']]
data = np.array(data_concatenated)

#documents = data[:,0]

#stemmer = SnowballStemmer('english')

#def lemmatize_stemming(text):
#    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

#def preprocess(text):
#    result = []
#    for token in gensim.utils.simple_preprocess(text):
#        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
#            result.append(lemmatize_stemming(token))
#    return result

#doc_sample = documents[documents['index'] == 4310].values[0][0]

#print('original document: ')
#words = []
#for word in doc_sample.split(' '):
#    words.append(word)
#print(words)
#print('\n\n tokenized and lemmatized document: ')
#print(preprocess(doc_sample))

#processed_docs = documents['headline_text'].map(preprocess)

#processed_docs[:10]