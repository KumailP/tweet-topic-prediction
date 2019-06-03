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
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

# Load spreadsheet with tweets of all users
df = pd.read_excel('Twitter_timeline.xlsx', None)

# Combine all users and drop null values
all_df = []
for key in df.keys():
    all_df.append(df[key])
data_concatenated = pd.concat(all_df,axis=0,ignore_index=True, sort=False)

lenBefore = data_concatenated.shape[0]

print("Tweets before cleaning: {0}".format(lenBefore))

data_concatenated.dropna(subset=['text'], inplace=True)
data_concatenated.dropna(subset=['tags'], inplace=True)

# Only select text and tags and save in np array
data_concatenated = data_concatenated[['text', 'tags']]
data = np.array(data_concatenated)

print("Total NaN Tweets dropped: {0}".format(lenBefore-data.shape[0]))
print("Total RJ Tweets dropped: {0}".format(np.where(data[:,1]=="RJ")[0].size))

data = data[np.where(data[:,1] != "RJ")]

# Data visualization
print("Tweets after cleaning: {0}".format(data.shape[0]))

tags = ["BN", "PT", "HT", "ST", "ED", "SP", "EN", "BN", "SI", "RE", "GM", "NW"]

noOfTweets = list(tags)

for i in range(len(tags)):
    noOfTweets[i] = np.where(data[:,1] == tags[i])[0].size
    print("Number of {0} Tweets: {1}".format(tags[i], noOfTweets[i]))

y_pos = np.arange(len(tags))
x_vals = noOfTweets

plt.style.use('ggplot')
plt.bar(y_pos, x_vals, align='center', alpha=0.5)
plt.xticks(y_pos, tags)
plt.ylabel('No. of Tweets')
plt.xlabel("Tags")
plt.title('Number of tweets for each tag')

plt.show()

#documents = data[:,0]

stemmer = SnowballStemmer('english')

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

processed_docs = []

for doc in data[:,0]:
    processed_docs.append(preprocess(doc))
    