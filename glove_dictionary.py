# -*- coding: utf-8 -*-
"""
This script needs to be executed only once in order to create a dictionary
Download the GloVe model from: https://nlp.stanford.edu/projects/glove/
"""
import numpy as np
import pickle
    
print("Loading Glove Model")
f = open('glove.840B.300d/glove.840B.300d.txt', 'r',encoding='utf-8')
model = {}
for line in f:
    splitLine = line.decode('utf8').strip().split(' ')
    word = splitLine[0]
    embedding = np.array([float(val) for val in splitLine[1:]])
    model[word] = embedding
print("Done.",len(model)," words loaded!")
    
# Saving the dictionary
pickle_out = open('glove.840B.300d_dict.pickle', 'wb')
pickle.dump(model, pickle_out)
pickle_out.close()