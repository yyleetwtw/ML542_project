import preprop as preprop
import logging
from gensim.models import Word2Vec
import numpy as np
import analysis
from sklearn.model_selection import train_test_split, KFold
import os
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, auc, roc_auc_score, roc_curve, accuracy_score, mean_squared_error
import sqlite3
import re
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

file = open('train.csv', 'r')
file = preprop.decoding(file)
y = preprop.getLabel(file)

#Generate uni, bi, and tri-gram embeddings
uni_gram = preprop.genNgram(file, 1)
bi_gram = preprop.genNgram(file, 2)
tri_gram = preprop.genNgram(file, 3)

def getEmbed(RE_TRAIN_EMBBED, X_train):
    if RE_TRAIN_EMBBED:
        """ w2v_uni = Word2Vec(sentences=uni_gram, size=10, window=5, min_count=1, workers=2,
                    sg=1, iter=10)
        w2v_uni.save("w2v_uni.model")
        w2v_bi = Word2Vec(sentences=bi_gram, size=100, window=5, min_count=1, workers=2,
                    sg=1, iter=10)
        w2v_bi.save("w2v_bi.model")
        w2v_tri = Word2Vec(sentences=tri_gram, size=100, window=5, min_count=1, workers=2,
                    sg=1, iter=10)
        w2v_tri.save("w2v_tri.model")"""
        
        w2v_train = Word2Vec(sentences=X_train, size=50, window=5, min_count=1, workers=2,
                        sg=1, iter=10)
        w2v_train.save("w2v_train.model")
        
    #w2v_uni = Word2Vec.load("w2v_uni.model")
    #w2v_bi = Word2Vec.load("w2v_bi.model")
    #w2v_tri = Word2Vec.load("w2v_tri.model")
    w2v_train = Word2Vec.load("w2v_train.model")
    
    return w2v_train

def make_feature_vec(words, model, num_features):
    """
    Average the word vectors for a set of words
    """
    feature_vec = np.zeros((num_features,),dtype="float32")  # pre-initialize (for speed)
    nwords = 1.
    index2word_set = set(model.wv.index2word)  # words known to the model

    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            feature_vec = np.add(feature_vec, model[word])
    
    feature_vec = np.divide(feature_vec, nwords)
    return feature_vec

def get_avg_feature_vecs(names, model, num_features):
    """
    Calculate average feature vectors for all reviews
    """
    counter = 0
    name_feature_vecs = np.zeros((len(names),num_features), dtype='float32')  # pre-initialize (for speed)
    
    for name in names:
        name_feature_vecs[counter] = make_feature_vec(name, model, num_features)
        counter = counter + 1
    return name_feature_vecs

def getVecs(dataset, w2v_train):
    # calculate average feature vectors for training and test sets
    clean_data = []
    for name in dataset:
        clean_data.append(name)
    dataVecs = get_avg_feature_vecs(clean_data, w2v_train, num_features)
    return dataVecs


# control flags
RE_TRAIN_EMBBED = True
num_features = 50


#Generate 10-fold cross validation training and testing splits
kfold = KFold(10)
counter = 1
for train, test in kfold.split(uni_gram):
    print("Fold: " + str(counter))
    counter += 1

    dataset = tri_gram
    
    X_train = dataset[train[0]:test[0]] + dataset[test[-1]:train[-1]]
    y_train = y[train[0]:test[0]] + y[test[-1]:train[-1]]
    X_test = dataset[test[0]:test[-1]]
    y_test = y[test[0]:test[-1]]
    
    #Apply Word2Vec to training set
    w2v_train = getEmbed(RE_TRAIN_EMBBED, X_train)

    # Fit a random forest to the training data, using 100 trees
    forest = RandomForestClassifier(n_estimators = 50)
    trainVecs = getVecs(X_train, w2v_train)
    testVecs = getVecs(X_test, w2v_train)

    print("Fitting a random forest to labeled training data...")
    forest = forest.fit(trainVecs, y_train)

    print("Predicting labels for test data..")
    result = forest.predict(testVecs)

    print("Classification Report and Accuracy")
    print(classification_report(y_test, result))
    print(accuracy_score(y_test, result))

#Generate one time training and testing splits for tuning hyperparameters
X_train, X_test, y_train, y_test = train_test_split(tri_gram, y, test_size=0.2)

#Apply Word2Vec to training set
w2v_train = getEmbed(RE_TRAIN_EMBBED, X_train)
trainVecs = getVecs(X_train, w2v_train)
testVecs = getVecs(X_test, w2v_train)

plt.ylabel('error rate %')
plt.xlabel('number of estimators')

for depth in range(1, 15):
    errors = []
    for n_est in range(20, 150, 20): 
        forest = RandomForestClassifier(n_estimators = n_est, max_depth = depth)
        
        print("Fitting a random forest to labeled training data w/ {:5} estimators and {:3} depth".format(n_est, depth))
        forest = forest.fit(trainVecs, y_train)
        result = forest.predict(testVecs)
        errors.append(mean_squared_error(y_test, result))
        
    line, = plt.plot(range(20, 150, 20), errors, label="depth={:3}".format(depth))
       
plt.legend()        
plt.show()

def write_pred_output(predicted, filename='pred.csv'):
    entry_cnt = 0
    with open(filename, 'w') as file:
        file.write('Id,IsBrazilian\n')
        for pred in predicted:
            if entry_cnt+1 == len(predicted):
                towrite = str(entry_cnt) +',' + str(pred)
            else:
                towrite = str(entry_cnt) +',' +str(pred) + '\n'
            entry_cnt = entry_cnt + 1
            file.write(towrite)

#Create Final Model for Submission:
file = open('train.csv', 'r')
file = preprop.decoding(file)
labels = preprop.getLabel(file)

file_test = open('test.csv', 'r')
file_test = preprop.decoding(file_test)

# generate necessary data for word embedding
uni_gram = preprop.genNgram(file, 1)
bi_gram = preprop.genNgram(file, 2)
tri_gram = preprop.genNgram(file, 3)

uni_gram_test = preprop.genNgram(file_test, 1)
bi_gram_test = preprop.genNgram(file_test, 2)
tri_gram_test = preprop.genNgram(file_test, 3)

if RE_TRAIN_EMBBED:
    w2v_uni = Word2Vec(sentences=uni_gram+uni_gram_test, size=10, window=5, min_count=1, workers=2,
                 sg=1, iter=10)
    w2v_uni.save("w2v_uni10.model")
    w2v_bi = Word2Vec(sentences=bi_gram+bi_gram_test, size=50, window=5, min_count=1, workers=2,
                       sg=1, iter=10)
    w2v_bi.save("w2v_bi50.model")
    w2v_tri = Word2Vec(sentences=tri_gram+tri_gram_test, size=100, window=5, min_count=1, workers=2,
                       sg=1, iter=10)
    w2v_tri.save("w2v_tri100.model")

#Apply Word2Vec to training set
#w2v_train = getEmbed(RE_TRAIN_EMBBED, X_train)

# Fit a random forest to the training data, using 100 trees
forest = RandomForestClassifier(n_estimators = 50)
trainVecs = getVecs(bi_gram, w2v_bi)
testVecs = getVecs(bi_gram_test, w2v_bi)

print("Fitting a random forest to labeled training data...")
forest = forest.fit(trainVecs, y)

print("Predicting labels for test data..")
result = forest.predict(testVecs)

write_pred_output(result)

