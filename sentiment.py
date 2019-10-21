import xml.etree.ElementTree as ET
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import datetime
from sklearn.metrics import mutual_info_score
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import feature_extraction
import delta
import os
import re


def XML2arrayRAW(neg_path, pos_path):
    reviews = []
    negReviews = []
    posReviews = []

    for i in open(neg_path, 'r', encoding='utf-8'):
        i = re.sub('(\d)+', 'NUM', i)
        reviews.append(i)
        negReviews.append(i)
    for i in open(pos_path, 'r',encoding='utf-8'):
        i = re.sub('(\d)+', 'NUM', i)
        reviews.append(i)
        posReviews.append(i)


    return reviews,negReviews,posReviews



def XML2arrayRAW1(neg_path, pos_path):
    reviews = []
    negReviews = []
    posReviews = []

    neg_tree = ET.parse(neg_path)
    neg_root = neg_tree.getroot()
    for rev in neg_root.iter('review'):
        reviews.append(rev.text)
        negReviews.append(rev.text)



    pos_tree = ET.parse(pos_path)
    pos_root = pos_tree.getroot()

    for rev in pos_root.iter('review'):
        reviews.append(rev.text)
        posReviews.append(rev.text)

    return reviews,negReviews,posReviews

def GetTopNMI(n,CountVectorizer,X,target):
    MI = []
    length = X.shape[1]


    for i in range(length):
        temp=mutual_info_score(X[:, i], target)
        MI.append(temp)
    MIs = sorted(range(len(MI)), key=lambda i: MI[i])[-n:]
    return MIs,MI


def getCounts(X,i):

    return (sum(X[:,i]))

def extract_and_split(neg_path, pos_path):
    reviews,n,p = XML2arrayRAW(neg_path, pos_path)
    #train, train_target, test, test_target = split_data_balanced(reviews,1000,200)
    train=reviews
    train_target=[]
    test = []
    test_target=[]
    train_target = [0]*1000+[1]*1000
    return train, train_target, test, test_target



def sent(src,dest,pivot_num,pivot_min_st,dim,c_parm,modelo,extraction, time):
    pivotsCounts = []
    #get representation matrix
    if pivot_num>dim:
        weight_str = src + "_to_" + dest + "/weights/w_" + src + "_" + dest + "_" + str(dim)+".npy"
        mat= np.load(weight_str)


        filename = src + "_to_" + dest + "/split/"
        if not os.path.exists(os.path.dirname(filename)):
            #gets all the train and test for sentiment classification
            train, train_target, test, test_target = extract_and_split("data/"+src+"/negative.parsed","data/"+src+"/positive.parsed")
        else:
            with open(src + "_to_" + dest + "/split/train", 'rb') as f:
                train = pickle.load(f)
            with open(src + "_to_" + dest + "/split/test", 'rb') as f:
                test = pickle.load(f)
            with open(src + "_to_" + dest + "/split/train_target", 'rb') as f:
                train_target = pickle.load(f)
            with open(src + "_to_" + dest + "/split/test_target", 'rb') as f:
                test_target = pickle.load(f)

        unlabeled, source, target = XML2arrayRAW("data/" + src + "/" + src + "UN.txt","data/" + dest + "/" + dest + "UN.txt")

        unlabeled = source + train+ target


        #bigram_vectorizer_unlabeled = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=40)
        bigram_vectorizer_unlabeled = feature_extraction_methods(extraction, min_df=40)
        X_2_train_unlabeled = bigram_vectorizer_unlabeled.fit_transform(unlabeled).toarray()

        filename = src + "_to_" + dest + "/" + "pivotsCounts/" + "pivotsCounts" + src + "_" + dest + "_" + str(
            pivot_num) + "_" + str(pivot_min_st)
        with open(filename, 'rb') as f:
            pivotsCounts = pickle.load(f)


        print("train...")

        trainSent=train


        #bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=20)
        bigram_vectorizer = feature_extraction_methods(extraction, min_df=20)

        X_2_train = bigram_vectorizer.fit_transform(trainSent).toarray()
        X_2_test_unlabeld = bigram_vectorizer_unlabeled.transform(trainSent).toarray()

        XforREP = np.delete(X_2_test_unlabeld, pivotsCounts, 1)  # delete second column of C
        rep = XforREP.dot(mat)


        X_dev_test = bigram_vectorizer.transform(test).toarray()
        X_dev_test_unlabeled = bigram_vectorizer_unlabeled.transform(test).toarray()
        XforREP_dev = np.delete(X_dev_test_unlabeled, pivotsCounts, 1)  # delete second column of C
        XforREP_dev = XforREP_dev.dot(mat)
        devAllFeatures = np.concatenate((X_dev_test,XforREP_dev),1)




        allfeatures = np.concatenate((X_2_train, rep), axis=1)


        dest_test, source, target = XML2arrayRAW("data/"+dest+"/negative.parsed","data/"+dest+"/positive.parsed")
       # dest_test = source[0:200]+target[0:200]
        dest_test_target= [0]*1000+[1]*1000
        X_dest = bigram_vectorizer.transform(dest_test).toarray()
        X_2_test = bigram_vectorizer_unlabeled.transform(dest_test).toarray()
        XforREP_dest = np.delete(X_2_test, pivotsCounts, 1)  # delete second column of C
        rep_for_dest = XforREP_dest.dot(mat)
        allfeaturesKitchen = np.concatenate((X_dest, rep_for_dest), axis=1)



        logreg = model(modelo)
        logreg.fit(X_2_train, train_target)
        lgs = logreg.score(X_dest, dest_test_target)
        log_dev_source = logreg.score(X_dev_test, test_target)



        logreg =  model(modelo)

        logreg.fit(allfeatures, train_target)

        lg = logreg.score(allfeaturesKitchen, dest_test_target)
        log_dev_all = logreg.score(devAllFeatures,test_target)



        logregR =  model(modelo)

        logregR.fit(rep, train_target)
        log_dev_rep = logregR.score(XforREP_dev,test_target)
        lgR = logregR.score(rep_for_dest, dest_test_target)

        filename = "results/"+src+"_to_"+dest+"_"+modelo
        if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
        total_time = (datetime.datetime.now() - time).total_seconds()
        sentence = " MI = "+str(pivot_num)+" dim = "+str(dim)+"  on dev : rep = "+str(log_dev_rep)+" , non = " + str(log_dev_source)+" all = "+str(log_dev_all)+ ", on target: rep = "+ str(lgR) + " , non = "+ str(lgs) + " all = "+str(lg)+ " time = "+str(total_time)

        print(sentence)
        with open(filename, "a") as myfile:
            myfile.write(sentence+"\n")



def model(modelo):
        if modelo is 'naive':
            # modelo = naive_bayes.BernoulliNB()
            modelo = MultinomialNB()
        elif modelo is 'logistic':
            modelo = LogisticRegression(C=0.1)
        elif modelo is 'svm':
            # modelo = svm.SVC(kernel="linear")
            modelo = SVC(kernel="linear")
        elif modelo is 'random':
            modelo = RandomForestClassifier()
        elif modelo is 'tree':
            modelo = DecisionTreeClassifier()

        return modelo






def feature_extraction_methods(tipo, min_df):

    stopwords = None
    #bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=20)
    stopwords = open('stopwords', 'r', encoding='utf-8').read().split("\n")

    pre = []
    if tipo is 'tfidf':
        pre = feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b',  stop_words=stopwords, min_df=min_df)
    elif tipo is 'idf':
        pre = feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', stop_words=stopwords, binary=True, min_df=min_df)
    elif tipo is 'counter':
        pre = feature_extraction.text.CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', stop_words=stopwords, min_df=min_df)
    elif tipo is 'binario':
        pre = feature_extraction.text.CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', stop_words=stopwords, min_df=min_df)
    elif tipo is 'delta':
        pre = delta.DeltaTfidfVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', stop_words=stopwords, min_df=min_df)

    return pre