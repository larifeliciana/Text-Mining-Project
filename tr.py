



import pre

import os
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import linear_model
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA






def train(src,dest,pivot_num,pivot_min_st):
    dim = pivot_num
    x, y, x_valid, y_valid,inputs= pre.preproc(pivot_num,pivot_min_st,src,dest)
    print("training......")
    pivot_mat = np.zeros((pivot_num, inputs))
    print("0")
    for i in range(pivot_num):
   #     print(i)
        clf = linear_model.SGDClassifier(loss="modified_huber")
        clf.fit(x,y[:,i])

        pivot_mat[i]=clf.coef_
    print("finish traning")
    pivot_mat=pivot_mat.transpose()
    if 10 < dim:
        svd10 = TruncatedSVD(n_components=10)
        pivot_mat10 = svd10.fit_transform(pivot_mat)
        weight_str = src + "_to_" + dest + "/weights/w_" + src + "_" + dest + "_" + str(10)
        filename = weight_str
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        np.save(weight_str, pivot_mat10)

    if 40<dim:
        svd40 = TruncatedSVD(n_components=40)
        pivot_mat40=svd40.fit_transform(pivot_mat)
        weight_str = src + "_to_" + dest + "/weights/w_" + src + "_" + dest + "_" + str(40)
        filename = weight_str
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        np.save(weight_str, pivot_mat40)

    if 50<dim:
        svd50 = TruncatedSVD(n_components=50)
        pivot_mat50=svd50.fit_transform(pivot_mat)
        weight_str = src + "_to_" + dest + "/weights/w_" + src + "_" + dest + "_" + str(50)
        filename = weight_str
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        np.save(weight_str, pivot_mat50)

    if 100 < dim:

        svd100 = TruncatedSVD(n_components=100)
        pivot_mat100=svd100.fit_transform(pivot_mat)
        weight_str = src + "_to_" + dest + "/weights/w_" + src + "_" + dest + "_" + str(100)
        filename = weight_str
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        np.save(weight_str, pivot_mat100)

    if 150 < dim:

        svd150 = TruncatedSVD(n_components=150)
        pivot_mat150=svd150.fit_transform(pivot_mat)

        weight_str = src + "_to_" + dest + "/weights/w_" + src + "_" + dest + "_" + str(150)
        filename = weight_str
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        np.save(weight_str, pivot_mat150)


    print("finished svd")

