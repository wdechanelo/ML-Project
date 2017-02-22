import matplotlib.pyplot as plt

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 17:07:19 2017

@author: lingyu, hehu
"""
# Example of loading the data.

import numpy as np
import os
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

if __name__== '__main__':

    data_path = "." # This folder holds the csv files

    # load csv files. We use np.loadtxt. Delimiter is ","
    # and the text-only header row will be skipped.

    print("Loading data...")
    x_train = np.loadtxt(data_path + os.sep + "x_train.csv",
                         delimiter = ",", skiprows = 1)
    x_test  = np.loadtxt(data_path + os.sep + "x_test.csv",
                         delimiter = ",", skiprows = 1)
    y_train = np.loadtxt(data_path + os.sep + "y_train.csv",
                         delimiter = ",", skiprows = 1)

    print "All files loaded. Preprocessing..."

    # remove the first column(Id)
    x_train = x_train[:,1:]
    x_test  = x_test[:,1:]
    y_train = y_train[:,1:]

    # Every 100 rows correspond to one gene.
    # Extract all 100-row-blocks into a list using np.split.
    num_genes_train = x_train.shape[0] / 100
    num_genes_test  = x_test.shape[0] / 100

    print("Train / test data has %d / %d genes." % \
          (num_genes_train, num_genes_test))
    x_train = np.split(x_train, num_genes_train)
    x_test  = np.split(x_test, num_genes_test)

    # Reshape by raveling each 100x5 array into a 500-length vector
    x_train = [g.ravel() for g in x_train]
    x_test  = [g.ravel() for g in x_test]

    # convert data from list to array
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test  = np.array(x_test)
    y_train = np.ravel(y_train)

    ##################################3

    #Split data to training and test sets
    from sklearn.model_selection import train_test_split
    a_train, a_test, b_train, b_test = train_test_split(x_train, y_train, test_size=0.2)

    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=1000)
    clf.fit(a_train,b_train)

    # Calculate the AUC value with test set
    from sklearn.metrics import roc_auc_score
    predictions = clf.predict(a_test)
    auc = roc_auc_score(b_test,predictions)
    print "AUC " + str(auc)

    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    #Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(a_train.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(a_train.shape[1]), indices,rotation='vertical')
    plt.xlim([-1, 50])
    plt.show()
