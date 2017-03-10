#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 03:09:01 2017

@author: willy
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import LinearSVC
from math import log10 
from sklearn.cross_validation import cross_val_score
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV as rfe
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Activation
    


def submit(y_pred, name):
    filename = "%s.csv" %name
    with open(filename, 'w') as f:
        f.write("GeneId,Prediction\n")
        for i in range(1, len(y_pred)+1):
            f.write("{},{}\n".format(i, y_pred[i - 1]))
    print("Submission ready!")


def bestClassifier(Results):
    ind_max = 0 
    max_acc = Results[0][1]
    for i in range(1,len(Results)):
#        print  float(Results[i][1])
#        print max_acc
#        print  float(Results[i][1]) > max_acc
#        print ("\n\n")
        if (float(Results[i][1]) > max_acc):
            max_acc = Results[i][1]
            ind_max = i
    return ind_max
    
def returnIndexOfMax(array):
    ind = -1 
    max_value = max(array)
    for value in array:
        ind +=1
        if value == max_value:
            break
    return ind
                

def featureSelector (support, x_data):   
    selected_features = []
    inter = []
    support = np.array(support)
    x_data = np.array(x_data)
    for i in range(x_data.shape[0]):
        for j in range(support.shape[0]):
            if support[j]==True:
                inter.append(x_data[i,j])
        selected_features.append(inter)
        inter = []
    selected_features = np.array(selected_features)
    return selected_features
     
    
    
    
if __name__== '__main__':


    # load csv files. We use np.loadtxt. Delimiter is ","
    # and the text-only header row will be skipped.
    
    print("Loading data...")
    x_train = np.loadtxt("x_train.csv",delimiter = ",", skiprows = 1)
    x_test  = np.loadtxt("x_test.csv",delimiter = ",", skiprows = 1)    
    y_train = np.loadtxt("y_train.csv",delimiter = ",", skiprows = 1)

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
    
    # Now x_train should be 15485 x 500 and x_test 3871 x 500.
    # y_train is 15485-long vector.
    
    print("x_train shape is %s" % str(x_train.shape))    
    print("y_train shape is %s" % str(y_train.shape))
    print("x_test shape is %s" % str(x_test.shape))
    
    print('Data preprocessing done...')
    

    
    
    
    
    
    """I. Logistic regression with default parameters"""
    
    
    
    
    #fit and predit 
    LR_default = LogisticRegression()
    LR_default.fit(x_train,y_train)
    y_pred = LR_default.predict_proba(x_test)
    #preparing the submission
    y_hat = y_pred[:,1]
    submit(y_hat,"LR_defaut")
    
    
    
    
    
    
    """ II. Experiments with parameter C and penalty"""
    
    
    
    
    #First let's split our training datas into two parts each
    X_train2, X_test2, y_train2, y_test2 = train_test_split(x_train, y_train, test_size=0.2)
    
    #first let test the impact of C parameter on the accuracy  
    result  = []; score = [[],[],[]]
    ind = 0;ind2=0
    C_range = np.array([1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e-0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    rang = np.arange(-10,10)
    clf  = LogisticRegression()
    score.append(list(C_range))
    for penalty in ["l1", "l2"]:
       clf.penalty = penalty
       for C in C_range:
            clf.C = C
            clf.fit(X_train2, y_train2)
            y_pred = clf.predict_proba(X_test2)
            y_f = y_pred[:,1]
            accuracy = 100*roc_auc_score(y_test2, y_f)
            score[ind+1].append(accuracy) 
       ind += 1 
     
    # we are going to select now the best params
    selected_params = []
    ind = 1 
    if (max(score[1])<max(score[2])):
        ind = 2
    ind_best = returnIndexOfMax(score[ind])
    selected_params.append(["l%s"%(ind),C_range[ind_best]])
 
    # Let's plot the accuracy depending on the C parameter for the 1st classifier
    for i in range(1,3):
        plt.figure(i)
        plt.title("Accuracy depending on C param : logistic reg with l%d penalty" %(i))
        #plt.axis([min(C_range),max(C_range),0.75,0.9])
        plt.plot(rang,score[i])

  
    #prediction with the best parameter
    clf.penalty = selected_params[0][0]
    clf.C = selected_params[0][1]
    clf.fit(x_train,y_train)
    y_sub1 = clf.predict_proba(x_test)
    y_sub = y_sub1[:,1]
    submit(y_sub,"Logistic_best_params")

    
    #Table to present
    TableLogReg = []

    TableLogReg.append(score)
    TableLogReg = np.array(TableLogReg).T
#    TableLogReg = list(TableLogR)
    print(TableLogReg)
    
    
    
    
    
    
    
    """III. Preprocessing and Cross-validation"""
    
    
    
    
    pca = decomposition.PCA()
    pipe = Pipeline(steps=[('pca', pca),('logistic', clf)])
    n_components = [10,20,30,40,50,60,70,80,90,100,120,150,200,250,300,350,400]

    #pipelining with gridsearch
    estimator = GridSearchCV(pipe,
                             dict(pca__n_components=n_components),
                             cv = 5,
                             scoring = 'roc_auc')
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict_proba(x_test)
    y_f = y_pred[:,1]
    submit (y_f,"pca_pipelining")
    
    
    
    
    
    
    
    
    """IV. Customer Tasks """
    
    
    """ The main objectives was to detect the best histone markers by making a features selection 
    we choose to do it in two phase, the first with rfecv and also with random forest"""
    
    
    
    
    
    
    """-> Features selection with random forest estimator"""
    
    clf_rf = RandomForestClassifier(n_estimators = 100)
    clf_rf.fit(x_train, y_train)
    importances = clf_rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]
    
    print("Feature ranking:")
    proba = [0] * 5
    for f in range(x_train.shape[1]):
        #each line in proba correspond to one histone marker
        print("%d. Histone %d (%f)" % (f + 1, indices[f] % 5, importances[indices[f]]))
        proba[indices[f] % 5]+=importances[indices[f]]
    print(proba)
    print(sum(proba))
    
    
    """-> Features selection with RFECV with our logistic regression classifier as estimator"""
    
    rfecv = rfe(estimator=XGBClassifier(n_estimators=650,max_depth=18),step=50,verbose = 1,cv=10)
    rfecv.fit(x_train, y_train)
    rfecv_scores = rfecv.grid_scores_

    #now we are going to identify the number of features selected from each histone marker
    support = rfecv.support_
    
    nb_selected_features = 0
    histone_marker = []
    #histone marker H3K4me3
    for i in range(0,500, 5):
        if (support[i]==True):
            nb_selected_features +=1
    histone_marker.append(['H3K4me3',nb_selected_features])
    #histone marker H3K4me1
    nb_selected_features=0
    for i in range(1,500, 5):
        if (support[i]==True):
            nb_selected_features +=1
    histone_marker.append(['H3K4me1',nb_selected_features])
    #histone marker H3K36me3
    nb_selected_features=0
    for i in range(2,500, 5):
        if (support[i]==True):
            nb_selected_features +=1
    histone_marker.append(['H3K36me3',nb_selected_features])
    #histone marker H3K9me3
    nb_selected_features=0
    for i in range(3, 500, 5):
        if (support[i]==True):
            nb_selected_features +=1
    histone_marker.append(['H3K9me3',nb_selected_features])
    #histone marker H3K27me3
    nb_selected_features=0
    for i in range(4,500,5):
        if (support[i]==True):
            nb_selected_features +=1
    histone_marker.append(['H3K27me3',nb_selected_features])
    
    #Let's print the result
    percentage_hist= 0
    print("Proportion of selected features from each histone marker")
    for i in range(5):
        value = histone_marker[i][1]/2.5
        print("%s : %f"%(histone_marker[i][0],value))
        percentage_hist += value
    print ("Total  : %f " %percentage_hist)
   
        
    
    
    
    
    
        

    """ V. Improvement of the accuracy with different classifiers"""
    
    
    
    
    
    
    
   
#First step : training the classifier seen in class
    classifiers1 = [(RandomForestClassifier(), "Random Forest"),(ExtraTreesClassifier(), "Extra-Trees"),(AdaBoostClassifier(), "AdaBoost"),(GradientBoostingClassifier(), "GB-Trees")]
    classifiers3 = [(KNeighborsClassifier(),"KNeighbors")]
    Results = []
    Predicted_data = []              
    counter = 0 
    #spliting training data in two in order to test the accuracy
    X_train2, X_test2, y_train2, y_test2 = train_test_split(x_train, y_train, test_size=0.2)
    
    
    #implementing the 1st group of classifiers
    estimators = [100,500,1000]
    for clf, name in classifiers1:
        for est_val in estimators:
            clf.n_estimators = est_val
            #clf.n_jobs = -1
            counter += 1
           # for iteration in range(100):
            clf.fit(X_train2, y_train2)
            y_hat1 = clf.predict_proba(X_test2)
            y_hat = y_hat1[:,1]
            accuracy = roc_auc_score(y_test2, y_hat)
            print ("Test %d" %counter)
            print ("classifier name : %s " %name)
            print("score %f" %accuracy)
            print("n_estimators %d" %clf.n_estimators )
            print("\n\n")
            acc_and_name = [name,accuracy,clf.n_estimators]
            Results.append(acc_and_name)
            #clf.fit(x_train, y_train)
            #Predicted_data.append(clf.predict(x_test))
        
                

    neighbors_number = [5,10,20,30,40,60,80,100]
    metric_range = ['euclidean','manhattan','chebyshev']
    # implementing 3rd classifier 
    for clf,name in classifiers3:
       for metric in metric_range:
           clf.metric = metric
           for n_neighbors in neighbors_number:
                clf.n_neighbors = n_neighbors 
                counter += 1 
                clf.fit(X_train2, y_train2)
                y_pred1 = clf.predict_proba(X_test2)
                y_pred = y_pred1[:,1]              
                score = roc_auc_score(y_test2, y_pred)
                acc_and_name = [name,score,metric,n_neighbors]
                Results.append(acc_and_name)   
                print ("Test %d" %counter)
                print ("classifier name : %s " %name)
                print("score %f" %score)
                print("neighbors %f" %n_neighbors )
                print("metric  %s" %metric )
                print("\n\n")

        

    #selection of the best accuracy and generation of the submission
    best_clf_ind = bestClassifier(Results)
    res = str(Results[best_clf_ind])
    print("The best classifier is %s" %res)
                
                
  
    
# classifier : XGBoost

   # First we tried to find the best max_depth parameter 
   #Here we take many value of n_estimators to generalise 
    tes_scores= []
    depth_range = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,30,50,70,90,100]
    clf_test = XGBClassifier(n_estimators=10,silent=False)
    for valeur in depth_range:      
        clf_test.max_depth=valeur
        clf_test.fit(X_train2,y_train2)
        y_pr = clf_test.predict_proba(X_test2)
        tes_scores.append(roc_auc_score(y_test2,y_pr[:,1]))
    
    tes_scores3= []
    clf_test.n_estimators=100
    for valeur in depth_range:      
        clf_test.max_depth=valeur
        clf_test.fit(X_train2,y_train2)
        y_pr = clf_test.predict_proba(X_test2)
        tes_scores3.append(roc_auc_score(y_test2,y_pr[:,1]))
        
    tes_scores2= []
    clf_test.n_estimators=20
    for valeur in depth_range:      
        clf_test.max_depth=valeur
        clf_test.fit(X_train2,y_train2)
        y_pr = clf_test.predict_proba(X_test2)
        tes_scores2.append(roc_auc_score(y_test2,y_pr[:,1]))
    
    tes_scores4= []
    clf_test.n_estimators=1000
    for valeur in depth_range:      
        clf_test.max_depth=valeur
        clf_test.fit(X_train2,y_train2)
        y_pr = clf_test.predict_proba(X_test2)
        tes_scores4.append(roc_auc_score(y_test2,y_pr[:,1]))

    tes_scores5= []
    clf_test.n_estimators=500
    for valeur in depth_range:      
        clf_test.max_depth=valeur
        clf_test.fit(X_train2,y_train2)
        y_pr = clf_test.predict_proba(X_test2)
        tes_scores5.append(roc_auc_score(y_test2,y_pr[:,1]))

    #plot of the results 
    plt.figure(3)
    plt.title("accuracy depending on the max_depth value")
    plt.ylim(ymax = 0.925, ymin = 0.895)
    line1 = plt.plot(depth_range,tes_scores5,'c',label="500 estimators");
    line2 = plt.plot(depth_range,tes_scores4,'k',label="1000 estimators");
    line3 = plt.plot(depth_range,tes_scores2,'r',label="20 estimators");
    line4 = plt.plot(depth_range,tes_scores3,'g',label="100 estimators");
    line5 = plt.plot(depth_range,tes_scores, label = "10 estimators");    
    plt.legend((line1,line2,line3,line4,line5),('500 estimators','1000 estimators','100 estimators','20 estimators','10 estimators')) 
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    best_depth = depth_range[returnIndexOfMax(tes_scores5)]
    
    #on the plot we can see that the 500 estimators curve is slightly above the 1000 one
    #so the best estimators number will be < 1000
    #features selection
    x_train_selected = np.array(featureSelector(support,x_train))
    x_test_selected = np.array(featureSelector(support, x_test))
    
    
    #But let's verifiy it in a figure
    
    scoreXGB4 = []
    clf_test4 = XGBClassifier(max_depth=18,silent=False)
    est_range4 = [100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]
    for value in est_range4:
        clf_test4.n_estimators=value
        clf_test4.fit(x_train_selected,y_train)
        y_pred2 =  clf_test4.predict_proba(x_test_selected)
        y_f2 = y_pred2[:,1]
        submit(y_f2,"XGboos_est_v2_%d" %value)
        scoreXGB4.append(100*np.mean(cross_val_score(clf_test4, x_train_selected, y_train, cv=5, scoring='roc_auc'))) 
    plt.figure(4)
    plt.title("accuracy depending on the nb_estimators, max_depth=%d"%best_depth)
    plt.plot(est_range4[5:],scoreXGB4[5:])
    
    
    # we can see on the plot that we were right 
    #then we restrain the interval again and find the maximum then submit to kaggle
   
    scoreXGB2 = []
    clf_test4 = XGBClassifier(max_depth=18,silent=False)
    est_range2 = [100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]
    for value in est_range2:
        clf_test4.n_estimators=value
        clf_test4.fit(x_train_selected,y_train)
        y_pred2 =  clf_test4.predict_proba(x_test_selected)
        y_f2 = y_pred2[:,1]
        submit(y_f2,"XGboos_est_v2_%d" %value)
        scoreXGB2.append(100*np.mean(cross_val_score(clf_test4, x_train_selected, y_train, cv=5, scoring='roc_auc')))
    plt.figure(5)
    plt.plot(est_range2,scoreXGB2)
    
    #now we select the file ith the best accuracy and submit it
    print("the file to submit is  XGboos_est_v2_%d.csv"%est_range2[returnIndexOfMax(scoreXGB2)])

   
    
    
    
    
    
    
    
    
    
    """ VI. Neural network """
    
    
        
  

    
    
    
    # Add layers one at the time. 
    nodes_range = [50,100,150,200,250]
    epoch_range = [10,15,20,25,30,35,40]
    batch_range = [1,2,3,4,5,6,7,8,9,10]
    result_per_nodes = []
    for nb_nodes in nodes_range :
        clf_keras = Sequential() 
        clf_keras.add(Dense(nb_nodes, input_dim=X_train2.shape[1], activation = "sigmoid"))
        clf_keras.add(Dense(nb_nodes, activation = "sigmoid"))
        clf_keras.add(Dense(nb_nodes, activation = "sigmoid"))
        clf_keras.add(Dense(nb_nodes, activation = "sigmoid"))
        clf_keras.add(Dense(1, activation = "sigmoid")) 
        clf_keras.compile(loss="mean_squared_error", optimizer="sgd")
        result = []
        for epoch_value in epoch_range:
            roc_score = []
            for batch_value in batch_range:
                clf_keras.fit(X_train2, y_train2, nb_epoch=epoch_value, batch_size=batch_value) # takes a few seconds
                keras_pred = clf_keras.predict_proba(X_test2)
                submit(keras_pred[:,0],"NN_params_%f_%f_%f"%(nb_nodes,epoch_value,batch_value))
                roc_value = 100*roc_auc_score(y_test2,keras_pred)
                roc_score.append(roc_value)
            result.append(roc_score)
        result_per_nodes.append(result)
        

    #get the node with the best score
    max_score2= []
    for i in range (len(nodes_range)):
        max_score1= []
        for j in range(len(epoch_range)):  
            ind_best_batch = returnIndexOfMax(result_per_nodes[i][j][:])
            max_score1.append(result_per_nodes[i][j][ind_best_batch])
        ind_best_epoch = returnIndexOfMax(max_score1)
        max_score2.append(max_score1[ind_best_epoch])
    ind_best_node = returnIndexOfMax(max_score2)
    
    #get the number of epoch of that node which give this best score
    max_score1 = []
    for k in range(len(epoch_range)):
       ind_best_batch =  returnIndexOfMax(result_per_nodes[ind_best_node][k][:])
       max_score1.append(result_per_nodes[ind_best_node][k][ind_best_batch])
    ind_best_epoch = returnIndexOfMax(max_score1)
    
    #get the batch_size corresponding
    ind_best_batch = returnIndexOfMax(result_per_nodes[ind_best_node][ind_best_epoch][:])
   
    print ("the best parameter are : \n node : %d \n epoch : %d \n batch_size : %d"%(nodes_range[ind_best_node],epoch_range[ind_best_epoch],batch_range[ind_best_batch]))
    print("\n with the score %f"%(result_per_nodes[ind_best_node][ind_best_epoch][ind_best_batch]))
 
    
    
    
