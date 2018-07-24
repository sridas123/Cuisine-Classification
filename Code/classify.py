from __future__ import division  # floating point division
#from sklearn.cross_validation import StratifiedShuffleSplit
import csv
import random
import math
import numpy as np
import collections
import algos as algs
import sklearn
 
def splitdataset(dataset, trainsize=20000, testsize=10000, testfile=None):
    
    randindices = np.random.randint(0,dataset.shape[0],trainsize+testsize)
    numinputs = dataset.shape[1]-1
    Xtrain = dataset[randindices[0:trainsize],0:numinputs]
    ytrain = dataset[randindices[0:trainsize],numinputs]
    Xtest = dataset[randindices[trainsize:trainsize+testsize],0:numinputs]
    ytest = dataset[randindices[trainsize:trainsize+testsize],numinputs]
    if testfile is not None:
        testdataset = loadcsv(testfile)
        Xtest = dataset[:,0:numinputs]
        ytest = dataset[:,numinputs]        
        
    #Add a column of ones; done after to avoid modifying entire dataset
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))
    Xtrain=Xtrain.astype(float)
    ytrain=ytrain.astype(float)
    Xtest=Xtest.astype(float)
    ytest=ytest.astype(float)
                              
    return ((Xtrain,ytrain), (Xtest,ytest))
 
def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def load_ingred():
    dataset = np.genfromtxt('D:/Grad Studies/Machine Learning/Project/Data/train.csv', delimiter=',',skip_header=1)
    trainset, testset = splitdataset(dataset) 
    return trainset,testset

if __name__ == '__main__':
    
    path="D:\Grad Studies\Machine Learning\Project\Data"
    #print('The scikit-learn version is {}.'.format(sklearn.__version__))
    
    """You have to comment out the line containing loadsusy() or loadmadelon() depending on which dataset you want to run the algorithms on"""
    trainset, testset = load_ingred()
    print("The shapes are",trainset[0].shape,testset[0].shape)
    
    print('Running on train={0} and test={1} samples').format(trainset[0].shape[0], testset[0].shape[0])
    nnparams = {'ni': trainset[0].shape[1], 'nh': 64, 'no': 1}
    """type parameter should be L1,L2,None or Other"""
    """regwt should be user defined parameter"""
    
    classalgs = { 'Random Forest': algs.RandomForest(),
                  'Logistic Regression':algs.Logistic(),
                  'KNearest'  : algs.Kneighbour()
                 }

    classalgs1 = collections.OrderedDict(sorted(classalgs.items()))
                          
    for learnername , learner in classalgs1.iteritems():
        
        print 'Running learner = ' + learnername
        
        # Train model
        learner.learn(trainset[0], trainset[1])
        
        # Test model   
        predictions = learner.predict(testset[0])
        
        accuracy = getaccuracy(testset[1], predictions)
        print 'Accuracy for ' + learnername + ': ' + str(accuracy)
 
