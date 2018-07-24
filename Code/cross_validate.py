# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 22:34:38 2015

@author: Srijita
"""

from __future__ import division  # floating point division\
from random import shuffle
import csv
import random
import math
import numpy as np
import collections

import algos as algs
 
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
        
    # Add a column of ones; done after to avoid modifying entire dataset
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))
                              
    return ((Xtrain,ytrain), (Xtest,ytest))

"""10-fold cross validation fold creation"""    
def cross_validate_dataset(slices,fold,numinputs,count):
            
    validation = slices[count]
    training = [item
                for s in slices if s is not validation
                for item in s]
    
    validation=np.asarray(validation)
    training=np.asarray(training) 
               
    Xtrain = training[:,0:numinputs]
    ytrain = training[:,numinputs]
    Xtest =  validation[:,0:numinputs]
    ytest =  validation[:,numinputs]
    
    return ((Xtrain,ytrain),(Xtest,ytest))
    
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
    
    """You have to comment out the line containing loadsusy() or loadmadelon() depending on which dataset you want to run the algorithms on"""
    
    final_accuracy={"Random Forest":[],"Logistic Regression":[]}               
    parm_dict={ 'Random Forest':[5,10,15,20,25,30,35,40,45,50],
          #      'Kneighbour':[5,10],
                'Logistic Regression':[0.0001,0.001,0.01,0.1,1,10,100,1000,10000]}
  #  parm_dict={ 'Random Forest':[5,10],
  #              'Kneighbour':[3,5],
  #              'Logistic Regression':[0.01,1]}
                      
                    
    for repeat in xrange(30):
        
       trainset, testset = load_ingred()
       """The choice of the number of folds should be user-input"""
       fold=10
    
       trainlabel=np.reshape(trainset[1],(-1,1))
       trset = np.hstack((trainset[0],trainlabel))
       numinputs = trset.shape[1]-1
       np.random.shuffle(trset)
       parts = [trset[i::fold] for i in xrange(fold)]
       obj=[] 
       print('Running on train={0} and test={1} samples').format(trainset[0].shape[0], testset[0].shape[0])
       parm_pass={'Random Forest':{'ensemble': 10},
             #     'Kneighbour':{'neighbours':3},
                  'Logistic Regression':{'regwt':0}}
               
       classalgs = {'Random Forest':       algs.RandomForest(parm_pass['Random Forest']),
#                    'Kneighbour':          algs.Kneighbour(parm_pass['Kneighbour']),
                    'Logistic Regression': algs.Logistic(parm_pass['Logistic Regression'])
                 }
                 
       classalgs1 = collections.OrderedDict(sorted(classalgs.items())) 
        
       best_parm=[]
       
       for learnername , learner in classalgs1.iteritems():
        
           print 'Running learner = ' + learnername
        
#           # Train model
           parm_accuracy={}
        
           for j in range(0,len(parm_dict[learnername])):
               parm=[]
               algo_accuracy=[]
               for i in xrange(fold):
            
                   trainset1,validation = cross_validate_dataset(parts,fold,numinputs,i)
            
               
                   if learnername=="Kneighbour":
               
                      parm_pass['Kneighbour']['neighbours']=parm_dict[learnername][j]
                      print("Running Kneighbour with number of neighbours",parm_dict[learnername][j])
                      learner=algs.Kneighbour(parm_pass['Kneighbour'])
                      learner.learn(trainset1[0], trainset1[1])
            
                   elif learnername=="Logistic Regression":
                    
                        parm_pass['Logistic Regression']['regwt']=parm_dict[learnername][j]
                        print("Running Logistic Regression with regularisation parameter",parm_dict[learnername][j])
                        learner=algs.Logistic(parm_pass['Logistic Regression'])
                        learner.learn(trainset1[0], trainset1[1])
                        
                   elif learnername=="Random Forest":
                    
                        parm_pass['Random Forest']['ensemble']=parm_dict[learnername][j]
                        print("Running Random Forest with ensemble",parm_dict[learnername][j])
                        learner=algs.RandomForest(parm_pass['Random Forest'])
                        learner.learn(trainset1[0], trainset1[1])
                   
                   else:       
                        learner.learn(trainset1[0], trainset1[1],None) 
#        
#                  # Test model   
                   predictions = learner.predict(validation[0])
             
                   """Calculating the Accuracy"""
                   accuracy = getaccuracy(validation[1], predictions)
                   algo_accuracy.append(accuracy)
             
               print("The each fold accuracy of "+ learnername +"  ")
               print(algo_accuracy)            
               avg=sum(algo_accuracy)/fold
               print 'Average accuracy for ' + learnername + ': ' + str(avg)


               if learnername=="Linear Regression" or learnername=="Naive Bayes Ones": 
                  break 
               else:
                  parm_accuracy[parm_dict[learnername][j]]=avg
                              
               """Choosing the best parameter"""
               print (parm_accuracy)
        
           if learnername=="Logistic Regression" or learnername=="Random Forest" or learnername=="Kneighbour":        
              maxitem=0
              bestparm=0
              for parm , parm_accur in parm_accuracy.iteritems():   
                  if parm_accur>maxitem:
                     maxitem=parm_accur
                     bestparm=parm 
              best_parm.append(bestparm)       
              print("The best parameter for"+learnername+"is:  "+str(bestparm)+"with accuracy  "+ str(maxitem))       
              print (best_parm)
           
     #  parm_pass['Kneighbour']['neighbours']=best_parm[0]
       parm_pass['Logistic Regression']['regwt']=best_parm[0]  
       parm_pass['Random Forest']['ensemble']=best_parm[1]      
       print parm_pass

       """Run the model on Training data with the best parameter"""
    
       for learnername , learner in classalgs1.iteritems():
           print 'Running learner = ' + learnername
        
           
           if learnername=="Logistic Regression":
              learner= algs.Logistic(parm_pass['Logistic Regression'])
              learner.learn(trainset[0], trainset[1])
           elif learnername=="Kneighbour":
                learner=algs.Kneighbour(parm_pass['Kneighbour'])
                learner.learn(trainset[0], trainset[1]) 
           elif learnername=="Random Forest":
                learner=algs.RandomForest(parm_pass['Random Forest'])
                learner.learn(trainset[0], trainset[1]) 
           else:
               learner.learn(trainset[0], trainset[1],None)
           
           """Run the model on the new test data"""   
           
           predictions = learner.predict(testset[0])
           accuracy = getaccuracy(testset[1], predictions)
           print 'Accuracy for ' + learnername + ': ' + str(accuracy)
           final_accuracy[learnername].append(accuracy)
       print("-----------------------------------------------------------------------")
       print("-----------------------------------------------------------------------")
        
    print (final_accuracy)   
    temp=[]
    for  learnername , accr_queue in final_accuracy.iteritems():
         for i in range(0,len(accr_queue)):
             temp.append(accr_queue[i])
             
    print("The length of temp is",len(temp)) 
        
    for j in range(0,len(temp)):
        
        """error.csv is the file that captures errors for all the algorithms across 40 runs"""
        with open('error.csv','ab') as f1:
            csv_writer = csv.writer(f1,delimiter=',',dialect = 'excel')
            row=(temp[j],"\n")
            print(row)
            csv_writer.writerows([row])

        
            
 
