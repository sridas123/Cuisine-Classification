from __future__ import division  # floating point division
import numpy as np
import utilities as utils
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

class RandomForest:
    """
    Implements Random Forest Classifier for Multiclass classification.
    """
    
    def __init__( self, params=None ):
        
         """ Params can contain any useful parameters for the algorithm """
         if params is not None and 'ensemble' in params:
            self.ens = params['ensemble']
         else:
            self.ens = 10
            
         self.model = RandomForestClassifier(n_estimators=self.ens)
        
    def learn(self, Xtrain,ytrain):
        
        """ Learns using the traindata """
        
        self.model = self.model.fit(Xtrain, ytrain)
        
    def predict(self, Xtest):
        
        """Predicts using the Test data"""
        
        ytest = self.model.predict(Xtest)
        return ytest
        
class Kneighbour:
    """
    Implements K-nearest neighbour classifier for multiclass classification
    """
    
    def __init__( self, params=None ):
        
        """ Params can contain any useful parameters for the algorithm """
        
        if params is not None and 'neighbours' in params:
            self.neighbour = params['neighbours']
        else:
            self.neighbour = 3
            
        self.model = KNeighborsClassifier(n_neighbors=self.neighbour)
        
    def learn(self, Xtrain,ytrain):
        
        """ Learns using the traindata """
        
        self.model = self.model.fit(Xtrain, ytrain)
        
    def predict(self, Xtest):
        
        """Predicts using Test Data"""
        
        ytest = self.model.predict(Xtest)
        return ytest
                
class Logistic:
    """
    Logistic Regression classifier which internally employs one vs all classification
    """
    
    def __init__( self, params=None ):
        
        """ Params can contain any useful parameters for the algorithm """
        
        if params is not None and 'regwt' in params:
            self.regwt = params['regwt']
        else:
            self.regwt = 0.1
        
        self.model = LogisticRegression(penalty='l2',multi_class='ovr',fit_intercept=
        False,solver='lbfgs',tol=0.1,C=self.regwt,max_iter=50)
        
    def learn(self, Xtrain,ytrain):
        
        """ Learns using the traindata """

        self.model = self.model.fit(Xtrain, ytrain)
        
    def predict(self, Xtest):
        
        ytest = self.model.predict(Xtest)
        return ytest