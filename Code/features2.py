import dictionary as dicts
import numpy as np
import unicodedata
import random
import csv
   #!/usr/bin/python
          # -*- coding: iso-8859-15 -*-
import os, sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import metrics
from sympy.mpmath.tests.test_quad import xtest_double_7

FEATURES_IN = 'readable_ingredients.out' #list of chosen features 
FEATURES_OUT = 'readable.mid_features.out'# list of features and associated column
TEST_IN = 'test_data' #features to test against
TEST_OUT= open('test_data.csv','wb')
TRAIN_IN = 'train_data' #1 recipe per list unmodified total 1000 50 of each
FULL_IN = 'result.csv' #orignal file
FULL_OUT = open('Ingredients_cuisine_recipe_train.csv', 'wb')
#TRAIN_OUT = 'train_data.csv' # 100 column of ingredients followed by cuisine, and recipe number
TRAIN_OUT= open('train_data.csv', 'wb')
#INGREDIENT_COUNT = 118
#TRAIN_COUNT = 1000 #set this to a small number if ONE_AT_A_TIME is true
#TEST_COUNT = 37000
BUILD_CSV = False #use this if the train count is unknown or very large
CUISINE = False# adds a column for the cuisine
RECIPE =False # adds a column for ingredients
FULL_DICT = True  #uses all the ingredients
SELECT_SAMPLE =False #have a preselected subset of the datta
KAGGLE = True
KAGGLE_OUT=open('kaggle.out','wb')
#figures out the number of lines in a file
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


#takes a .csv of features to use
def generate_features():
    feature_dict = {} #dictionary of features value is array location key is ingredient name
    local_count = 0 #used to create array locations 
    fileout = open(FEATURES_OUT,'w')
    datas= open(FEATURES_IN,'r') 
    line = datas.read()
    line = line.replace('\r','')
    line = line.split('\n')

    for item in sorted(line):
        if len(item)>2: #removes eampty item
            feature_dict[item]= local_count #builds dictionary    
            item = "'"+item+"':"+str(local_count)+'\n'#builds a readable list for the dictionanry
            fileout.write(item)
            local_count+=1
            
    datas.close()
    fileout.close()
    return feature_dict
 #makes a full array
#distro = np.zeros((TRAIN_COUNT, INGREDIENT_COUNT+XTRA_FEATURES))

# Y array
#y = np.zeros(TRAIN_COUNT)

#uesd to write a csv file one row at a time
#spamwriter = csv.writer(TRAIN_OUT, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)



#takes a string removes some of the symbols and and returns a list spit on commas
def formatting(line):
    line = line.replace(", ",' ')
    line = line.replace("'",'')
    line = line.replace('\n','')
    line = line.replace('"','')
    #start =line.find( '(' )
    #end = line.find( ')' )
    #if start != -1 and end != -1:
    #    line = line[start+1:end]
    line = line.split(',')
    return line

#
def build_data(features, ingredient, infile, outfile, kaggle=False):
    feat_len=len(features)
    train_count = file_len(infile)
    print("Sample size of ", train_count)
    print('here we go')
    
    
    distro = np.zeros((train_count, len(features)+CUISINE+RECIPE))
    y = np.zeros(train_count)
    #uesd to write a csv file one row at a time
    spamwriter = csv.writer(outfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    print(kaggle)
    #with open('result.csv','r',encoding='utf-8') as datas:
    with open(infile,'r') as datas: #reads training data
        row_count=0
        while True:
            line=datas.readline()
            if not line: break
            line = formatting(line)
            if not kaggle:
                if len(line)>2:               #avoids empty item
                    for x in range(2,len(line)): #skips cusine and recipe number\
                        if line[x] in features:
                            distro[row_count][features[line[x]]] = 1#adds each ingredient to the correct column if it exists
                        try:
                            y[row_count] = dicts.cuisine_dict[line[1]]#next to last column is cuisine
                            if CUISINE:
                                distro[row_count][feat_len] = dicts.cuisine_dict[line[1]]#next to last column is cuisine
                            if RECIPE:
                                distro[row_count][feat_len+CUISINE] = line[0] # makes the last item in the array the recipie #
                        except:
                            print(line[1])
                            print (dicts.cuisine_dict[line[1]])
                            temp=raw_input('geh')
                    if BUILD_CSV:                   
                        build_string= ''
                        for location in range(len(distro[row_count])):
                            build_string+=str(distro[row_count][location])+','
                        build_string+= str(dicts.cuisine_dict[line[1]])+','+str(line[0])
                        spamwriter.writerow(build_string)
                        
                    
                    row_count+=1
            else:
                if len(line)>2:               #avoids empty item
                    for x in range(1,len(line)): #skips  recipe number\
                        if line[x] in features:
                            distro[row_count][features[line[x]]] = 1#adds each ingredient to the correct column if it exists
                        try:
                            y[row_count] = line[0]#next to last column is cuisine
                        except:
                            print(line[1])
                            print (dicts.cuisine_dict[line[1]])
                            temp=raw_input('geh')  

                    row_count+=1
    #             if row_count %5 == 0:
    #                 print('row',row_count)
        datas.close()
    return distro, y

#main
#Both
ingredient = []

#creates the feature list
if FULL_DICT:
    features= dicts.ingredient_dict
else:
    features=generate_features()
print(features)


if SELECT_SAMPLE:
    Xtrain,Ytrain = build_data(features, ingredient, TRAIN_IN, TRAIN_OUT)  
    test_count = file_len(TEST_IN)
    print(test_count)
    raw_input('hey')   
    Ytest=np.zeros(test_count)
    Xtest=np.zeros((test_count,len(features)))
elif KAGGLE:
    print("you are here")
    Xtrain,Ytrain = build_data(features, ingredient, FULL_IN, FULL_OUT)  
    print(Xtrain[0])
    Xtest, Ytest = build_data(features, ingredient, 'kaggle.csv',KAGGLE_OUT, kaggle =KAGGLE)
    print(Xtest[0])
else:
    X,Y = build_data(features, ingredient, FULL_IN, FULL_OUT)  
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33, random_state=1)

    with open(TEST_IN,'r') as datas: #reads training data

            row_count=0
            while True:
                
                line=datas.readline()
                if not line: break
                line = formatting(line)
                if len(line)>2:               #avoids empty item
                        for x in range(2,len(line)): #skips cusine and recipe number\
                            if line[x] in features:
                                Xtest[row_count][features[line[x]]] = 1#adds each ingredient to the correct column if it exists
                            try:
                                Ytest[row_count] = dicts.cuisine_dict[line[1]]#next to last column is cuisine
                                #print(new_Ytest, new_Xtest)
    
                            except:
                                print(line[1])

                row_count+=1

                if(row_count%10==0):
                    print(row_count)

    
FULL_OUT.close()
TRAIN_OUT.close()
TEST_OUT.close()

#Xtest = np.vstack((Xtrain,Xtest))
#Ytest = np.hstack((Ytrain, Ytest))
#multinomial    
clf = MultinomialNB()
clf.fit(Xtrain, Ytrain)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

#print("Accuracy using the training data multinomial")
#print(clf.score(Xtrain, Ytrain))  
#print("Accuracy using test data multinomial")
print(clf.score(Xtest, Ytest))   
#print("Class probability with test data")    
#print(clf.predict_proba(Xtest))
#print("Class probability with test data")    
#print(clf.predict(Xtest))
#for item in Xtest:
#    print(item)
print len(Xtest)


        


#Bernoulli
clg = BernoulliNB()
clg.fit(Xtrain, Ytrain)
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
print("Accuracy using the training data Bernuli")
print(clg.score(Xtrain, Ytrain))  
print("Accuracy using test data Bernuli")
print(clg.score(Xtest, Ytest))   
#print("Class probability with test data")    
#print(clg.predict_proba(Xtest))
print("Class probability with test data")    
print(clg.predict(Xtest))

#Gaussian
# 
# 
# clh = GaussianNB()
# y_pred = clh.fit(Xtrain, Ytrain).predict(Xtest)
# print("Accuracy using test data Gaussian")
# print("Number of mislabeled points out of a total %d points : %d" % (Xtest.shape[0],(Ytest != y_pred).sum()))

# 
# #random forest
clj=RandomForestClassifier(n_estimators=10)
clj.fit(Xtrain, Ytrain)
# print("Class probability with test data")    
# print(clf.predict(Xtest))
# 
# 


if KAGGLE:
    #print( clf.predict(Xtest))
    build_string= 'id', 'cuisine'
    
    spamwriter = csv.writer(open('submission.txt','wb'), delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(build_string)
    for item in range(len(Xtest)):
       
        results = clj.predict(Xtest[item])
        for key,value in dicts.cuisine_dict.items():
            if int(results) ==int(value):
                build_string =str(int(Ytest[item])),str(key)
        spamwriter.writerow(build_string)



 #cross validation Score
#scores = cross_validation.cross_val_score(clf, Xtest, Ytest,cv=10)
#print("Cross Validation with Multinomial Score")  
#print(scores)
# damnit= open('40scores','wb')
# for item in scores:
#     damnit.write(item)
# damnit.close()
# 
# scores = cross_validation.cross_val_score(clg, Xtest, Ytest,cv=5)
# print("Cross Validation with Bernuli Score")  
# print(scores)
# 
# scores = cross_validation.cross_val_score(clh, Xtest, Ytest,cv=5)
# print("Cross Validation with Gaussian Score")  
# print(scores)
# 
# scores = cross_validation.cross_val_score(clj, Xtest, Ytest,cv=5)
# print("Cross Validation with Random Forest")  
# print(scores)

#accuracy
# predicted = cross_validation.cross_val_predict(clf, Xtest, Ytest, verbose = 1, cv=10)
# print("Cross Validation with Multinomial Accuracy")  
# print(metrics.accuracy_score(Ytest, predicted))



#predicted = cross_validation.cross_val_predict(clg, Xtest, Ytest, cv=10)
#print("Cross Validation with Bernuli Accuracy")  
#print(metrics.accuracy_score(Ytest, predicted))

#predicted = cross_validation.cross_val_predict(clh, Xtest, Ytest, cv=10)
#print("Cross Validation with Gaussian Accuracy")  
#print(metrics.accuracy_score(Ytest, predicted))

#predicted = cross_validation.cross_val_predict(clj, Xtest, Ytest, cv=10)
#print("Cross Validation with Random Forest Accuracy")  
#print(metrics.accuracy_score(Ytest, predicted))