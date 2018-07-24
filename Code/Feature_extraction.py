import csv
path="D:/Grad Studies/Machine Learning/Project/Data/Ingredients/"

cuisines=[["brazilian",467],["british",804],["cajun_creole",1546],["chinese",2673],["filipino",755],["french",2646],["greek",1175],["indian",3003],
          ["irish",667],["italian",7838],["jamaican",526],["japanese",1423],["korean",830],["mexican",6438],["moroccan",821],["russian",489],
          ["southern_us",4320],["spanish",989],["thai",1539],["vietnamese",825]]
          
print ("The total number of cuisines are",len(cuisines))
"""Initialize the lists to load csv data"""
cuisine={}

"""Reads the list of 100 most used ingredients and computes the fraction"""

for i in range(0,len(cuisines)):
    with open(path+cuisines[i][0] + ".csv",'rb') as dishfile:
         dreader=csv.reader(dishfile,delimiter=',')
         next(dreader)
         ing1=[]
         for row in dreader:
             fraction=0
             ing=[]
             ingr=row[0].strip(" \r")
             count=row[1].strip(" \n")
             fraction=float(count)/float(cuisines[i][1])
             fraction=fraction*100
             if fraction >= 5:
                ing.extend((ingr,count))
                ing1.append(ing)
         cuisine[cuisines[i][0]]=ing1
feature_list=[]        

for cuis, ingr in cuisine.items():
    for j in range(0,len(ingr)):
        count=0
        ingr1=ingr[j][0]
        if ingr1 not in feature_list:
           for cuis1, ingr2 in cuisine.items():
               if cuis1 not in cuis:
                  ingr_list=[]
                  for k in range(0,len(ingr2)):
                      ingr_list.append(ingr2[k][0])
                      
                  if ingr1 in ingr_list:
                     count=count+1; 
           
           if count < 16: 
              feature_list.append(ingr1) 

print("The total number of features is",len(feature_list))

for i in range(0,len(feature_list)):
    with open(path+'features1.csv','ab') as f1:
         csv_writer = csv.writer(f1,delimiter=',',dialect = 'excel')
         row=(feature_list[i])
         csv_writer.writerows([[row]])
                   
               
        