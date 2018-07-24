import csv
import json
labels={'brazilian':0,'british':1,'cajun_creole':2,'chinese':3,'filipino':4,'french':5,'greek':6,
        'indian':7,'irish':8,'italian':9,'jamaican':10,'japanese':11,'korean':12,'mexican':13,'moroccan':14,
        'russian':15,'southern_us':16,'spanish':17,'thai':18,'vietnamese':19}
featpath="D:/Grad Studies/Machine Learning/Project/Data/Ingredients/"
with open(featpath+"features1.csv",'rb') as featfile:
         freader=csv.reader(featfile,delimiter=',')
         header=[]
         for row in freader:
             header.append(row[0].strip(" "))
             
header.append("label")
header.append("id")
             
with open(featpath+"train.csv",'wb') as trainfile:
     csv_writer = csv.writer(trainfile,delimiter=',',dialect = 'excel')
     csv_writer.writerow(header)
     
with open("D:/Grad Studies/Machine Learning/Project/train.json/test.json") as json_file:
     data=json.load(json_file)
     print (len(data))
     for i in range(0,len(data)):
         pdict={}
         pdict=data[i]
         cid=pdict["id"]
         cuisine=pdict["cuisine"]
         ingrd=[]
         ingrd=pdict["ingredients"]
         row=[]
         on=1
         off=0
         feature=[]
         for i in range(0,243):
             feature.append(int(off))
             
         for j in range(0,len(ingrd)):
             text= ingrd[j].encode('ascii', 'ignore')+" "
             text=text.rstrip()    
             for k in range(0,len(header)):
                 if header[k]==text:
                    feature[k]=int(on)
                    break
                    
         feature.append(labels[cuisine])
         feature.append(cid)
         
         with open(featpath+"test.csv",'ab') as trainfile:
              csv_writer = csv.writer(trainfile,delimiter=',',dialect = 'excel')
              csv_writer.writerow(feature)
              
         