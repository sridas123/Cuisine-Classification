import json
import csv
data=[]
with open("cuisine_data.csv",'ab') as f1:
     csv_writer = csv.writer(f1,delimiter=',',dialect = 'excel')
     header=["id","cuisine","ingredient"]
     csv_writer.writerows([header])
     
with open("D:/Grad Studies/Machine Learning/Project/train.json/train.json") as json_file:
     data=json.load(json_file)
     for i in range(0,len(data)):
         pdict={}
         pdict=data[i]
         cid=pdict["id"]
         cuisine=pdict["cuisine"]
         ingrd=[]
         ingrd=pdict["ingredients"]
         row=[]
         for i in range(0,len(ingrd)):
             text= ingrd[i].encode('ascii', 'ignore')+" "

         text=text.rstrip()    
         with open("cuisine_data.csv",'ab') as f1:
              row.extend([cid,cuisine,text])
              csv_writer = csv.writer(f1,delimiter=',',dialect = 'excel')
              csv_writer.writerows([row])
                      
