import csv
path="C:\\Project\\Data"
with open(path+'\\train_big.csv','rb') as trainfile:
    trainreader=csv.reader(trainfile,delimiter=',')
    for row in trainreader:
        temp=[]
        for j in range(0,119):
            temp.append(row[j].strip(" "))
            temp[j]=temp[j].replace(" ", "")
            #print row[i]
        with open(path+'\\train_big1.csv','ab') as f1:
            csv_writer = csv.writer(f1,delimiter=',',dialect = 'excel')
            csv_writer.writerows([temp])
