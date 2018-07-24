import dictionary as dicts
import numpy as np
import unicodedata
import random
FEATURES_IN = 'features.csv' #list of chosen features 
FEATURES_OUT = 'readable.features.out'# list of features and associated column
TRAIN_IN = 'train_data' #1 recipe per list unmodified
TRAIN_OUT = 'train_data.out' # 100 column of ingredients followed by cuisine, and recipe number
INGREDIENT_COUNT = 118
TRAIN_COUNT = 1000



def generate_features():
    feature_dict = {} #dictionary of features value is array location key is ingredient name
    local_count = 0 #used to create array locations 
    fileout = open(FEATURES_OUT,'w')
    datas= open(FEATURES_IN,'r') 
    line = datas.read()
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

distro = np.zeros((TRAIN_COUNT, INGREDIENT_COUNT+2))
ingredient = []
features=generate_features()
#with open('result.csv','r',encoding='utf-8') as datas:
with open(TRAIN_IN,'r') as datas: #reads training data
    row_count=0
    while True:
        line=datas.readline()
        if not line: break
        line = line.replace(", ",' ')
        line = line.replace("'",'')
        line = line.replace('\n','')
        line = line.replace('"','')
        start =line.find( '(' )
        end = line.find( ')' )
        if start != -1 and end != -1:
            line = line[start+1:end]
        line = line.split(',')
        if len(line)>2:               #avoids empty item
            for x in range(2,len(line)): #skips cusine and recipe number
                if line[x] in features:
                    distro[row_count][features[line[x]]] = 1#adds each ingredient to the correct column if it exists
            try:
                distro[row_count][INGREDIENT_COUNT] = dicts.cuisine_dict[line[1]]#next to last column is cuisine
                distro[row_count][INGREDIENT_COUNT+1] = line[0] # makes the last item in the array the recipie #
            except:
                print(line[1])
                print (dicts.cuisine_dict[line[1]])
            
    datas.close()
    np.savetxt(TRAIN_OUT, distro, delimiter=',') #creates the output matrix
       
        