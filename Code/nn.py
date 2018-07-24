path="D:\\Grad Studies\\Machine Learning\\Project\\Code"
text_file = open("Output.txt", "w")
for i in range(0,119):
    string="@attribute "+"'A"+str(i)+"' {1,0}"+"\n"
    if i==118:
       string= "@attribute "+"'class'"+" {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}"+"\n"
    text_file.write(string)
text_file.close()
   
       