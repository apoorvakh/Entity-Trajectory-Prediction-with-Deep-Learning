import csv,os,re
import cv2, glob
import operator


classesSource="C:\\Anju\\Final Year Project\\Stanford drone dataset\\newannotations\\train\\classes"
entityClassDestination="C:\\Anju\\Final Year Project\\Stanford drone dataset\\newannotations\\train\\classes\\entities"


#folders=['bookstore','coupa','deathCircle','gates','hyang','little','nexus','quad']
classes=['Biker','Pedestrian','Cart','Skater','Car']


files=glob.glob(classesSource+"\\"+'*.csv');
#files=["C:\\Anju\\Final Year Project\\Stanford drone dataset\\newannotations\\train\\test.csv"]
for file in files:
    print(file)
    #x=file.split("\\")
    #file="\\\\".join(x)
    #print(file)
    s=open(file)
    reader=csv.reader(s)
    
    #find name
    y=re.match('(.*)\\\\(.*).csv',file)
    name=y.groups()[1]
    print(name)
   
    readerIterator=reader.__iter__()
    sortedlist = sorted(readerIterator, key=operator.itemgetter(0))
    entitynum=[]
    for s in sortedlist:
        '''if s[0] not in entitynum:
           entitynum.append(s[0])'''
        #print(entityClassDestination)
        entityfile=open(entityClassDestination+'\\'+name+s[0]+'.csv','a',newline='')
        #print(entityfile)
        entitywrite=csv.writer(entityfile)
        entitywrite.writerow(s)

    

