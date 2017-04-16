import csv,os,re
import cv2, glob
import operator
from threading import Thread


source="C:\\Anju\\Final Year Project\\Stanford drone dataset\\newannotations"
trainSource="C:\\Anju\\Final Year Project\\Stanford drone dataset\\newannotations\\train"
frameTrainDestination="C:\\Anju\\Final Year Project\\Stanford drone dataset\\newannotations\\train\\frame"


def threadFunc(files):
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
        os.makedirs(frameTrainDestination+'\\'+name)
        readerIterator=reader.__iter__()
        sortedlist = sorted(readerIterator, key=operator.itemgetter(7))

        for item in sortedlist:
            file=open(frameTrainDestination+'\\'+name+'\\'+name+'_'+str(item[7])+'.csv','a',newline='')
            fileWriter=csv.writer(file)
            fileWriter.writerow(item)


files=glob.glob(trainSource+"\\"+'*.csv');
#files=["C:\\Anju\\Final Year Project\\Stanford drone dataset\\newannotations\\train\\test.csv"]

thread1 = Thread(target = threadFunc, args = (files[:15], ))
thread2 = Thread(target = threadFunc, args=(files[15:30],))
thread3 = Thread(target = threadFunc, args=(files[30:45],))
thread4 = Thread(target = threadFunc, args=(files[45:62],))
thread1.start()
thread2.start()
thread3.start()
thread4.start()

    

    

