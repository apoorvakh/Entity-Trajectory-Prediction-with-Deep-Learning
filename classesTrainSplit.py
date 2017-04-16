import csv, os, re
import cv2, glob
import operator

source = "C:\\Anju\\Final Year Project\\Stanford drone dataset\\newannotations"
trainSource = "C:\\Anju\\Final Year Project\\Stanford drone dataset\\newannotations\\train"
classesTrainDestination = "C:\\Anju\\Final Year Project\\Stanford drone dataset\\newannotations\\train\\classes"

# folders=['bookstore','coupa','deathCircle','gates','hyang','little','nexus','quad']
classes = ['Biker', 'Pedestrian', 'Cart', 'Skater', 'Car']

files = glob.glob(trainSource + "\\" + '*.csv');
# files=["C:\\Anju\\Final Year Project\\Stanford drone dataset\\newannotations\\train\\test.csv"]
for file in files:
    print(file)
    # x=file.split("\\")
    # file="\\\\".join(x)
    # print(file)
    s = open(file)
    reader = csv.reader(s)

    # find name
    y = re.match('(.*)\\\\(.*).csv', file)
    name = y.groups()[1]

    bikerDestination = open(classesTrainDestination + '\\' + name + 'Biker.csv', 'w', newline='')
    bikerWrite = csv.writer(bikerDestination)
    pedestrianDestination = open(classesTrainDestination + "\\" + name + 'Pedestrian.csv', 'w', newline='')
    pedestrianWrite = csv.writer(pedestrianDestination)
    cartDestination = open(classesTrainDestination + '\\' + name + 'Cart.csv', 'w', newline='')
    cartWrite = csv.writer(cartDestination)
    carDestination = open(classesTrainDestination + '\\' + name + 'Car.csv', 'w', newline='')
    carWrite = csv.writer(carDestination)
    skaterDestination = open(classesTrainDestination + '\\' + name + 'Skater.csv', 'w', newline='')
    skaterWrite = csv.writer(skaterDestination)
    readerIterator = reader.__iter__()
    sortedlist = sorted(readerIterator, key=operator.itemgetter(11))
    bc = 0
    pc = 0
    cc = 0
    ccar = 0
    sc = 0
    e = 0
    for s in sortedlist:
        # print(s)
        if 'Biker' in s:
            bikerWrite.writerow(s)
            # print("B")
            bc += 1
        elif 'Pedestrian' in s:
            pedestrianWrite.writerow(s)
            # print("P")
            pc += 1
        elif 'Cart' in s:
            cartWrite.writerow(s)
            cc += 1
            # print("C")
        elif 'Car' in s:
            carWrite.writerow(s)
            ccar += 1
            # print("C")
        elif 'Skater' in s:
            skaterWrite.writerow(s)
            sc += 1
            # print("S")
        else:
            e += 1

    print(len(sortedlist), sc + cc + pc + bc + ccar, e, sc, ccar, cc, pc, bc)

'''for file in folders:
    num=0
    i=0
    for x,y,z in os.walk(videos+'\\'+file):
        num=len(y)
    
    files=glob.glob(source+"\\"+file+'*.csv')
    for f in files:
        x=f.split("\\")
        f="\\\\".join(x)
        print(f)
        #break
        video_stream.open(videos+'\\'+file+'\\video'+str(i)+'\\video.mov')
        if not video_stream.isOpened():
            raise RuntimeError( "Error when reading image file")      
        total_frame_count = video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
        trainframecount=0.75*total_frame_count
        #testframecount=total_frame_count-trainframecount
        s=open(f)
        #C:\Anju\Final Year Project\Stanford drone dataset\newannotations\bookstore0.csv
        
        reader=csv.reader(s)
        #for line in reader:
           #print(line)
        readerIterator=reader.__iter__()
        #sortedlist = sorted(readerIterator, key=operator.itemgetter(7))
        dtrain=open(destinationclasstrain+'\\'+file+str(i)+'train.csv','w',newline='')
        trainwrite=csv.writer(dtrain)
        dtest=open(destinationclasstest+'\\'+file+str(i)+'test.csv','w',newline='')
        testwrite=csv.writer(dtest)
        for item in reader:
            if int(item[7])<trainframecount:
                trainwrite.writerow(item)
            else:
                testwrite.writerow(item)
        dtrain.close()
        dtest.close()

        i+=1'''
