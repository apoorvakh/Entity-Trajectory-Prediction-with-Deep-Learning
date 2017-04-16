import csv,os
import cv2, glob
import operator

video_stream = cv2.VideoCapture()
video_stream.open('C:\\video.mov')
if not video_stream.isOpened():
    raise RuntimeError( "Error when reading image file")
  
total_frame_count = video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
fps= video_stream.get(cv2.CAP_PROP_FPS)
print(total_frame_count, fps)

videos="C:\\Anju\\Final Year Project\\Stanford drone dataset\\videos"
source="C:\\Anju\\Final Year Project\\Stanford drone dataset\\newannotations"
destinationclasstrain="C:\\Anju\\Final Year Project\\Stanford drone dataset\\newannotations\\train"
destinationclasstest="C:\\Anju\\Final Year Project\\Stanford drone dataset\\newannotations\\test"
#destination="C:\\Anju\\Final Year Project\\Stanford drone dataset\\newannotations\\test"

folders=['bookstore','coupa','deathCircle','gates','hyang','little','nexus','quad']
classes=['Biker','Pedestrain','Car','Skater']

for file in folders:
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

        i+=1

    

