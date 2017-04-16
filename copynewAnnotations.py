import shutil,os

source="C:\\Anju\\Final Year Project\\Stanford drone dataset\\annotations"
destination="C:\\Anju\\Final Year Project\\Stanford drone dataset\\newannotations"
#ann = open("C:\\Anju\\Final/ Year/ Project\\Stanford drone dataset\\annotations","r")
#new_ann = open("C:\\Anju\\Final/ Year/ Project\\Stanford drone dataset\\annotations","w")
#reader = csv.reader(ann)
folders=['bookstore','coupa','deathCircle','gates','hyang','little','nexus','quad']

for folder in folders:
    num=0
    for x,y,z in os.walk(source+'\\'+folder):
        num=len(y)
        break
    for video in range(0,num):
        print(folder,video)
        shutil.copy2(source+'\\'+folder+'\\video'+str(video)+'\\'+folder+str(video)+'.csv',destination)
        
