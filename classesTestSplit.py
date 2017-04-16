import csv, os, re
import cv2, glob
import operator

source = "C:\\Anju\\Final Year Project\\Stanford drone dataset\\newannotations"
testSource = "C:\\Anju\\Final Year Project\\Stanford drone dataset\\newannotations\\test"
classesTestDestination = "C:\\Anju\\Final Year Project\\Stanford drone dataset\\newannotations\\test\\classes"

# folders=['bookstore','coupa','deathCircle','gates','hyang','little','nexus','quad']
classes = ['Biker', 'Pedestrian', 'Cart', 'Skater', 'Car']

files = glob.glob(testSource + "\\" + '*.csv');
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

    bikerDestination = open(classesTestDestination + '\\' + name + 'Biker.csv', 'w', newline='')
    bikerWrite = csv.writer(bikerDestination)
    pedestrianDestination = open(classesTestDestination + "\\" + name + 'Pedestrian.csv', 'w', newline='')
    pedestrianWrite = csv.writer(pedestrianDestination)
    cartDestination = open(classesTestDestination + '\\' + name + 'Cart.csv', 'w', newline='')
    cartWrite = csv.writer(cartDestination)
    carDestination = open(classesTestDestination + '\\' + name + 'Car.csv', 'w', newline='')
    carWrite = csv.writer(carDestination)
    skaterDestination = open(classesTestDestination + '\\' + name + 'Skater.csv', 'w', newline='')
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
