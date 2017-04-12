output = open("rnnSingleOut.txt",'r')

threshold = 15
count = 0
TP = 0
for row in output:
    count += 1
    points = [ float(i) for i in row.split() ]
    #print points[0]+ threshold, points[2]
    if ( points[0]+threshold > points[2] and points[0]-threshold < points[2] ) and ( points[1]+threshold > points[3] and points[1]-threshold < points[3] ) :
        TP += 1

print TP, count
print float(TP)/count
#print rows