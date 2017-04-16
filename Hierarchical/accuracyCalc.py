from scipy.spatial import distance

outputpath = "rnnoutput/frame/"

threshold = 100
count = 0
TP = 0
TPe = 0

for i in xrange(0, 46):
    with open(outputpath+str(i+1)+".csv") as output:
        #print i+1
        for row in output:
            #print row.split(), type(row), row[0], type(row[0])
            #exit(0)
            count += 1
            points = [ float(i) for i in row.split(", ") ]
            #print points
            #exit(0)
            #print points[0]+ threshold, points[2]
            if ( points[1]+threshold > points[3] and points[1]-threshold < points[3] ) or ( points[2]+threshold > points[4] and points[2]-threshold < points[4] ) :
                TP += 1
            if ( distance.euclidean([points[1], points[2]], [points[3], points[4]]) <= threshold):
                TPe += 1

            #print points

    #exit(0)
print TP, count, TPe
print float(TP)/count, float(TPe)/count