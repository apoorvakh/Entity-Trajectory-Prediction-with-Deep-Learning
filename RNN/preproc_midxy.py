# Input Pre-Processing scripts ::
#
# - to segregate classes
# - to separate train and test data *
#
# - to form x-mid and y-mid



# # Forming x-mid and y-mid
import csv
ann = open("F:\\ABCD\\PESIT\\Year4\\Sem8\\RNN\\annotations.csv","r")
new_ann = open("F:\\ABCD\\PESIT\\Year4\\Sem8\\RNN\\annotations_new.csv","wb")
reader = csv.reader(ann)
writer = csv.writer(new_ann)
for row in reader :
    #print type(row), row
    x_mid = (int(row[1]) + int(row[3])) / 2
    y_mid = (int(row[2]) + int(row[4])) / 2
    row.insert(5,x_mid)
    row.insert(6,y_mid)
    #print x_mid
    #print type(row), row
    writer.writerow(row)

ann.close()
new_ann.close()