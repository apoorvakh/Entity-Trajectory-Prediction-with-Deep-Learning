import numpy as np
import csv
from scipy.spatial import distance

# hyperparameters
hidden_size = 32  # size of hidden layer of neurons
seq_length = 25  # number of steps to unroll the RNN for
learning_rate = 0.6
vocab_size = 2

# path = '/home/dhanya/Downloads/8th sem/newannotations/bookstore0train/bookstore0train_'
path = 'bookstore0train/bookstore0train/bookstore0train_'
outpath = 'rnnoutput/'

# Five set of wt matrices for each class

wtMatrices = {'Pedestrian': dict(), 'Car': dict(), 'Biker': dict(), 'Skater': dict(), 'Cart': dict()}

for c in wtMatrices:
    #print c
    d = dict()
    d['Wxh'] = np.random.randn(hidden_size, vocab_size) * 0.3  # input to hidden
    d['Whh'] = np.random.randn(hidden_size, hidden_size) * 0.3  # hidden to hidden
    d['Why'] = np.random.randn(vocab_size, hidden_size) * 0.3  # hidden to output
    d['bh'] = np.random.randn(hidden_size, 1) * 0.3  # hidden bias
    d['by'] = np.random.randn(vocab_size, 1) * 0.3  # output bias
    d['mWxh'], d['mWhh'], d['mWhy'] = np.zeros_like(d['Wxh']), np.zeros_like(d['Whh']), np.zeros_like(d['Why'])
    d['mbh'], d['mby'] = np.zeros_like(d['bh']), np.zeros_like(d['by'])
    wtMatrices[c]=d
    #print wtMatrices


# The RNN class - each entity has an instance

class RNN():
    def __init__(self):
        # model parameters
        self.Neighbours = []  # id numbers of neighbours
        self.hs = np.zeros((hidden_size, 1))
        self.Wd = np.zeros((1, 20, 20))
        self.mWd = np.zeros_like(self.Wd)

    def lossFunc(self, inputs, eClass, Ht, targets, frameNo, entityNo, hprev):
        xs, ys, ps = {}, {}, {}
        #self.hs[-1] = np.copy(hprev)
        loss = 0
        # forward pass
        #print "i",inputs, "\t", targets
        for t in xrange(len(inputs)):
            xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
            # xs[t][inputs[t]] = 1
            xs[t][0] = inputs[t][0]
            xs[t][1] = inputs[t][1]
            # print xs[t], xs[t].shape

            # Calculate hs from Ht
            # ## (1x20x20).(20x 20xh)
            #print np.compress(self.Wd, Ht, 0)
            temp = np.tensordot( self.Wd, Ht)
            # print temp
            #print temp, temp.shape, self.hs, self.hs.shape
            self.hst1 = temp.T + hprev
            #print self.hst1, hprev, self.hst1+hprev
            #exit(0)
            self.hs = np.tanh(np.dot(wtMatrices[eClass]['Wxh'], xs[t]) + np.dot(wtMatrices[eClass]['Whh'], self.hst1) + wtMatrices[eClass]['bh'])
            # print self.hs.shape,"here"
            #print xs[t], np.dot(wtMatrices[eClass]['Wxh'], xs[t])
            #print self.hs, self.hs.shape
            #print wtMatrices[eClass]['Whh'], wtMatrices[eClass]['Whh'].shape
            #print wtMatrices[eClass]['bh']
            #self.hs = np.tanh(np.dot(wtMatrices[eClass]['Wxh'], xs[t]) + np.dot(wtMatrices[eClass]['Whh'], self.hs) + wtMatrices[eClass]['bh']) # hidden state # we made from hs(t-1) to hs
        ys = np.dot(wtMatrices[eClass]['Why'], self.hs) + wtMatrices[eClass]['by'] # unnormalized log probabilities for next chars
        # ps = np.exp(ys) / np.sum(np.exp(ys)) # probabilities for next chars
        #ps = np.zeros((vocab_size, 1))
        # print "ys", ys

        # exit()
        # ps = np.copy([ [max(0, y)] for y in ys])
        ps = ys  # np.copy([ y for y in ys])
        # print ps, targets
        # print type(ps), ps
        # print type(ps)
        #print "\nPredicted value: ", [ps[0][0], ps[1][0]], "    Target: ", targets
        with open(outpath+'//frame//'+str(frameNo)+'.csv', 'a') as outfile:
            outfile.write(  str(entityNo) + ", " + str(ps[0][0]) + ", " + str(ps[1][0]) + ", " + str(targets[0]) + ", " + str(targets[1]) +"\n" )
        with open(outpath+'//entity//'+str(entityNo)+'.csv', 'a') as outfile:
            outfile.write(  str(frameNo) + ", " + str(ps[0][0]) + ", " + str(ps[1][0]) + ", " + str(targets[0]) + ", " + str(targets[1]) +"\n" )

        # loss is the euclidian distance
        # loss += -np.log(ps) # softmax (cross-entropy loss)
        #loss = distance.euclidean(ps, targets)
        # print "loss",loss
        # exit()
        # backward pass: compute gradients going backwards
        # print xs[t],xs[t].shape
        dWxh, dWhh, dWhy, dWd = np.zeros_like(wtMatrices[eClass]['Wxh']), np.zeros_like(wtMatrices[eClass]['Whh']), np.zeros_like(wtMatrices[eClass]['Why']), np.zeros_like(self.Wd)
        dbh, dby = np.zeros_like(wtMatrices[eClass]['bh']), np.zeros_like(wtMatrices[eClass]['by']) # can be put out
        dhnext = np.zeros_like(self.hs)
        #print self.hs, self.hs[0], dhnext

        # back prop
        dy = np.copy([[i] for i in targets])
        # print dy.shape, type(dy), dy
        dy = (ps - dy)
        # print dy, dy.shape, ps, targets
        dWhy += np.dot(dy, self.hs.T)
        dby += dy
        for t in reversed(xrange(len(inputs))):
            # dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
            dh = np.dot(wtMatrices[eClass]['Why'].T, dy) + dhnext  # backprop into h
            #print dh
            #exit(0)
            dhraw = (1 - self.hs * self.hs) * dh  # backprop through tanh nonlinearity
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, self.hs.T) ## was hs(t-1)
            #print dhraw.shape, Ht.shape
            #exit(0)
            dWd += np.inner(dhraw.T, Ht)
            dhnext = np.dot(wtMatrices[eClass]['Whh'].T, dhraw)
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
        return loss, dWxh, dWhh, dWhy, dWd, dbh, dby, self.hs, ps


# Entity Dict - { 'entity' : RNN() }
entities = dict()

# opening framewise files
n = 600
output = None
done = False
hprev = np.zeros((hidden_size,1))
# frame-wise
for f in xrange(n):
    print "Frame : ", f
    # Make sure every new entity has an RNN instance
    tempE = dict()
    with open(path + str(f) + '.csv', 'r') as csvfile:
        data = csv.reader(csvfile)
        for i in data:
            #print i[0]
            entities[i[0]]=entities.get(i[0], RNN())
            tempE[i[0]] = i
            # Print frame no. for an entity
            #with open(outpath+str(f+1)+'.txt','a') as outfile:
                #outfile.write( str(f) )
            # inputlist.append([int(i[5]), int(i[6])])
    #print tempE
    #print entities
    # For this frame(time t) - call lossFunc of RNN for ALL entities in frame t
    # Find neighbours N for all enities
    tempEN = tempE.values()
    #print tempEN
    for e in xrange(len(tempEN) - 1):
        for n in xrange(e + 1, len(tempEN)):
            # If inside 200x200 box
            if (int(tempEN[e][5]) + 100 >= int(tempEN[n][5])) or (int(tempEN[e][5]) - 100 <= int(tempEN[n][5])) or (
                            int(tempEN[e][6]) + 100 >= int(tempEN[n][6])) or (int(tempEN[e][6]) - 100 <= int(tempEN[n][6])):
                entities[tempEN[e][0]].Neighbours.append(tempEN[n][0])
                entities[tempEN[n][0]].Neighbours.append(tempEN[e][0])

    # Calculate Ht - do pooling of 200x200 into 10x10
    for e in xrange(len(tempEN)):
        Ht = np.zeros((20, 20, hidden_size))
        # Do calculation here!!!
        ex = int(tempEN[e][5])
        ey = int(tempEN[e][6])
        # pool to 10x10
        y = ey - 100
        # filling Ht with neighbours hs
        for n in entities[tempEN[e][0]].Neighbours:
            i = 0
            j = 0
            for x in xrange(ex - 100, ex + 100, 10):
                for y in xrange(ey - 100, ey + 100, 10):
                    #print i,j,x,y
                    if tempE[n][5]>=x and tempE[n][5]<=x+10 and tempE[n][6]>=y and tempE[n][6]<=y+10 :
                        Ht[i][j] += reduce(lambda x, y: x + y, entities[n].hs.tolist())             # Hierarchical weight induction !!!
                    i += 1
                j += 1
        #  open next frame to get target and call loss function
        #print "Entity : ", tempEN[e][0]
        with open(path + str(f+1) + '.csv', 'r') as csvfile:
            data = csv.reader(csvfile)
            for i in data:
                # print i[0]=
                train = True
                if f > 70:
                    train = False
                    #print "###################TESTING : #####################"
                epochs = 1
                for epoch in xrange(epochs):
                    global output
                    if i[0]==tempEN[e][0]:
                        target = [ int(i[5]), int(i[6])]
                        if train == True: # Training
                            loss, dWxh, dWhh, dWhy, dWd, dbh, dby, hprev, output =  entities[tempEN[e][0]].lossFunc([[int(tempEN[e][5]), int(tempEN[e][6])]], tempEN[e][-1], Ht, target, f+1, tempEN[e][0], hprev)

                            for param, dparam, mem in zip([wtMatrices[tempEN[e][-1]]['Wxh'], wtMatrices[tempEN[e][-1]]['Whh'], wtMatrices[tempEN[e][-1]]['Why'], entities[tempEN[e][0]].Wd, wtMatrices[tempEN[e][-1]]['bh'], wtMatrices[tempEN[e][-1]]['by']],
                                                          [dWxh, dWhh, dWhy,dWd, dbh, dby],
                                                          [wtMatrices[tempEN[e][-1]]['mWxh'], wtMatrices[tempEN[e][-1]]['mWhh'], wtMatrices[tempEN[e][-1]]['mWhy'], entities[tempEN[e][0]].mWd, wtMatrices[tempEN[e][-1]]['mbh'], wtMatrices[tempEN[e][-1]]['mby']]):
                                mem += dparam * dparam
                                param += -learning_rate * dparam #/ np.sqrt(mem + 1e-8)  # adagrad update
                        else: # Testing
                            if done == False :
                                print "########################## TESTING : ################################"
                                #with open(outpath+i[0]+'.txt','a') as outfile:
                                #    outfile.write("\n########################## TESTING : ################################\n" )
                                done = True
                            loss, dWxh, dWhh, dWhy, dWd, dbh, dby, hprev, output =  entities[tempEN[e][0]].lossFunc([[output[0][0], output[1][0]]], tempEN[e][-1], Ht, target, f+1, i[0], hprev) # i[0]==tempEN[e][0]

                            for param, dparam, mem in zip([wtMatrices[tempEN[e][-1]]['Wxh'], wtMatrices[tempEN[e][-1]]['Whh'], wtMatrices[tempEN[e][-1]]['Why'], entities[tempEN[e][0]].Wd, wtMatrices[tempEN[e][-1]]['bh'], wtMatrices[tempEN[e][-1]]['by']],
                                                          [dWxh, dWhh, dWhy,dWd, dbh, dby],
                                                          [wtMatrices[tempEN[e][-1]]['mWxh'], wtMatrices[tempEN[e][-1]]['mWhh'], wtMatrices[tempEN[e][-1]]['mWhy'], entities[tempEN[e][0]].mWd, wtMatrices[tempEN[e][-1]]['mbh'], wtMatrices[tempEN[e][-1]]['mby']]):
                                mem += dparam * dparam
                                param += -learning_rate * dparam #/ np.sqrt(mem + 1e-8)  # adagrad update

		#print "Predicted: ",output," Target:", [tempEN[e][5], tempEN[e][6]]




