import numpy as np
import sequences
from optimizers import Adagrad
from util import gradCheck
import sys
import csv

"""
Minimal single layer GRU implementation.
"""
import numpy as np
import sys

# Uncomment to remove determinism
np.random.seed(0)

# Set to True to perform gradient checking
GRAD_CHECK = False

vec_size = 2
out_size = vec_size # Size of output bit vector at each time step
in_size = vec_size  # Input vector size, bigger because of start+stop bits
hidden_size = 8 # Size of hidden layer of neurons
learning_rate = 0.6

path = 'bookstore0train/bookstore0train/bookstore0train_'
outpath = 'gruoutput'

wtMatrices = {'Pedestrian': dict(), 'Car': dict(), 'Biker': dict(), 'Skater': dict(), 'Cart': dict()}

for c in wtMatrices:
    #print c
    d = dict()
    d['Wxc'] = np.random.randn(hidden_size, in_size)*1 # input to candidate
    d['Wxr'] = np.random.randn(hidden_size, in_size)*1 # input to reset
    d['Wxz'] = np.random.randn(hidden_size, in_size)*1 # input to interpolate

    # Recurrent weights
    #print "2"
    d['Rhc'] = np.random.randn(hidden_size, hidden_size)*1 # hidden to candidate
    d['Rhr'] = np.random.randn(hidden_size, hidden_size)*1 # hidden to reset
    d['Rhz'] = np.random.randn(hidden_size, hidden_size)*1 # hidden to interpolate

    # Weights from hidden layer to output layer
    #print "3"
    d['Why'] = np.random.randn(out_size, hidden_size)*1 # hidden to output

    # biases
    d['bc']  = np.zeros((hidden_size, 1)) # bias for candidate
    d['br']  = np.zeros((hidden_size, 1)) # bias for reset
    d['bz']  = np.zeros((hidden_size, 1)) # bias for interpolate
    d['by']  = np.zeros((out_size, 1)) # output bias

    wtMatrices[c]=d
    #print wtMatrices

class GRU(object):

  def __init__(self, in_size, out_size, hidden_size):
    """
    This class implements a GRU.
    """

    # TODO: go back to 0.01 initialization
    # TODO: use glorot initialization?

    self.Neighbours = []  # id numbers of neighbours
    self.hs = np.zeros((hidden_size, 1))
    self.Wd = np.zeros((1, 20, 20))
    self.mWd = np.zeros_like(self.Wd)

  def lossFunc(self, inputs, eClass, Ht, targets, frameNo, entityNo, hprev):
    """
    Does a forward and backward pass on the network using (inputs, targets)
    inputs is a bit-vector of seq-length
    targets is a bit-vector of seq-length
    """
    #print inputs, type(inputs)#, inputs.shape
    xs, rbars, rs, zbars, zs, cbars, cs, hs, ys, ps = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

    #xs are inputs
    #hs are hiddens
    #ys are outputs
    #ps are the activation of last layer

    # This resets the hidden state after every new sequence
    # TODO: maybe we don't need to
    #hs[-1] = np.zeros((self.Wxc.shape[0],1))
    loss = 0
    # forward pass, compute outputs, t indexes time
    for t in xrange(len(inputs)):

      # For every variable V, Vbar represents the pre-activation version
      # For every variable Q, Qnext represents that variable at time t+1
        # where t is understood from context

      # xs is the input vector at this time
      xs[t] = np.matrix(inputs[t]).T
      temp = np.tensordot( self.Wd, Ht)
      # print temp
      #print temp, temp.shape, self.hs, self.hs.shape
      self.hst1 = temp.T + hprev
      #print xs[t]
      #exit(0)
      # The r gate, which modulates how much signal from h[t-1] goes to the candidate
      rbars[t] = np.dot(wtMatrices[eClass]['Wxr'], xs[t]) + np.dot(wtMatrices[eClass]['Rhr'], self.hst1) + wtMatrices[eClass]['br']
      rs[t] = 1 / (1 + np.exp(-rbars[t]))
      # TODO: use an already existing sigmoid function

      # The z gate, which interpolates between candidate and h[t-1] to compute h[t]
      zbars[t] = np.dot(wtMatrices[eClass]['Wxz'], xs[t]) + np.dot(wtMatrices[eClass]['Rhz'], self.hst1) + wtMatrices[eClass]['bz']
      zs[t] = 1 / (1 + np.exp(-zbars[t]))

      # The candidate, which is computed and used as described above.
      cbars[t] = np.dot(wtMatrices[eClass]['Wxc'], xs[t]) + np.dot(wtMatrices[eClass]['Rhc'], np.multiply(rs[t] , self.hst1)) + wtMatrices[eClass]['bc']
      cs[t] = np.tanh(cbars[t])

      #TODO: get rid of this
      ones = np.ones_like(zs[t])

      # Compute new h by interpolating between candidate and old h
      self.hs = np.multiply(cs[t],zs[t]) + np.multiply(self.hst1,ones - zs[t])

      # pre-activation output <- current hidden, bias
    ys = np.dot(wtMatrices[eClass]['Why'], self.hs) + wtMatrices[eClass]['by']

      # the outputs use a sigmoid nonlinearity
    #ps = 1 / (1 + np.exp(-ys))
    ps = ys
    #print ps.tolist()[1][0],ps[0][0],ps[1][0]
    #exit(0)
    with open(outpath+'//frame//'+str(frameNo)+'.csv', 'a') as outfile:
        outfile.write(  str(entityNo) + ", " + str(ps.tolist()[0][0]) + ", " + str(ps.tolist()[1][0]) + ", " + str(targets.tolist()[0][0]) + ", " + str(targets.tolist()[0][1]) +"\n" )
    with open(outpath+'//entity//'+str(entityNo)+'.csv', 'a') as outfile:
        outfile.write(  str(frameNo) + ", " + str(ps[0][0]) + ", " + str(ps[1][0]) + ", " + str(targets.tolist()[0][0]) + ", " + str(targets.tolist()[0][1]) +"\n" )

      #create a vector of all ones the size of our outpout
    one = np.ones_like(ps)
    #print "ys", ys
    #print "ps", ps
    #exit(0)
      # compute the vectorized cros s-entropy loss
    a = np.multiply(targets[0].T , np.log(ps))
    b = np.multiply(one - targets[0].T, np.log(one-ps))
    loss -= (a + b)

    # allocate space for the grads of loss with respect to the weights

    dWxc = np.zeros_like(wtMatrices[eClass]['Wxc'])
    dWxr = np.zeros_like(wtMatrices[eClass]['Wxr'])
    dWxz = np.zeros_like(wtMatrices[eClass]['Wxz'])
    dRhc = np.zeros_like(wtMatrices[eClass]['Rhc'])
    dRhr = np.zeros_like(wtMatrices[eClass]['Rhr'])
    dRhz = np.zeros_like(wtMatrices[eClass]['Rhz'])
    dWhy = np.zeros_like(wtMatrices[eClass]['Why'])
    dWd = np.zeros_like(self.Wd)

    # allocate space for the grads of loss with respect to biases
    dbc = np.zeros_like(wtMatrices[eClass]['bc'])
    dbr = np.zeros_like(wtMatrices[eClass]['br'])
    dbz = np.zeros_like(wtMatrices[eClass]['bz'])
    dby = np.zeros_like(wtMatrices[eClass]['by'])

    # no error is received from beyond the end of the sequence
    dhnext = np.zeros_like(self.hs)
    drbarnext = np.zeros_like(rbars[0])
    dzbarnext = np.zeros_like(zbars[0])
    dcbarnext = np.zeros_like(cbars[0])
    zs[len(inputs)] = np.zeros_like(zs[0])
    rs[len(inputs)] = np.zeros_like(rs[0])

    # go backwards through time
    dy = np.copy(ps)
    #print dy,targets[0].T
    #exit(0)
    dy -= targets[0].T # backprop into y
    dby += dy
    for t in reversed(xrange(len(inputs))):

      # For every variable X, dX represents dC/dX
      # For variables that influence C at multiple time steps,
        # such as the weights, the delta is a sum of deltas at multiple
        # time steps
      dWhy += np.dot(dy, self.hs.T)
      # h[t] influences the cost in 5 ways:

      # through the interpolation using z at t+1
      dha = np.multiply(dhnext, ones - zs[t+1])

      # through transformation by weights into rbar
      dhb = np.dot(wtMatrices[eClass]['Rhr'].T,drbarnext)

      # through transformation by weights into zbar
      dhc = np.dot(wtMatrices[eClass]['Rhz'].T,dzbarnext)

      # through transformation by weights into cbar
      dhd = np.multiply(rs[t+1],np.dot(wtMatrices[eClass]['Rhc'].T,dcbarnext))

      # through the output layer at time t
      dhe = np.dot(wtMatrices[eClass]['Why'].T,dy)

      dh = dha + dhb + dhc + dhd + dhe
      #print dh, dh.shape
      #exit(0)

      dc = np.multiply(dh,zs[t])

      #backprop through tanh
      dcbar = np.multiply(dc , ones - np.square(cs[t]))

      dr = np.multiply(self.hst1,np.dot(wtMatrices[eClass]['Rhc'].T,dcbar))
      dz = np.multiply( dh, (cs[t] - self.hst1) )

      # backprop through sigmoids
      drbar = np.multiply( dr , np.multiply( rs[t] , (ones - rs[t])) )
      dzbar = np.multiply( dz , np.multiply( zs[t] , (ones - zs[t])) )

      dWxr += np.dot(drbar, xs[t].T)
      dWxz += np.dot(dzbar, xs[t].T)
      dWxc += np.dot(dcbar, xs[t].T)
      dWd += np.inner(dh.T, Ht)

      dRhr += np.dot(drbar, self.hst1.T)
      dRhz += np.dot(dzbar, self.hst1.T)
      dRhc += np.dot(dcbar, np.multiply(rs[t],self.hst1).T)

      dbr += drbar
      dbc += dcbar
      dbz += dzbar

      dhnext =    dh
      drbarnext = drbar
      dzbarnext = dzbar
      dcbarnext = dcbar

    deltas = [   dWxc
               , dWxr
               , dWxz
               , dRhc
               , dRhr
               , dRhz
               , dWhy
               , dbc
               , dbr
               , dbz
               , dby
               , dWd
             ]

    return loss, deltas, self.hs, ps

#############################################################
# Entity Dict - { 'entity' : RNN() }

entities = dict()

# opening framewise files
n = 600
output = None
done = False
hprev = np.zeros((hidden_size,1))

classes = ['Pedestrian', 'Car', 'Biker', 'Skater' , 'Cart' ]
weights = dict()
for c in classes:
    weights[c] = [wtMatrices[c]['Wxc'], wtMatrices[c]['Wxr'], wtMatrices[c]['Wxz'], wtMatrices[c]['Rhc'], wtMatrices[c]['Rhr'], wtMatrices[c]['Rhz'],
            wtMatrices[c]['Why'], wtMatrices[c]['bc'], wtMatrices[c]['br'], wtMatrices[c]['bz'], wtMatrices[c]['by'] ]

optimizer = dict()
for c in classes:
    t = weights[c]
    t.append(np.zeros((1, 20, 20)))
    optimizer[c]= Adagrad(t ,learning_rate)

# frame-wise
for f in xrange(n):
    print "Frame : ", f
    # Make sure every new entity has an RNN instance
    tempE = dict()
    with open(path + str(f) + '.csv', 'r') as csvfile:
        data = csv.reader(csvfile)
        for i in data:
            #print i[0]
            entities[i[0]]=entities.get(i[0], GRU(in_size, out_size, hidden_size))
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
                if f > 300:
                    train = False
                    #print "###################TESTING : #####################"
                epochs = 1
                for epoch in xrange(epochs):
                    global output
                    if i[0]==tempEN[e][0]:
                        target = [ int(i[5]), int(i[6]) ]
                        if train == True: # Training
                            loss, deltas, hprev, output =  entities[tempEN[e][0]].lossFunc([[int(tempEN[e][5]), int(tempEN[e][6])]], tempEN[e][-1], Ht, np.asmatrix(target), f+1, tempEN[e][0], hprev)
                            t = weights[tempEN[e][-1]]
                            t.append(entities[tempEN[e][0]].Wd)
                            optimizer[tempEN[e][-1]].update_weights(t, deltas)
                        else: # Testing
                            if done == False :
                                print "########################## TESTING : ################################"
                                #with open(outpath+i[0]+'.txt','a') as outfile:
                                #    outfile.write("\n########################## TESTING : ################################\n" )
                                done = True
                            loss, deltas, hprev, output =  entities[tempEN[e][0]].lossFunc([[output.tolist()[0][0], output.tolist()[1][0]]], tempEN[e][-1], Ht, np.asmatrix(target), f+1, i[0], hprev) # i[0]==tempEN[e][0]
                            t = weights[tempEN[e][-1]]
                            t.append(entities[tempEN[e][0]].Wd)
                            optimizer[tempEN[e][-1]].update_weights(t, deltas)

		#print "Predicted: ",output," Target:", [tempEN[e][5], tempEN[e][6]]

