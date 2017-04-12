"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import csv
from scipy.spatial import distance
# data I/O
data = open('tempdata.csv', 'r').read() # should be simple plain text file
inputlist = []
width = 1424.0
height = 1088.0
with open('tempdata.csv', 'r') as csvfile :
    data = csv.reader(csvfile)
    for i in data:
        inputlist.append([int(i[5]), int(i[6])])

#print inputlist[:3]
#exit()

#chars = list(set(data))
#data_size, vocab_size = len(data), len(chars)
#print 'data has %d characters, %d unique.' % (data_size, vocab_size)

# hyperparameters
hidden_size = 32 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 0.1
vocab_size = 2


class RNN():

    def __init__(self):
        # model parameters
        self.Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
        self.Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
        self.bh = np.zeros((hidden_size, 1)) # hidden bias
        self.by = np.zeros((vocab_size, 1)) # output bias
    def sigmoid(self, x):
        #print x
        newx = np.zeros((hidden_size, 1))
        for i in xrange(len(x)):
            newx[i] = ( 1 / (1 + np.exp(-x[i])))
        #print "n",newx, type(newx),newx.shape
        return newx
    def dsigmoid(self, y):
        return y * (1.0 - y)
    def lossFun(self, inputs, targets, hprev):
      """
      inputs,targets are both list of integers.
      hprev is Hx1 array of initial hidden state
      returns the loss, gradients on model parameters, and last hidden state
      """
      xs, hs, ys, ps = {}, {}, {}, {}
      hs[-1] = np.copy(hprev)
      loss = 0
      # forward pass
      #print "i",inputs
      for t in xrange(len(inputs)):
        xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
        # 2x1   ::: [ x y ]
        #xs[t][inputs[t]] = 1
        xs[t][0] = inputs[t][0]
        xs[t][1] = inputs[t][1]
        #print xs[t], xs[t].shape
        hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state
        #hs[t] = self.sigmoid(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state
      #print hs[t], type(hs[t]), hs[t].shape
      #exit()
      ys = np.dot(self.Why, hs[t]) + self.by # unnormalized log probabilities for next chars
      #ps = np.exp(ys) / np.sum(np.exp(ys)) # probabilities for next chars
      #print "a"
      ps = np.zeros((vocab_size,1))
      #print "ys", ys
      #print " HS : ", hs
      #exit()
      #ps = np.copy([ [max(0, y)] for y in ys])
      ps = ys #np.copy([ y for y in ys])
      #print ps, targets
      #print type(ps), ps
      #print type(ps)
      # #print "\nPredicted value: ",ps, "Target: ",targets
      # loss is the euclidian distance
      #loss += -np.log(ps) # softmax (cross-entropy loss)
      loss = distance.euclidean(ps, targets)
      #print "loss",loss
      #exit()
      # backward pass: compute gradients going backwards
      #print xs[t],xs[t].shape
      dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
      dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
      dhnext = np.zeros_like(hs[0])

      # back prop
      dy = np.copy([[i] for i in targets])
      #print dy.shape, type(dy), dy
      dy = ( ps - dy )
      #print dy, dy.shape, ps, targets
      dWhy += np.dot(dy, hs[t].T)
      dby += dy
      for t in reversed(xrange(len(inputs))):
        #dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
        dh = np.dot(self.Why.T, dy) + dhnext # backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh
        #dhraw = self.dsigmoid(hs[t]) * dh # backprop through tanh nonlinearity
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(self.Whh.T, dhraw)
      for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
      return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1], ps

    def predict(self, inputs, targets, hprev):
      """
      inputs,targets are both list of integers.
      hprev is Hx1 array of initial hidden state
      returns the loss, gradients on model parameters, and last hidden state
      """
      xs, hs, ys, ps = {}, {}, {}, {}
      hs[-1] = np.copy(hprev)
      loss = 0
      for t in xrange(len(inputs)):
        xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
        xs[t][0] = inputs[t][0]
        xs[t][1] = inputs[t][1]
        hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state
      ys = np.dot(self.Why, hs[t]) + self.by # unnormalized log probabilities for next chars
      ps = np.zeros((vocab_size,1))

      ps = ys #np.copy([ y for y in ys])
      print "\nPredicted value: ",ps, "Target: ",targets#, "hs : ", hs
      loss = distance.euclidean(ps, targets)
      dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
      dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
      dhnext = np.zeros_like(hs[0])
      # back prop
      dy = np.copy([[i] for i in targets])
      dy = ( ps - dy )
      dWhy += np.dot(dy, hs[t].T)
      dby += dy
      for t in reversed(xrange(len(inputs))):
        #dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
        dh = np.dot(self.Why.T, dy) + dhnext # backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(self.Whh.T, dhraw)
      for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
      return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1],ps

rnnModel = RNN()

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(rnnModel.Wxh), np.zeros_like(rnnModel.Whh), np.zeros_like(rnnModel.Why)
mbh, mby = np.zeros_like(rnnModel.bh), np.zeros_like(rnnModel.by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

epoch = 0
hprev = np.zeros((hidden_size,1))
while epoch < 1:
    while n<=400:#len(inputlist)-26:
      # forward seq_length characters through the net and fetch gradient
      #hprev = np.zeros((hidden_size,1))


      loss, dWxh, dWhh, dWhy, dbh, dby, hprev,output = rnnModel.lossFun(inputlist[n:n+25], inputlist[n+25], hprev)
      smooth_loss = smooth_loss * 0.999 + loss * 0.001

      # perform parameter update with Adagrad
      for param, dparam, mem in zip([rnnModel.Wxh, rnnModel.Whh, rnnModel.Why, rnnModel.bh, rnnModel.by],
                                        [dWxh, dWhh, dWhy, dbh, dby],
                                        [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam #/ np.sqrt(mem + 1e-8) # adagrad update
        #epoch+=1
      output = output.flatten()
      #print "\nPredicted value: ",output[0]*width, output[1]*height, "Target: ",inputlist[n+25][0]*width, inputlist[n+25][1]*height, "hprev : ", hprev
      print "\nPredicted value: ",output[0], output[1], "Target: ",inputlist[n+25][0], inputlist[n+25][1]
      with open('rnnSingleOut.txt','a') as outfile:
        outfile.write(str(output[0])+" "+str(output[1])+" "+str(inputlist[n+25][0])+" "+str(inputlist[n+25][1])+"\n")
      if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss)#, "hprev : ",hprev # print progress
      p += seq_length # move data pointer
      n += 1 # iteration counter
    epoch+=1

t =0
newinputlist = inputlist[n:n+25]
print newinputlist
print " ***  TESTING *** "
#hprev = np.zeros((hidden_size,1))
while t<100:
  #print newinputlist[t:t+25]
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev,output = rnnModel.predict(newinputlist[t:t+25], inputlist[n+25], hprev)
  newinputlist.append(output.flatten().tolist())

  for param, dparam, mem in zip([rnnModel.Wxh, rnnModel.Whh, rnnModel.Why, rnnModel.bh, rnnModel.by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam #/ np.sqrt(mem + 1e-8) # adagrad update"""
  t+=1
  n+=1

