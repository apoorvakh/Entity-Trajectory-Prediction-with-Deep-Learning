import numpy as np
import sequences
from rnn import RNN
from gru import GRU
from optimizers import Adagrad
from util import gradCheck
import sys
import csv

"""
Minimal single layer GRU implementation.
"""
import numpy as np
import sys

class GRU(object):

  def __init__(self, in_size, out_size, hidden_size):
    """
    This class implements a GRU.
    """

    # TODO: go back to 0.01 initialization
    # TODO: use glorot initialization?
    # input weights
    #print "1"
    self.Wxc = np.random.randn(hidden_size, in_size)*1 # input to candidate
    self.Wxr = np.random.randn(hidden_size, in_size)*1 # input to reset
    self.Wxz = np.random.randn(hidden_size, in_size)*1 # input to interpolate

    # Recurrent weights
    #print "2"
    self.Rhc = np.random.randn(hidden_size, hidden_size)*1 # hidden to candidate
    self.Rhr = np.random.randn(hidden_size, hidden_size)*1 # hidden to reset
    self.Rhz = np.random.randn(hidden_size, hidden_size)*1 # hidden to interpolate

    # Weights from hidden layer to output layer
    #print "3"
    self.Why = np.random.randn(out_size, hidden_size)*1 # hidden to output

    # biases
    self.bc  = np.zeros((hidden_size, 1)) # bias for candidate
    self.br  = np.zeros((hidden_size, 1)) # bias for reset
    self.bz  = np.zeros((hidden_size, 1)) # bias for interpolate

    self.by  = np.zeros((out_size, 1)) # output bias

    self.weights = [   self.Wxc
                     , self.Wxr
                     , self.Wxz
                     , self.Rhc
                     , self.Rhr
                     , self.Rhz
                     , self.Why
                     , self.bc
                     , self.br
                     , self.bz
                     , self.by
                   ]

    # I used this for grad checking, but I should clean up
    self.names   = [   "Wxc"
                     , "Wxr"
                     , "Wxz"
                     , "Rhc"
                     , "Rhr"
                     , "Rhz"
                     , "Why"
                     , "bc"
                     , "br"
                     , "bz"
                     , "by"
                   ]


  def lossFun(self, inputs, targets):
    """
    Does a forward and backward pass on the network using (inputs, targets)
    inputs is a bit-vector of seq-length
    targets is a bit-vector of seq-length
    """
    #print "inputs : ",inputs
    xs, rbars, rs, zbars, zs, cbars, cs, hs, ys, ps = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

    #xs are inputs
    #hs are hiddens
    #ys are outputs
    #ps are the activation of last layer

    # This resets the hidden state after every new sequence
    # TODO: maybe we don't need to
    hs[-1] = np.zeros((self.Wxc.shape[0],1))
    loss = 0
    # forward pass, compute outputs, t indexes time
    for t in xrange(len(inputs)):

      # For every variable V, Vbar represents the pre-activation version
      # For every variable Q, Qnext represents that variable at time t+1
        # where t is understood from context

      # xs is the input vector at this time
      xs[t] = np.matrix(inputs[t]).T
      #print xs[t]
      #exit(0)
      # The r gate, which modulates how much signal from h[t-1] goes to the candidate
      rbars[t] = np.dot(self.Wxr, xs[t]) + np.dot(self.Rhr, hs[t-1]) + self.br
      rs[t] = 1 / (1 + np.exp(-rbars[t]))
      # TODO: use an already existing sigmoid function

      # The z gate, which interpolates between candidate and h[t-1] to compute h[t]
      zbars[t] = np.dot(self.Wxz, xs[t]) + np.dot(self.Rhz, hs[t-1]) + self.bz
      zs[t] = 1 / (1 + np.exp(-zbars[t]))

      # The candidate, which is computed and used as described above.
      cbars[t] = np.dot(self.Wxc, xs[t]) + np.dot(self.Rhc, np.multiply(rs[t] , hs[t-1])) + self.bc
      cs[t] = np.tanh(cbars[t])

      #TODO: get rid of this
      ones = np.ones_like(zs[t])

      # Compute new h by interpolating between candidate and old h
      hs[t] = np.multiply(cs[t],zs[t]) + np.multiply(hs[t-1],ones - zs[t])

      # pre-activation output <- current hidden, bias
    ys = np.dot(self.Why, hs[t]) + self.by

      # the outputs use a sigmoid nonlinearity
    #ps = 1 / (1 + np.exp(-ys))
    ps = ys

      #create a vector of all ones the size of our outpout
    one = np.ones_like(ps)
    #print "ys", ys
    #print "ps", ps
    #exit(0)
      # compute the vectorized cross-entropy loss
    a = np.multiply(targets[0].T , np.log(ps))
    b = np.multiply(one - targets[0].T, np.log(one-ps))
    loss -= (a + b)
    print a, b, loss, np.log(ps), ps

    # allocate space for the grads of loss with respect to the weights

    dWxc = np.zeros_like(self.Wxc)
    dWxr = np.zeros_like(self.Wxr)
    dWxz = np.zeros_like(self.Wxz)
    dRhc = np.zeros_like(self.Rhc)
    dRhr = np.zeros_like(self.Rhr)
    dRhz = np.zeros_like(self.Rhz)
    dWhy = np.zeros_like(self.Why)

    # allocate space for the grads of loss with respect to biases
    dbc = np.zeros_like(self.bc)
    dbr = np.zeros_like(self.br)
    dbz = np.zeros_like(self.bz)
    dby = np.zeros_like(self.by)

    # no error is received from beyond the end of the sequence
    dhnext = np.zeros_like(hs[0])
    drbarnext = np.zeros_like(rbars[0])
    dzbarnext = np.zeros_like(zbars[0])
    dcbarnext = np.zeros_like(cbars[0])
    zs[len(inputs)] = np.zeros_like(zs[0])
    rs[len(inputs)] = np.zeros_like(rs[0])

    # go backwards through time
    dy = np.copy(ps)
    dy -= targets[0].T # backprop into y
    print "t",targets, dy
    exit(0)
    dby += dy
    for t in reversed(xrange(len(inputs))):

      # For every variable X, dX represents dC/dX
      # For variables that influence C at multiple time steps,
        # such as the weights, the delta is a sum of deltas at multiple
        # time steps



      dWhy += np.dot(dy, hs[t].T)


      # h[t] influences the cost in 5 ways:

      # through the interpolation using z at t+1
      dha = np.multiply(dhnext, ones - zs[t+1])

      # through transformation by weights into rbar
      dhb = np.dot(self.Rhr.T,drbarnext)

      # through transformation by weights into zbar
      dhc = np.dot(self.Rhz.T,dzbarnext)

      # through transformation by weights into cbar
      dhd = np.multiply(rs[t+1],np.dot(self.Rhc.T,dcbarnext))

      # through the output layer at time t
      dhe = np.dot(self.Why.T,dy)

      dh = dha + dhb + dhc + dhd + dhe

      dc = np.multiply(dh,zs[t])

      #backprop through tanh
      dcbar = np.multiply(dc , ones - np.square(cs[t]))

      dr = np.multiply(hs[t-1],np.dot(self.Rhc.T,dcbar))
      dz = np.multiply( dh, (cs[t] - hs[t-1]) )

      # backprop through sigmoids
      drbar = np.multiply( dr , np.multiply( rs[t] , (ones - rs[t])) )
      dzbar = np.multiply( dz , np.multiply( zs[t] , (ones - zs[t])) )

      dWxr += np.dot(drbar, xs[t].T)
      dWxz += np.dot(dzbar, xs[t].T)
      dWxc += np.dot(dcbar, xs[t].T)

      dRhr += np.dot(drbar, hs[t-1].T)
      dRhz += np.dot(dzbar, hs[t-1].T)
      dRhc += np.dot(dcbar, np.multiply(rs[t],hs[t-1]).T)

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
             ]

    return loss, deltas, ps



data = open('tempdata.csv', 'r').read() # should be simple plain text file
inputlist = []
with open('tempdata.csv', 'r') as csvfile :
    data = csv.reader(csvfile)
    for i in data:
        inputlist.append([int(i[5]), int(i[6])])


# Uncomment to remove determinism
np.random.seed(0)

# Set to True to perform gradient checking
GRAD_CHECK = False

vec_size = 2
out_size = vec_size # Size of output bit vector at each time step
in_size = vec_size  # Input vector size, bigger because of start+stop bits
hidden_size = 8 # Size of hidden layer of neurons
learning_rate = 0.99

# An object that keeps the network state during training.
model = GRU(in_size, out_size, hidden_size)

# An object that keeps the optimizer state during training
optimizer = Adagrad(model.weights,learning_rate)

n = 0 # counts the number of sequences trained on

while n<2:
    print n
      # train on sequences of length from 1 to 4
    seq_length = np.random.randint(1,5)
    i, t = sequences.copy_sequence(seq_length, vec_size)
    inputs = np.matrix(i)
    targets = np.matrix(t)
    #print inputs.shape
    inputs = np.asmatrix(inputlist[n:n+25])
    targets = np.asmatrix(inputlist[n+25])
    #print inputs.shape
    print targets.shape
    epoch =0
    while epoch<100:
      # forward seq_length characters through the net and fetch gradient
      # deltas is a list of deltas oriented same as list of weights
      loss, deltas, outputs = model.lossFun(inputs, targets)
      epoch+=1
      optimizer.update_weights(model.weights, deltas)
    if n % 1 == 0:
        print 'iter N %d' % (n)
        print "inputs: "
        #print inputs
        print "target", targets
        print "outputs: "
        for k in outputs:
          print k.T
        print"inputs:"
        #print targets[0][0] - outputs[0]
        # calculate the BPC
        print "bpc:"
        # this is actually nats-per-char
        # if we count on the whole sequence it's unfair to
        print np.sum(loss) / ((seq_length*2 + 2) * vec_size)

        if GRAD_CHECK:
          # Check weights using finite differences
          check = gradCheck(model, deltas, inputs, targets, 1e-5, 1e-7)
          print "PASS DIFF CHECK?: ", check
          if not check:
            sys.exit(1)
    n += 1
