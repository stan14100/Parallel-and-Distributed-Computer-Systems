"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import time

#in order to pass data to C
import ctypes

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        #constant seed for debugging purposes
        np.random.seed(7801);
        self.biases = [(np.random.randn(y, 1)).astype(np.float32) for y in sizes[1:]]
        self.weights = [(np.random.randn(y, x)).astype(np.float32)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        #make the one-dimensional so that they can pass to C easier
        self.weights1d_set = [self.weights[k].flatten() for k in range(self.num_layers-1)]
        self.biases1d_set = [self.biases[k].flatten() for k in range(self.num_layers-1)]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        i=0
        for b, w in zip(self.biases1d_set, self.weights1d_set):
            #b and w and now one dimensional array. must convert the to 2d in order np.dot to happer
            b = np.reshape(b, (self.sizes[i+1],1) )
            w = np.reshape(w, (self.sizes[i+1],self.sizes[i]) )
            a = sigmoid(np.dot(w, a)+b)
            i = i +1
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        #Stochastic gradient descent is used !
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        print "START parallel ! ! !"
        trdata_dim= len(training_data[0][0]);
        trres_dim= len(training_data[0][1]);
        #make eta a float32 whatever it is
        eta = np.float32(eta);
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):  #iterate gia oles tis epoxes
            start_time=time.time()
        #uncomment
    #    random.shuffle(training_data)  #shuffle the training data
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
        #    count = 0;
            for mini_batch in mini_batches:
        #        count = count + 1;
    #        mini_batch = training_data[0:mini_batch_size]
            #    print "Start measuring time"
            #    start_time=time.time()
                #PASS DATA TO C
                trdata, trres = zip(*mini_batch)
                trdata = np.array(trdata).flatten().astype(np.float32)
                trres = np.array(trres).flatten().astype(np.int)
                ctypes_weights = [np.ctypeslib.as_ctypes(weights1d) for weights1d in self.weights1d_set]
                ctypes_biases = [np.ctypeslib.as_ctypes(biases1d) for biases1d in self.biases1d_set]

                weights_pointer_ar = (ctypes.POINTER(ctypes.c_float)* (self.num_layers-1)) (*ctypes_weights)
                biases_pointer_ar = (ctypes.POINTER(ctypes.c_float)* (self.num_layers-1)) (*ctypes_biases)

                sizes_pointer = ctypes.c_int*self.num_layers
                trres_pointer = ctypes.c_int*(mini_batch_size*trres_dim)
                trdata_pointer = ctypes.c_float*(mini_batch_size*trdata_dim)

                #call the C function
                #gradient descent : update biases & weights
                ctypes.CDLL("concBP.so").parallelBP(ctypes.c_int(self.num_layers), weights_pointer_ar \
                ,biases_pointer_ar, sizes_pointer(*self.sizes), trdata_pointer(*trdata) \
                ,trres_pointer(*trres), ctypes.c_int(mini_batch_size) \
                ,ctypes.c_int(trdata_dim) , ctypes.c_int(trres_dim), ctypes.c_float(eta))
                #    print "Elapsed time :{0}".format(time.time()-start_time)
    #            if (count==2):
    #                print self.biases1d_set
            if test_data:
                elapsed_time = time.time()-start_time;
                print "Epoch {0}: {1} / {2}. Elapsed time : {3}. Average time : {4} ".format(
                    j, self.evaluate(test_data), n_test, elapsed_time, elapsed_time/(n/mini_batch_size) )
            #    print self.biases1d_set
            else:
                print "Epoch {0} complete".format(j)


    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

# biases and weights dimensions is len(sizes)-1
# weights1d = [weights[k].flatten() for k in range(len(size)-1)]
# biases1d = [biases[k].flatten() for k in range(len(size)-1)]
