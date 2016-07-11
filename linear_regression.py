import numpy
import theano
from theano import tensor as T

class LinearRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                               name='W', borrow=True)
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)
        self.input = input
        self.p_y_given_x = T.dot(input, self.W) + self.b
        self.y_pred = self.p_y_given_x[:,0]
        self.params = [self.W, self.b]

    def squared_errors(self, y):
        return T.mean((self.y_pred - y) ** 2)