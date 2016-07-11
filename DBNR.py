
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from mlp import HiddenLayer
from rbm import RBM
from linear_regression import LinearRegression

class DBNR(object):
    def __init__(self, numpy_rng, theano_rng=None, n_ins=100,
                 hidden_layers_size=None, n_outs=1, L1_reg=0.00,
                 L2_reg=0.0001):
        if hidden_layers_size is None:
            hidden_layers_size = [100, 100]

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_size)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 30))

        self.x = T.matrix('x')
        self.y = T.vector('y')

        for i in range(self.n_layers):
            if i == 0:
                input_sizes = n_ins
            else:
                input_sizes = hidden_layers_size[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng, input=layer_input,
                                        n_in=input_sizes, n_out=hidden_layers_size[i],
                                        activation=T.nnet.sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            rbm_layer = RBM(numpy_rng=numpy_rng, theano_rng=theano_rng,
                            input=layer_input, n_visible=input_sizes,
                            n_hidden=hidden_layers_size[i], W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        self.linearRegressionLayer = LinearRegression(input=self.sigmoid_layers[-1].output,
                                                      n_in=hidden_layers_size[-1],
                                                      n_out=n_outs)
        self.L1 = abs(self.sigmoid_layers[-1].W).sum() + abs(self.linearRegressionLayer.W).sum()
        self.L2_sqr = (self.sigmoid_layers[-1].W ** 2).sum() + (self.linearRegressionLayer.W ** 2).sum()
        self.squared_errors = self.linearRegressionLayer.squared_errors(self.y)
        self.finetune_cost = self.squared_errors + L1_reg * self.L1 + L2_reg * self.L2_sqr
        self.y_pred = self.linearRegressionLayer.p_y_given_x
        self.params = self.params + self.linearRegressionLayer.params

    def pretraining_function(self, train_set_x, batch_size, k):
        index = T.lscalar('index')
        learning_rate = T.scalar('lr')
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:
            cost, updates = rbm.get_cost_updates(learning_rate, persistent=None, k = k)
            fn = theano.function(
                inputs=[index, theano.In(learning_rate, value=0.1)],
                outputs=cost, updates=updates, givens={
                    self.x: train_set_x[batch_begin:batch_end]
                }
            )
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')

        gparams = T.grad(self.finetune_cost, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(
            inputs=[index], outputs=self.finetune_cost, updates=updates,
            givens={
                self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        test_score_i = theano.function(
            inputs=[index], outputs=self.squared_errors, givens={
                self.x: test_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        valid_score_i = theano.function(
            inputs=[index], outputs=self.squared_errors, givens={
                self.x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score
