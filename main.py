import numpy
import theano
import sys
import timeit
import csv
import math
import six.moves.cPickle as pickle
from DBNR import DBNR
from time import strftime
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score, mean_squared_error

def load_data(datasets=None, train_index=None, test_index=None):
    x = datasets[0]
    y = datasets[1]

    with open(x, 'r') as g:
        read = csv.reader(g)
        gen = numpy.array(list(read), dtype='float64')
        g.close()

    with open(y, 'r') as f:
        read = csv.reader(f)
        fen = numpy.array(list(read), dtype='float64').flatten()
        f.close()

    max_ = max(fen)
    min_ = min(fen)
    fen = (fen - min_) / (max_ - min_)
    '''
    x_train, x_test, y_train, y_test = train_test_split(
        gen, fen, test_size=0.1, random_state=42)'''

    x_train, x_test = gen[train_index], gen[test_index]
    y_train, y_test = fen[train_index], fen[test_index]

    train_set = x_train, y_train
    test_set = x_test, y_test
    validation_set = x_test, y_test

    def shared_dataset(data_xy):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
        return shared_x, shared_y

    train_set_x, train_set_y = shared_dataset(train_set)
    val_set_x, val_set_y = shared_dataset(validation_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    dataset = [(train_set_x, train_set_y), (val_set_x, val_set_y), (test_set_x, test_set_y)]

    return dataset, max_

def test_DBNR(fold=None, finetune_lr=0.001, pretraining_epochs=200, pretrain_lr=0.001, k=1,
              training_epochs=200, datasets=None, batch_size=1, n_ins=None,
              hidden_layers_sizes=None, n_outs=None, train_index=None,
              test_index=None, out_save=None, model_save=None):
    dataset, max_y = load_data(datasets, train_index=train_index,
                        test_index=test_index)
    train_set_x, train_set_y = dataset[0]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    numpy_rng = numpy.random.RandomState(123)
    print '... membangun model'

    dbnr = DBNR(numpy_rng=numpy_rng, n_ins=n_ins,
                hidden_layers_size=hidden_layers_sizes, n_outs=n_outs)

    ###################################
    '''     PRETRAINING THE MODEL   '''
    ###################################
    print '... setting pretraining function'
    pretraining_fns = dbnr.pretraining_function(train_set_x=train_set_x,
                                                batch_size=batch_size, k=k)

    print '... pre-training model'
    start_time = timeit.default_timer()
    for i in range(dbnr.n_layers):
        for epoch in range(pretraining_epochs):
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
            if epoch % 100 == 0:
                print 'Pre-training layer ke-%i epoch ke-%d, rata-rata cost = ' % (i, epoch)
                print numpy.mean(c)

    end_time =timeit.default_timer()
    print >> sys.stderr, ('Pretraining selesai dalam waktu %.2fm' % ((end_time - start_time) / 60))

    ###################################
    '''     FINETUNING THE MODEL    '''
    ###################################
    print '... setting finetuning functions'
    train_fn, validate_model, test_model = dbnr.build_finetune_functions(
        datasets=dataset, batch_size=batch_size, learning_rate=finetune_lr
    )

    print '... finetuning model'

    patience = 4 * n_train_batches
    patience_increase = 2.
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs): # and (not done_looping):
        epoch += 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print(
                    'epoch ke-%i, minibatch %i/%i, validation MSE sebesar %f '
                    % (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss)
                )

                if this_validation_loss < best_validation_loss:
                    if (
                        this_validation_loss < best_validation_loss * improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch ke-%i, minibatch %i/%i, test MSE dari '
                           'best model adalah = %f ') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score))

                    with open(model_save, 'wb') as f:
                        pickle.dump(dbnr, f)

            if patience <= iter:
                done_looping = True
                break

    end_time =timeit.default_timer()
    print(
        (
            'Optimalisasi selesai dengan score validasi terbaik = %f , '
            'didapat pada iterasi ke- %i, '
            'dengan performa test %f '
        ) % (best_validation_loss, best_iter + 1, test_score)
    )
    print >> sys.stderr, ('Finetuning selesai dalam waktu %.2fm' % ((end_time - start_time) / 60.))

    predict(dataset, out_save, model_save, fold, max_y)

def predict(data=None, out_saved=None, model_saved=None, fold=None, y_max=None):
    # load the saved model
    classifier = pickle.load(open(model_saved))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.x],
        outputs=classifier.y_pred)

    # We can test it on some examples from test test
    test_set_x, test_set_y = data[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x)
    y_model = []
    for i in xrange(len(predicted_values)):
        y_model.append(predicted_values[i][0])
    y_model = numpy.array(y_model) * y_max
    numpy.set_printoptions(precision=4)
    true_value = test_set_y.get_value() * y_max

    r_squared = r2_score(true_value, y_model)
    mean_a = numpy.mean(true_value)
    mean_b = numpy.mean(y_model)
    pears = (sum((true_value - mean_a) * (y_model - mean_b))) / math.sqrt((sum((true_value - mean_a) ** 2)) * (sum((y_model - mean_b) ** 2)))
    #r_square_adj = 1 - (((1 - r_squared) * (9 - 1)) / (9 - 1072 - 1))
    mse = mean_squared_error(true_value, y_model)
    orig_stdout = sys.stdout
    sys.stdout = open(out_saved, 'a+')
    print "=================================================================================="
    print "Fold ke-%i \n" % fold
    print("Nilai prediksi dari test set:")
    print y_model
    print "\nNilai sebenarnya dari test set:"
    print true_value
    print "\nMSE: %.3f" % mse
    print "\nR-Square: %.3f" % r_squared
    print "\nPearson's Correlation Coef: %.3f" % pears
    print "==================================================================================\n"
    sys.stdout.close()
    sys.stdout = orig_stdout
    print("Nilai prediksi dari test set:")
    print y_model
    print "\nNilai sebenarnya dari test set:"
    print true_value
    print "\nMSE: %.3f" % mse
    print "\nR-Square: %.3f" % r_squared
    print "\nPearson's Correlation Coef: %.3f" % pears


if __name__ == '__main__':
    data = ['Data/dataCorn_WW_flf_geno_nolabel_biner.csv', 'Data/dataCorn_WW_flf_feno_nolabel.csv']
    waktu_eksekusi = strftime("%d-%m-%Y(%H.%M)")
    filename_out = 'doc deepbeliefregression/output_FLF_WW_jagung-' + waktu_eksekusi + '.txt'
    filename_model = 'doc deepbeliefregression/best_modelDBR_FLF_WW_jagung' + waktu_eksekusi + '.pkl'
    kf = KFold(284, n_folds=10, shuffle=True)
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        test_DBNR(fold=fold, finetune_lr=0.001, pretraining_epochs=500,
                  pretrain_lr=0.001, k=1, training_epochs=1000,
                  datasets=data, batch_size=4, n_ins=1148,
                  hidden_layers_sizes=[512, 512, 512], n_outs=1,
                  train_index=train_index, test_index=test_index,
                  out_save=filename_out, model_save=filename_model)

