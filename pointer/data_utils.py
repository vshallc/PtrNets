import cPickle
import gzip
import os
import sys

import numpy
import theano


def prepare_data(examples):
    lengths = [(len(s), len(t)) for s, t in examples]

    n_samples = len(examples)
    maxlens = numpy.max(lengths, axis=0) + (1, 1)  # (x_max, y_max)
    ndim = examples[0][0][0].shape[0]

    p = numpy.zeros((maxlens[0], n_samples, ndim)).astype(theano.config.floatX)
    p_mask = numpy.zeros((maxlens[0], n_samples)).astype(theano.config.floatX)

    x = numpy.zeros((maxlens[1], n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlens[1], n_samples)).astype(theano.config.floatX)

    y = numpy.zeros((maxlens[1], n_samples)).astype('int64')
    y_mask = numpy.zeros((maxlens[1], n_samples)).astype(theano.config.floatX)
    # we add an extra zeros header to p and p_mask
    # when pointing to the header, it means pointing to the "terminate signal"
    for idx, st in enumerate(examples):
        s, t = st
        p[1:lengths[idx][0]+1, idx, :] = s
        p_mask[1:lengths[idx][0]+1, idx] = 1

        x[1:lengths[idx][1]+1, idx] = t
        x_mask[1:lengths[idx][1]+1, idx] = 1

        y[:lengths[idx][1], idx] = t
        y_mask[:lengths[idx][1]+1, idx] = 1
    return p, p_mask, x, x_mask, y, y_mask


def load_data(path='tsp_test.pkl.gz'):
    data_dir, data_file = os.path.split(path)
    if data_dir == "" and not os.path.isfile(path):
        path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            path
        )

    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    train, valid, test = cPickle.load(f)
    f.close()
    return train, valid, test


if __name__ == '__main__':
    pass