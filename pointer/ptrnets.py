from collections import OrderedDict
import cPickle as pkl
import sys
import time

import random
import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import data_utils

datasets = {'tsp': (data_utils.load_data, data_utils.prepare_data)}

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if minibatch_start != n:
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def get_dataset(name):
    return datasets[name][0], datasets[name][1]


def _p(pp, name):
    return '%s_%s' % (pp, name)


def _pd(pp, name, ix):
    return '%s_%s_%s' % (pp, name, ix)


def rand_weight(ndim, ddim, lo, hi):
    randn = numpy.random.rand(ndim, ddim)
    randn = randn * (hi - lo) + lo
    return randn.astype(config.floatX)


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def init_params(options):
    params = OrderedDict()

    # lstm gates parameters
    W = numpy.concatenate([rand_weight(options['data_dim'], options['dim_proj'], -0.08, 0.08),
                           rand_weight(options['data_dim'], options['dim_proj'], -0.08, 0.08),
                           rand_weight(options['data_dim'], options['dim_proj'], -0.08, 0.08),
                           rand_weight(options['data_dim'], options['dim_proj'], -0.08, 0.08)], axis=1)
    params['lstm_en_W'] = W
    U = numpy.concatenate([rand_weight(options['dim_proj'], options['dim_proj'], -0.08, 0.08),
                           rand_weight(options['dim_proj'], options['dim_proj'], -0.08, 0.08),
                           rand_weight(options['dim_proj'], options['dim_proj'], -0.08, 0.08),
                           rand_weight(options['dim_proj'], options['dim_proj'], -0.08, 0.08)], axis=1)
    params['lstm_en_U'] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params['lstm_en_b'] = b.astype(config.floatX)

    W = numpy.concatenate([rand_weight(options['data_dim'], options['dim_proj'], -0.08, 0.08),
                           rand_weight(options['data_dim'], options['dim_proj'], -0.08, 0.08),
                           rand_weight(options['data_dim'], options['dim_proj'], -0.08, 0.08),
                           rand_weight(options['data_dim'], options['dim_proj'], -0.08, 0.08)], axis=1)
    params['lstm_de_W'] = W
    U = numpy.concatenate([rand_weight(options['dim_proj'], options['dim_proj'], -0.08, 0.08),
                           rand_weight(options['dim_proj'], options['dim_proj'], -0.08, 0.08),
                           rand_weight(options['dim_proj'], options['dim_proj'], -0.08, 0.08),
                           rand_weight(options['dim_proj'], options['dim_proj'], -0.08, 0.08)], axis=1)
    params['lstm_de_U'] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params['lstm_de_b'] = b.astype(config.floatX)

    params['lstm_hterm'] = rand_weight(options['dim_proj'], 1, -0.08, 0.08)[:, 0]

    # ptr parameters
    params['ptr_W1'] = rand_weight(options['dim_proj'], options['dim_proj'], -0.08, 0.08)
    params['ptr_W2'] = rand_weight(options['dim_proj'], options['dim_proj'], -0.08, 0.08)
    params['ptr_v'] = rand_weight(options['dim_proj'], 1, -0.08, 0.08)[:, 0]

    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def sgd(lr, tparams, grads, x, x_mask, yin, yin_mask, yid, yid_mask, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, x_mask, yin, yin_mask, yid, yid_mask], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, p, p_mask, x, x_mask, y, y_mask, cost):
    zipped_grads = [theano.shared(q.get_value() * numpy_floatX(0.), name='%s_grad' % k)
                    for k, q in tparams.iteritems()]
    running_grads = [theano.shared(q.get_value() * numpy_floatX(0.), name='%s_rgrad' % k)
                     for k, q in tparams.iteritems()]
    running_grads2 = [theano.shared(q.get_value() * numpy_floatX(0.), name='%s_rgrad2' % k)
                      for k, q in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([p, p_mask, x, x_mask, y, y_mask], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(q.get_value() * numpy_floatX(0.), name='%s_updir' % k)
             for k, q in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(q, q + udn[1])
                for q, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def ptr_network(tparams, p, p_mask, x, x_mask, xi, xi_mask, hidi, celi, hids, options):
    n_sizes = p.shape[0]
    n_samples = p.shape[1] if p.ndim == 3 else 1
    n_steps = x.shape[0]
    beam_width = xi.shape[0]

    assert p_mask is not None
    assert x_mask is not None
    assert xi_mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        if _x.ndim == 2:
            return _x[:, n * dim:(n + 1) * dim]
        return _x[n * dim:(n + 1) * dim]

    def softmax(m_, x_):
        maxes = tensor.max(x_, axis=0, keepdims=True)
        e = tensor.exp(x_ - maxes)
        dist = e / tensor.sum(e * m_, axis=0)
        return dist

    def _lstm(m_, x_, h_, c_, prefix='lstm_en'):
        preact = tensor.dot(x_, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
        preact += tensor.dot(h_, tparams[_p(prefix, 'U')])

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_
        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    def _ptr_probs(xm_, x_, h_, c_, _, hprevs, hprevs_m):
        xemb = p[x_, tensor.arange(n_samples), :]  # n_samples * dim_proj
        h, c = _lstm(xm_, xemb, h_, c_, 'lstm_de')
        u = tensor.dot(hprevs, tparams['ptr_W1']) + tensor.dot(h, tparams['ptr_W2'])  # n_steps * n_samples * dim
        u = tensor.tanh(u)  # n_sizes * n_samples * dim_proj
        u = tensor.dot(u, tparams['ptr_v'])  # n_sizes * n_samples
        # prob = tensor.nnet.softmax(u.T).T  # n_sizes * n_samples
        prob = softmax(hprevs_m, u)
        return h, c, prob

    # encoding
    # we add a blank header to p and p_mask, so pointer to the blank header means pointing to the terminated mark
    # see data_utils.prepare_data for more details
    ones = tensor.ones((n_samples,), dtype=p.dtype)
    h0 = tensor.outer(ones, tparams['lstm_hterm'])  # n_samples * dim_proj; T.tile doesn't work on non-constant reps
    c0 = tensor.alloc(numpy_floatX(0.), n_samples, options['dim_proj'])
    rval, _ = theano.scan(_lstm,
                          sequences=[p_mask, p],
                          outputs_info=[h0, c0],
                          name='encoding',
                          n_steps=n_sizes)
    hiddens, cells = rval  # hiddens: n_sizes * n_samples * dim_proj
    # hiddens = tensor.concatenate([tensor.shape_padleft(h0), hiddens], axis=0)
    f_encode = theano.function([p_mask, p], hiddens)

    # decoding
    hiddens_mask = tensor.set_subtensor(p_mask[0, :], tensor.constant(1, dtype=config.floatX))
    # hiddens_mask = tensor.concatenate([tensor.ones((1, n_samples), dtype=config.floatX), p_mask], axis=0)
    rval, _ = theano.scan(_ptr_probs,
                          sequences=[x_mask, x],
                          outputs_info=[hiddens[-1],  # n_samples * dim_proj
                                        tensor.alloc(numpy_floatX(0.), n_samples, options['dim_proj']),  # cells[-1],
                                        tensor.alloc(numpy_floatX(0.), n_sizes, n_samples)],
                          non_sequences=[hiddens, hiddens_mask],
                          name='decoding',
                          n_steps=n_steps)
    preds = rval[2]
    f_decode = theano.function([p_mask, p, x_mask, x], preds)

    # generating
    # xi, vector
    # xi_mask, vector
    # hidi, matrix beam_width * dim_proj
    # celi matrix beam_width * dim_proj
    # hids, tensor3D
    # c0 = tensor.alloc(numpy_floatX(0.), beam_width, options['dim_proj'])
    u0 = tensor.alloc(numpy_floatX(0.), hidi.shape[0], beam_width)
    hiddeni, celli, probi = _ptr_probs(xi_mask, xi, hidi, celi, u0, hids, hiddens_mask)
    f_probi = theano.function(inputs=[xi_mask, xi, hidi, celi, hids, p_mask, p], outputs=[hiddeni, celli, probi])

    return preds, f_encode, f_decode, f_probi


def gen_model(p, p_mask, f_encode, f_probi, options):
    # p: n_sizes * n_samples * data_dim
    n_sizes = p.shape[0]
    n_samples = p.shape[1] if p.ndim == 3 else 1
    beam_width = n_sizes  # for beam search
    hprev = f_encode(p_mask, p)  # n_sizes * n_samples * data_dim
    c0 = numpy.zeros((n_samples, options['dim_proj']), dtype=config.floatX)
    xi = numpy.zeros((n_samples,), dtype='int64')
    # xi_mask = numpy.zeros((n_samples,), dtype=config.floatX)
    h, c, probi = f_probi(p_mask[0], xi, hprev[-1], c0, hprev, p_mask, p)  # probi n_sizes * n_samples
    route = -numpy.ones((beam_width, n_samples, n_sizes), dtype='int64')
    costi = -numpy.log(probi)
    idx = costi.argsort(axis=0)[:beam_width]  # beam_width * n_samples
    route[:, :, 0] = idx
    costs = costi[idx, numpy.arange(n_samples)]
    # tile to beam numbers
    hprev = numpy.tile(hprev[:, None, :, :], (1, beam_width, 1, 1))  # n_sizes * beam_width * n_samples * dim_proj
    h = numpy.tile(h[None, :, :], (beam_width, 1, 1))
    c = numpy.tile(c[None, :, :], (beam_width, 1, 1))
    probi = numpy.tile(probi[:, None, :], (1, beam_width, 1))
    # costs = numpy.tile(costs[:, None, :], (1, beam_width, 1))

    idr = numpy.tile(numpy.arange(n_sizes), (beam_width, 1)).T.flatten()
    idc = numpy.tile(numpy.arange(beam_width), (n_sizes, 1)).flatten()
    ids = numpy.tile(numpy.arange(n_samples)[None, :], (beam_width, 1))

    for i in range(1, n_sizes):
        for b in range(beam_width):
            # h: beam_width * n_sampels * dim_proj
            # c: beam_width * n_sampels * dim_proj
            # probi: n_sizes * beam_width * n_samples
            h[b], c[b], probi[:, b, :] = f_probi(p_mask[i], idx[b], h[b], c[b], hprev[:, b, :, :], p_mask, p)
            probi[:, b, :] *= p_mask[i]  # set unmasked to 0
            probi[:, b, :] += (1 - p_mask[i])  # then set to 1, since log(1) = 0 for calculating cost
        costi = -numpy.log(probi)  # costi: n_sizes * beam_width * n_samples
        costs = numpy.tile(costs[None, :, :], (n_sizes, 1, 1))  # duplicate costs x n_sizes
        costu = costi + costs
        # idb = numpy.outer(numpy.arange(beam_width),numpy.ones((i,))).astype('int64')
        # idbn = numpy.tile(idb[:,None,:], (1, n_samples, 1))
        idbn = numpy.tile(numpy.arange(beam_width)[:, None, None], (1, n_samples, i))
        idsn = numpy.tile(numpy.arange(n_samples)[None, :, None], (beam_width, 1, i))
        costu[route[:, :, :i], idbn, idsn] = numpy.inf
        idx = costu.reshape(n_sizes * beam_width, n_samples).argsort(axis=0)[:beam_width]  # duplication can be selected
        h = h[idc[idx], ids, :]
        c = c[idc[idx], ids, :]
        route = route[idc[idx], ids, :]
        route[:, :, i] = idr[idx]
        costi += costs
        costs = costi[idr[idx], idc[idx], ids]
        idx = idr[idx]
    costs /= numpy.tile((p_mask.sum(axis=0) + numpy.ones(p_mask[0].shape)), (beam_width, 1))
    # route: beam_width * n_samples * route
    # costs: beam_width * n_samples
    return route, costs


def tour_length(problem, route):
    n_sizes = route.shape[0] - 1
    n_from = problem[route[0]]
    length = 0.
    for i in range(1, n_sizes):
        n_to = problem[route[i]]
        length += numpy.linalg.norm(n_to - n_from)
        n_from = n_to
    n_to = problem[route[0]]
    length += numpy.linalg.norm(n_to - n_from)
    return length


def tour_efficiency(f_encode, f_probi, prepare_data, data, iterator, options):
    # routes = []
    # costs = []
    len_sum = 0
    for _, valid_index in iterator:
        tspv = [data[t] for t in valid_index]
        v, vm, vx, vxm, vy, vym = prepare_data(tspv)
        r, c = gen_model(v, vm, f_encode, f_probi, options)
        route = r[0]
        # routes.extend(route)
        for s in range(route.shape[0]):
            len_sum += tour_length(v[:, s, :], route[s])
    len_sum /= len(data)
    return len_sum


def build_model(tparams, options):
    # for training
    p = tensor.tensor3('p', dtype=config.floatX)  # Problems, n_sizes * n_samples * data_dim
    p_mask = tensor.matrix('p_mask', dtype=config.floatX)
    x = tensor.matrix('x', dtype='int64')  # n_steps * n_samples
    x_mask = tensor.matrix('x_mask', dtype=config.floatX)
    y = tensor.matrix('y', dtype='int64')  # n_steps * n_samples
    y_mask = tensor.matrix('y_mask', dtype=config.floatX)

    # for generation
    hidi = tensor.matrix('hidi', dtype=config.floatX)
    celi = tensor.matrix('celi', dtype=config.floatX)
    hids = tensor.tensor3('hids', dtype=config.floatX)
    xi = tensor.vector('xi', dtype='int64')
    xi_mask = tensor.vector('xi_mask', dtype=config.floatX)

    n_steps = x.shape[0]
    n_samples = x.shape[1]

    preds, f_encode, f_decode, f_probi = ptr_network(tparams, p, p_mask, x, x_mask, xi, xi_mask, hidi, celi, hids,
                                                     options)

    idx_steps = tensor.outer(tensor.arange(n_steps, dtype='int64'), tensor.ones((n_samples,), dtype='int64'))
    idx_samples = tensor.outer(tensor.ones((n_steps,), dtype='int64'), tensor.arange(n_samples, dtype='int64'))
    probs = preds[idx_steps, y, idx_samples]
    # probs *= y_mask
    off = 1e-8
    if probs.dtype == 'float16':
        off = 1e-6
    # probs += (1 - y_mask)  # change unmasked position to 1, since log(1) = 0
    probs += off
    # probs_printed = theano.printing.Print('this is probs')(probs)
    cost = -tensor.log(probs)
    cost *= y_mask
    cost = cost.sum(axis=0) / y_mask.sum(axis=0)
    cost = cost.mean()
    return p, p_mask, x, x_mask, y, y_mask, preds, cost, f_encode, f_decode, f_probi


def train_lstm(
        dim_proj=128,  # LSTM number of hidden units.
        max_steps=50,
        patience=10,  # Number of epoch to wait before early stop if no progress
        max_epochs=5000,  # The maximum number of epoch to run
        dispFreq=10,  # Display to stdout the training progress every N updates
        decay_c=0.,  # Weight decay for the classifier applied to the U weights.
        lrate=0.01,  # Learning rate for sgd (not used for adadelta and rmsprop)
        n_words=10000,  # Vocabulary size
        optimizer=rmsprop,
        # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
        # encoder='deep_lstm',
        depth=4,
        decoder_depth=1,
        saveto='ptr_model.npz',  # The best model will be saved there
        validFreq=370,  # Compute the validation error after this number of update.
        genFreq=100,
        saveFreq=1110,  # Save the parameters after every saveFreq updates
        maxlen=100,  # Sequence longer then this get ignored
        batch_size=16,  # The batch size during training.
        valid_batch_size=64,  # The batch size used for validation/test set.
        dataset='tsp',

        # Parameter for extra option
        noise_std=0.,
        use_dropout=False,  # if False slightly faster, but worst test error
        # This frequently need a bigger model.
        reload_model=None,  # Path to a saved model we want to start from.
        test_size=-1,  # If >0, we keep only this number of test example.
):
    model_options = locals().copy()
    load_data, prepare_data = get_dataset(dataset)

    print 'Loading data'
    train, valid, test = load_data()

    model_options['data_dim'] = train[0][0][0].shape[0]  # data_dim = 2, i.e (x,y)

    print 'Building model'
    params = init_params(model_options)

    if reload_model:
        load_params(reload_model, params)

    tparams = init_tparams(params)

    (p, p_mask, x, x_mask, y, y_mask, preds, cost, f_encode, f_decode, f_probi) = build_model(tparams, model_options)
    f_cost = theano.function([p, p_mask, x, x_mask, y, y_mask], cost, name='f_cost')

    grads = tensor.grad(theano.gradient.grad_clip(cost, -2.0, 2.0), wrt=tparams.values())
    # grads = tensor.grad(cost, wrt=tparams.values())
    f_grad = theano.function([p, p_mask, x, x_mask, y, y_mask], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads, p, p_mask, x, x_mask, y, y_mask, cost)

    # generation

    print 'Optimization'

    kf_valid = get_minibatches_idx(len(valid), valid_batch_size)
    kf_test = get_minibatches_idx(len(test), valid_batch_size)

    print "%d train examples" % len(train)
    print "%d valid examples" % len(valid)
    print "%d test examples" % len(test)

    history_tour = []
    best_p = None
    bad_counter = 0

    if validFreq == -1:
        validFreq = len(train) / batch_size
    if saveFreq == -1:
        saveFreq = len(train) / batch_size

    uidx = 0  # the number of update done
    eidx = 0
    estop = False
    start_time = time.time()

    a = numpy.asarray([[[0., 0., ]],
                       [[0.2, 0.5]],
                       [[0.8, 0.5]],
                       [[0.5, 0.9]],
                       [[0.2, 0.4]],
                       [[0.9, 0.1]]], dtype=config.floatX)
    b = numpy.asarray([[[0., 0., ]],
                       [[0.1, 0.1]],
                       [[0.7, 0.3]],
                       [[0.6, 0.6]],
                       [[0.3, 0.9]],
                       [[0.8, 0.8]]], dtype=config.floatX)
    am = numpy.asarray([[0.],
                        [1.],
                        [1.],
                        [1.],
                        [1.],
                        [1.]], dtype=config.floatX)

    try:
        for eidx in xrange(max_epochs):
            n_samples = 0

            kf = get_minibatches_idx(len(train), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                tsps = [train[t] for t in train_index]
                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                p, p_mask, x, x_mask, y, y_mask = prepare_data(tsps)
                n_samples += p.shape[1]

                cost = f_grad_shared(p, p_mask, x, x_mask, y, y_mask)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

                if numpy.mod(uidx, genFreq) == 0:
                    params = unzip(tparams)
                    route1, cost1 = gen_model(a, am, f_encode, f_probi, model_options)
                    route2, cost2 = gen_model(b, am, f_encode, f_probi, model_options)
                    print 'example a'
                    print route1
                    print cost1
                    print 'example b'
                    print route2
                    print cost2
                    sys.stdout.flush()

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print 'Saving...',

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_ppls=history_tour, **params)
                    pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print 'Done'

                if numpy.mod(uidx, validFreq) == 0:
                    # train_tour = tour_efficiency(f_encode, f_probi, prepare_data, train, kf, model_options)
                    valid_tour = tour_efficiency(f_encode, f_probi, prepare_data, valid, kf_valid, model_options)
                    test_tour = tour_efficiency(f_encode, f_probi, prepare_data, test, kf_test, model_options)

                    history_tour.append([valid_tour, test_tour])

                    if uidx == 0 or valid_tour < numpy.array(history_tour)[:, 0].min():
                        best_p = unzip(tparams)
                        bad_counter = 0

                    # print ('Train ', train_tour, 'Valid ', valid_tour, 'Test ', test_tour)
                    print ('Valid ', valid_tour, 'Test ', test_tour)

                    if len(history_tour) > patience and valid_tour >= numpy.array(history_tour)[-patience:, 0].min():
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

            print 'Seen %d samples' % n_samples
            sys.stdout.flush()

            if estop:
                break

    except KeyboardInterrupt:
        print "Training interrupted"

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    kf_train_sorted = get_minibatches_idx(len(train), batch_size)
    train_tour = tour_efficiency(f_encode, f_probi, prepare_data, train, kf_train_sorted, model_options)
    valid_tour = tour_efficiency(f_encode, f_probi, prepare_data, valid, kf_valid, model_options)
    test_tour = tour_efficiency(f_encode, f_probi, prepare_data, test, kf_test, model_options)

    print ('Train ', train_tour, 'Valid ', valid_tour, 'Test ', test_tour)
    if saveto:
        numpy.savez(saveto, train_tour=train_tour, valid_tour=valid_tour, test_tour=test_tour,
                    history_tour=history_tour,
                    **best_p)
    print 'The code run for %d epochs, with %f sec/epochs' % ((eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' % (end_time - start_time))

    return train_tour, valid_tour, test_tour


if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    train_lstm(
        max_epochs=10000,
        max_steps=10,
        patience=50,
        dim_proj=256,
        lrate=1.0,
        validFreq=1000,
        saveFreq=5000,
        dispFreq=100,
        genFreq=100,
        depth=2,
        batch_size=128,
        valid_batch_size=10,
        use_dropout=False,
        optimizer=rmsprop,
        saveto=None,
    )
