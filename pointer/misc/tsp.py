import itertools
import numpy
import sys
import gzip
import cPickle


def length(x, y):
    return numpy.linalg.norm(numpy.asarray(x) - numpy.asarray(y))


# https://gist.github.com/mlalevic/6222750
def solve_tsp_dynamic(points):
    # calc all lengths
    all_distances = [[length(x, y) for y in points] for x in points]
    # initial value - just distance from 0 to every other point + keep the track of edges
    A = {(frozenset([0, idx + 1]), idx + 1): (dist, [0, idx + 1]) for idx, dist in enumerate(all_distances[0][1:])}
    cnt = len(points)
    for m in range(2, cnt):
        B = {}
        for S in [frozenset(C) | {0} for C in itertools.combinations(range(1, cnt), m)]:
            for j in S - {0}:
                # this will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
                B[(S, j)] = min([(A[(S - {j}, k)][0] + all_distances[k][j], A[(S - {j}, k)][1] + [j])
                                 for k in S if k != 0 and k != j])
        A = B
    res = min([(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
    rres = res[1]
    # rres.append(0)
    rres = numpy.asarray(rres) + 1
    return rres


def generate_one_example(n_nodes):
    nodes = numpy.random.rand(n_nodes, 2)
    res = solve_tsp_dynamic(nodes)
    return nodes, res


def generate_examples(num, n_min, n_max):
    examples = []
    for i in range(num):
        n_nodes = numpy.random.randint(n_min, n_max + 1)
        nodes, res = generate_one_example(n_nodes)
        examples.append((nodes, res))
    return examples


def generate_one_test_example(n_nodes):
    nodes = numpy.random.rand(n_nodes, 2)
    res = numpy.arange(n_nodes) + 1  # doesn't matter...
    return nodes, res


def generate_test_examples(num, n_min, n_max):
    examples = []
    for i in range(num):
        n_nodes = numpy.random.randint(n_min, n_max + 1)
        nodes, res = generate_one_test_example(n_nodes)
        examples.append((nodes, res))
    return examples


if __name__ == '__main__':
    # planar tsp
    # this can be very slow for large n_max
    e_train = generate_examples(1048576, 5, 10)
    e_valid = generate_examples(1000, 5, 10)
    e_test = generate_examples(1000, 10, 10)
    obj = (e_train, e_valid, e_test)

    # generating test examples only
    # obj = generate_test_examples(10000, 100, 100)

    saveto = sys.argv[1]
    if saveto.endswith('.gz'):
        f_saveto = gzip.open(saveto, 'wb')
    else:
        f_saveto = open(saveto, 'wb')
    cPickle.dump(obj, f_saveto, -1)
    f_saveto.close()

