import sys
import gzip
import cPickle
import numpy
from scipy.spatial import ConvexHull


def generate_one_example(n_nodes):
    points = numpy.random.rand(n_nodes, 2)
    hull = ConvexHull(points)  # scipy.spatial.ConvexHull will generate points in CCW order
    v = hull.vertices
    # v = numpy.roll(v, -list(v).index(numpy.min(v)))  # start from the smallest indice
    return points, v + 1


def generate_examples(num, n_min, n_max):
    examples = []
    for i in range(num):
        n_nodes = numpy.random.randint(n_min, n_max + 1)
        nodes, res = generate_one_example(n_nodes)
        examples.append((nodes, res))
    return examples


if __name__ == '__main__':
    # planar convex hull
    e_train = generate_examples(1048576, 5, 50)
    e_valid = generate_examples(1000, 5, 50)
    e_test = generate_examples(1000, 50, 50)
    obj = (e_train, e_valid, e_test)
    saveto = sys.argv[1]
    if saveto.endswith('.gz'):
        f_saveto = gzip.open(saveto, 'wb')
    else:
        f_saveto = open(saveto, 'wb')
    cPickle.dump(obj, f_saveto, -1)
    f_saveto.close()
