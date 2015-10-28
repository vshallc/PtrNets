# PtrNets
An implementation of Pointer Networks [arXiv:1506.03134](http://arxiv.org/abs/1506.03134)

## Usage
To generate training data

    >> python misc/tsp.py tsp.pkl.gz
    
This will generate the training data, add the data path to ptrnets.py (or change the default path in data_util), then start training:

    >> python ptrnets.py
    


## References

* Oriol Vinyals, Meire Fortunato, Navdeep Jaitly,
  "[Pointer Networks](http://arxiv.org/abs/1506.03134)",
  *arXiv:1506.03134 [stat.ML]*.
