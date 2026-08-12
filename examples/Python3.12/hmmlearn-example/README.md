## Purpose: Demonstrates Hidden Markov Model training and inference using the hmmlearn library.

### Packages used:
hmmlearn, numpy, scipy, scikit-learn

### Functionality:

- Builds a Gaussian HMM with 2 hidden states (low / high emission).
- Samples a synthetic observation sequence from the model.
- Decodes the most likely hidden-state path using the Viterbi algorithm.
- Fits a new model on the sampled data and scores its log-likelihood.

### How to run the example:
```
chmod +x install_test_example.sh
./install_test_example.sh
```

### License:
It's covered under Apache 2.0 licenses
