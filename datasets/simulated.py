import numpy as np
from utils import simulate_data
from sklearn.model_selection import KFold


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p, :], b[p]


def load_simulated_data(cfg):
    features, lbls = simulate_data(cfg['mu1'], cfg['sigma1'], cfg['mu2'], cfg['sigma2'])
    n_category = 2
    features, lbls = unison_shuffled_copies(features, lbls)

    idx_test = np.arange(len(lbls) * 4 // 5, len(lbls))
    idx_trains = []
    idx_vals = []
    cv = KFold(n_splits=5, shuffle=False)
    for idx_train, idx_val in cv.split(np.arange(0, len(lbls) * 4 // 5)):
        idx_trains += [idx_train]
        idx_vals += [idx_val]

    return features, lbls, idx_trains, idx_vals, idx_test, n_category
