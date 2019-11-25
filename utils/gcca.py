# Implementation based on Rastogi et al. (2015) Multiview LSA: Representation Learning via Generalized CCA
# adapted from Nina Poerner <poerner[AT]cis.uni-muenchen.de>

import numpy as np
import scipy.linalg as linalg

class GCCA:
    def __init__(self):
        pass

    def fit(self, views):
        dims = [view.shape[1] for view in views]
        concat = np.concatenate(views, axis = 1)
        self.mean = concat.mean(axis = 0)
        cov = np.cov(concat.T)

        mask = linalg.block_diag(*[np.ones((dim, dim), bool) for dim in dims])

        eigvals, eigvecs = linalg.eigh(cov * np.invert(mask), cov * mask)
        tol = abs(eigvals.real).max() * len(eigvals) * np.finfo(float).eps

        self.theta = eigvecs.real[:, np.greater(abs(eigvals.real), tol)]
    
    def transform(self, views):
        concat = np.concatenate(views, axis = 1)
        return (concat - self.mean).dot(self.theta)

    def transform_as_list(self, views):
        dims = [view.shape[1] for view in views]
        outputs = []

        slc = slice(0,0)
        for i, dim in enumerate(dims):
            slc = slice(slc.stop, slc.stop + dim)
            outputs.append((views[i] - self.mean[slc]).dot(self.theta[slc]))
        return outputs
