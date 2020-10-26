"""
Interpretable multi-timescale models for predicting fMRI responses to continuous natural speech
NeurIPS`2020

Training interpolation weights $a_i$.

Author: Shailee Jain
"""
import numpy as np
from sklearn.linear_model import Ridge
import scipy
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import time


# Gaussian RBF kernel: e^(-(eps*distance)^2)
phi = lambda t, tau, epsilon: np.exp(-(epsilon * (t[None] - tau[:,None])) ** 2)

def interpolation_function(word_time, fine_time, vectors, i):
    P = phi(word_time, word_time, best_eps[i])
    r = Ridge(alpha=best_alpha[i], fit_intercept=False)
    r.fit(P, vectors[:, i])
    interp_P = phi(word_time, fine_time, best_eps[i])
    return i, r.coef_, np.dot(interp_P, r.coef_)

def lanczosinterp2D(data, oldtime, newtime, window=3, cutoff_mult=1.0, rectify=False):
    """Interpolates the columns of [data], assuming that the i'th row of data corresponds to
    oldtime(i). A new matrix with the same number of columns and a number of rows given
    by the length of [newtime] is returned.
    
    The time points in [newtime] are assumed to be evenly spaced, and their frequency will
    be used to calculate the low-pass cutoff of the interpolation filter.
    
    [window] lobes of the sinc function will be used. [window] should be an integer.

    Author: Alex Huth
    """
    ## Find the cutoff frequency ##
    cutoff = 1/np.mean(np.diff(newtime)) * cutoff_mult
    # print "Doing lanczos interpolation with cutoff=%0.3f and %d lobes." % (cutoff, window)
    
    ## Build up sinc matrix ##
    sincmat = np.zeros((len(newtime), len(oldtime)))
    for ndi in range(len(newtime)):
        sincmat[ndi,:] = lanczosfun(cutoff, newtime[ndi]-oldtime, window)
    
    if rectify:
        newdata = np.hstack([np.dot(sincmat, np.clip(data, -np.inf, 0)), 
                            np.dot(sincmat, np.clip(data, 0, np.inf))])
    else:
        ## Construct new signal by multiplying the sinc matrix by the data ##
        newdata = np.dot(sincmat, data)

    return newdata

if __name__ == '__main__':
    # Load epsilons and alpha values from `rbf_train.py`
    # best_eps [nhidden,]: Gaussian kernel width for every LSTM unit.
    # best_alpha [nhidden,]: Ridge parameter for interpolation for every LSTM unit.

    for story in allstories:
        # Load the story's word times, TR times, LSTM activations.
        # We use one story from the stimulus set for training interpolation weights $a$.
        # data_time (number of story words,): Time at which each word was spoken (We use midpoint of word duration)
        # tr_time (number of TRs,): Time at which each fMRI scan was taken
        # vecs (number of story words, LSTM hidden state dimensions) 
        nhidden = vecs.shape[1]
        print(data_time.shape, tr_time.shape, nhidden)
        # Timepoints for interpolation: 1e5 points evenly spaced between start and end of story.
        interp_time = np.linspace(np.floor(data_time[0]), np.ceil(data_time[-1]), int(1e5))

        # Parallelized RBF interpolation.
        start = time.time()
        pool = ThreadPool(processes=10)
        x = pool.map(lambda i: interpolation_function(data_time, interp_time, vecs, i), range(nhidden))
        x = np.array(x)
        end = time.time()
        print(x.shape, (end-start)/60)

        assert np.all(x[:, 0] == range(nhidden)), 'Incorrect LSTM unit order in parallelization step!: %s' % story

        # Save interpolation results.
        interp_wts = np.vstack(x[:, 1]).astype(np.float64)
        interp_vecs = np.vstack(x[:, 2]).astype(np.float64)
        dsinterp_vecs = lanczosinterp2D(interp_vecs.T, interp_time, tr_time).astype(np.float64)
        print(interp_vecs.shape, dsinterp_vecs.shape)
