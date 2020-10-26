"""
Interpretable multi-timescale models for predicting fMRI responses to continuous natural speech
NeurIPS`2020

Training interpolation weights $a_i$.

Author: Shailee Jain
"""
import numpy as np
from sklearn.linear_model import RidgeCV
import scipy
from multiprocessing.pool import ThreadPool
import argparse
import time

# Gaussian RBF kernel: e^(-(eps*distance)^2)
phi = lambda t, tau, epsilon: np.exp(-(epsilon * (t[None] - tau[:,None]))** 2)

def interpolation_function(word_time, fine_time, vectors, i):
    P = phi(word_time, word_time, best_eps[i])
    r = RidgeCV(alphas=alpha_vals, fit_intercept=False, store_cv_values=True)
    r.fit(P, vectors[:, i])
    interp_P = phi(word_time, fine_time, best_eps[i])
    return i, r.coef_, r.predict(interp_P), r.alpha_

if __name__ == '__main__':
    # Load training data: word times, TR times, LSTM activations (vecs).
    # We use one story from the stimulus set for training interpolation weights $a$.
    # data_time (number of story words,): Time at which each word was spoken (We use midpoint of word duration)
    # tr_time (number of TRs,): Time at which each fMRI scan was taken
    # vecs (number of story words, LSTM hidden state dimensions) 
    nhidden = vecs.shape[1]
    print(data_time.shape, tr_time.shape, nhidden)
    # Timepoints for interpolation: 1e5 points evenly spaced between start and end of story.
    interp_time = np.linspace(np.floor(data_time[0]), np.ceil(data_time[-1]), int(1e5))

    # Compute epsilons (Gaussian kernel width).
    # Mean word duration ~ stddev of word time.
    # allstories: List of all stories in the stimulus set.
    duration = []
    for story in allstories:
        # story_data_time (number of story words,): Time at which each word was spoken.
        duration.extend(story_data_time[1:] - story_data_time[:-1])
    mean_duration = np.mean(duration)
    print('Mean word duration:', mean_duration)

    # LSTM Timescale assignments.
    T = scipy.stats.invgamma.isf(np.linspace(0, 1, 1151), a=0.56, scale=1)[1:]
    best_eps = 1 / (T * mean_duration) # 1e2 * (T**(0.56))
    best_eps = np.clip(best_eps, 0.1, np.inf)
    best_eps[-1] = best_eps[-2] # best_eps[-1] is inf.

    # Alpha values for cross-validation.
    alpha_vals = np.logspace(-5, 0, 10)

    # Parallelized GCV for RBF interpolation.
    start = time.time()
    pool = ThreadPool(processes=10)
    # Each LSTM unit is interpolated separately.
    x = pool.map(lambda i: interpolation_function(data_time, interp_time, vecs, i), range(nhidden))
    x = np.array(x)
    end = time.time()
    print(x.shape, (end-start)/60)

    assert np.all(x[:, 0] == range(nhidden)), 'Incorrect LSTM unit order in parallelization step!'

    # Save interpolation results.
    interp_wts = np.vstack(x[:, 1]).astype(np.float64) # Learned interpolation weights $a_i$.
    interp_vecs = np.vstack(x[:, 2]).astype(np.float64) # Interpolated LSTM activations for the training story.
    best_alpha = x[:, 3].astype(np.float64) # Ridge parameter for interpolation for every LSTM unit.
    print(interp_vecs.shape)
