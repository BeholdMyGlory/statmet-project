
import collections
import itertools
import math
import random

import numpy as np

import generate_g
from state_probabilities import state_probabilities

# Genererar graf och data
def generate_graph_and_paths(n, t, d):
    print('Generating graph and data...');
    G, sig = generate_g.generate_graph_and_settings(n)
    D = [];  
    for i in range(0,d):
        a, _ = generate_g.simulate_train(G, sig, t);
        D.append(a);
    print('Done!');
    return G, sig, D;

def sample_sigma(old_sigma=None, n=None):
    n = n or old_sigma.shape[0]
    return np.random.randint(low=1, high=3, size=n)

def mcmc_chain(G, D, sig_prob=None):
    sig_prob = sig_prob if sig_prob is not None else dict()

    calculate_probability = lambda sigma: sum(
        state_probabilities(G, sigma, O)[1] for O in D)

    sample = sample_sigma(n=G.shape[0])
    prob = calculate_probability(sample)
    sig_prob[tuple(sample)] = prob

    while True:
        new_sample = sample_sigma(sample)
        try:
            new_prob = sig_prob[tuple(new_sample)]
        except KeyError:
            new_prob = calculate_probability(new_sample)
            sig_prob[tuple(new_sample)] = new_prob

        alpha = math.exp(new_prob - prob)

        r = min(1, alpha)
        u = random.random()

        if u < r:
            sample = new_sample
            prob = new_prob

        yield sample

# Beräknar mcmcn av sig
def sig_mcmc(G, D, t):
    sig_count = collections.defaultdict(int)
    n = G.shape[0];
    # sig_individual_prob är sannolikheten att sigma är lika med 1.
    # p(sig_i = L) = sig_individual_prob[sig_i]/t
    # p(sig_i = R) = 1 - sig_individual_prob[sig_i]/t
    sig_individual_prob = np.zeros(n);
    
    chain = mcmc_chain(G, D)

    for sample in itertools.islice(chain, t):
        sig_count[tuple(sample)] += 1
        for i in range(0, n):
            sig_individual_prob[i] += sample[i] == 1;
        
    return sig_count, sig_individual_prob, t

def sigma_hash(sigma):
    d = 0;
    hash_value = 0;
    for i in sigma:
        if i==2:
            hash_value += pow(2, d);
        d += 1;

    return hash_value;
