
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

def sample_posterior_sigma(sigma_ind_prob, t):
    sigma = np.zeros(sigma_ind_prob.shape[0]);
    for i in range(0, sigma_ind_prob.shape[0]):
        u = random.random();
        if u < sigma_ind_prob[i]/t:
            sigma[i] = 2;
        else:
            sigma[i] = 1;
    return sigma;

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
        
def mcmc_chain_2(G, D, sig_prob=None, sig_ind_prob=None):
    sig_prob = sig_prob if sig_prob is not None else dict()
    sig_ind_prob = sig_ind_prob if sig_ind_prob is not None else np.ones(G.shape[0]);
    sig_ind_prob *= 100;
    t = 200;

    calculate_probability = lambda sigma: sum(
        state_probabilities(G, sigma, O)[1] for O in D)

    sample = sample_sigma(n=G.shape[0])
    prob = calculate_probability(sample)
    sig_prob[tuple(sample)] = prob

    while True:
        new_sample = sample_posterior_sigma(sig_ind_prob, t)
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
            
        sig_ind_prob += sample - np.ones(G.shape[0]);
        t += 1;

        yield sample
        
# Beräknar mcmcn av sig
def sig_mcmc(G, D, t):
    sig_count = collections.defaultdict(int)
    n = G.shape[0];
    # sig_individual_prob är sannolikheten att sigma är lika med 1.
    # p(sig_i = L) = sig_individual_prob[sig_i]/t
    # p(sig_i = R) = 1 - sig_individual_prob[sig_i]/t
    sig_individual_prob = np.zeros(n);
    
    chain = mcmc_chain_2(G, D)

    for sample in itertools.islice(chain, t):
        sig_count[tuple(sample)] += 1
        for i in range(0, n):
            sig_individual_prob[i] += sample[i] == 1;
        
    return sig_count, sig_individual_prob, t+200

def sigma_hash(sigma):
    d = 0;
    hash_value = 0;
    for i in sigma:
        if i==2:
            hash_value += pow(2, d);
        d += 1;

    return hash_value;

# Beräknar EPSR
# sig_ind_prob_array np.array som förhoppningsvis motsvarar y_c
def simple_convergence_checker(nodes, chain_results_dict_array, sig_ind_prob_array, samples):
    C = len(chain_results_dict_array);
    y_dot = np.zeros(nodes);

    # Beräkna y_dot
    for c in range(0, C):
        y_dot += sig_ind_prob_array[c,:];
    y_dot /= C;

    # Beräkna B
    s = np.zeros(nodes);
    for i in range(0, C):
        s += np.power(sig_ind_prob_array[i,:] - y_dot, 2);
    B = (samples / (C - 1)) * sum(s);
    
    # Beräkna W
    W = 0;
    for c in range(0, C):
        tmpSum = 0;
        for sigma, n in chain_results_dict_array[c].items():
            tmp = np.zeros(nodes);
            for i in range(0, nodes):
                if(sigma[i] == 1):
                    tmp[i] = 1;
                else:
                    tmp[i] = 0;
            tmpSum += sum(np.power(tmp - sig_ind_prob_array[c], 2))*n;
        W += (1 / (samples - 1)) * tmpSum;
    W /= C;

    # Beräkna V_hat
    V = (samples - 1)*W/samples + (B/samples);

    # Beräkna R_hat:
    R = math.sqrt(V/W);
    return R;
