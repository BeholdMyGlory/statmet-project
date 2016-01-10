import generate_g
import state_probabilities
import numpy as np;
import random;
import math;

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

# Inte implementerad; borde beräkna sannolikheten för ALLA sigma
def calculate_sigma_probabilities(G, D):
    print('Calculating probabilities for sigma...');
    sig_prob = dict();
    sig = np.ones(G.shape[0]);

# Beräknar mcmcn av sig
def sig_mcmc(G, D, t):
    # Initiera kön med 1000 slumpvis utvalda samples
    # Vore bättre om queue innehöll alla möjliga sigman
    sig_prob = dict();
    sig_count = dict();
    queue = [];
    n = G.shape[0];
    for i in range(0,1000):
        queue.append(np.random.randint(low=1, high=3, size=n));

    for _ in range(0,t):
        x = queue.pop(0);
        
        if not tuple(x) in sig_prob:
            p = 0;
            for i in range(0,len(D)):
                _, O_prob = state_probabilities.state_probabilities(G, x, D[i]);
                p += O_prob;
            sig_prob[tuple(x)] = p;
            
        xp = np.random.randint(low=1, high=3, size=n);
        if not tuple(xp) in sig_prob:
            p = 0;
            for i in range(0,len(D)):
                _, O_prob = state_probabilities.state_probabilities(G, xp, D[i]);
                p += O_prob;
            sig_prob[tuple(xp)] = p;

        # Borde vi ha p*(xs)/p*(x) också?
        alpha = math.exp(sig_prob[tuple(x)] - sig_prob[tuple(xp)]);

        r = min(1, alpha);
        u = random.random();

        if(u < r):
            queue.append(xp);
            if not tuple(xp) in sig_count:
                sig_count[tuple(xp)] = 0;
            sig_count[tuple(xp)] += 1;
        else:
            queue.append(x);
            if not tuple(x) in sig_count:
                sig_count[tuple(x)] = 0;
            sig_count[tuple(x)] += 1;
    return sig_prob, sig_count;
