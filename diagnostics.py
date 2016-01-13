
import collections
import itertools
import math
import sys

import generate_g
import mcmc

def setup(nodes, burn_in=1000, num_chains=5):
    G, sigma = generate_g.generate_graph_and_settings(nodes)
    if G.min() < 0:
        print("Erroneous graph!")
        return

    O, path = generate_g.simulate_train(G, sigma, 10)

    probs = dict()

    chains = [mcmc.mcmc_chain(G, [O], sig_prob=probs) for _ in range(num_chains)]

    for _ in range(burn_in):
        for chain in chains:
            next(chain)

    return chains

def convergence(nodes=8, window=200):
    chains = setup(nodes, burn_in=0)

    samples = [[] for chain in chains]
    try:
        for i in range(10000):
            for chain, sample in zip(chains, samples):
                sample.append(next(chain))

            if i < 10:
                continue

            R_values = []

            for n in range(nodes):
                w_averages = []

                for sample in samples:
                    w_averages.append(sum(x[n] for x in sample) / len(sample))


                b_average = sum(w_averages) / len(w_averages)

                B = sum((yc - b_average)**2 for yc in w_averages) * len(samples[0]) / (len(w_averages) - 1)

                W = sum(sum((ysc[n] - yc)**2 for ysc in sample) / (len(sample) - 1)
                        for yc, sample in zip(w_averages, samples)) / len(w_averages)

                V = (len(samples[0]) - 1)/len(samples[0]) * W + 1/len(samples[0]) * B

                R_values.append(math.sqrt(V / W) if V > 0 and V != W else 1)

            print(max(abs(r - 1) for r in R_values))
    except KeyboardInterrupt:
        print("Samples:", i)


def jensen_shannon(P, Q, num_P, num_Q):
    entropy = lambda p: - p * math.log2(p) if p > 0 else 0

    jsd = 0
    for sample in P.keys() | Q.keys():
        p = P[sample] / num_P
        q = Q[sample] / num_Q

        jsd += entropy((p + q) / 2) - (entropy(p) + entropy(q))/2

    return jsd

def samples_required(nodes=10, num_chains=5, num_samples=50000, test_chains=5):
    chains = setup(nodes, burn_in=1000, num_chains=num_chains + test_chains)
    sample_chains, test_chains = chains[:num_chains], chains[num_chains:]
    truth = collections.defaultdict(int)

    combined_chain = itertools.chain.from_iterable(zip(*sample_chains))
    for sample in itertools.islice(combined_chain, num_samples):
        truth[tuple(sample)] += 1

    samples = collections.defaultdict(int)
    combined_test_chain = itertools.chain.from_iterable(zip(*test_chains))
    for i, sample in enumerate(combined_test_chain, 1):
        samples[tuple(sample)] += 1
        print(i, jensen_shannon(samples, truth, i, num_samples))


if __name__ == '__main__':
    #convergence(int(sys.argv[1]))
    samples_required()
