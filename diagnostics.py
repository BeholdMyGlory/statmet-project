
import collections
import functools
import itertools
import math
import sys

import numpy
from matplotlib import pyplot

import generate_g
import mcmc
import state_probabilities

def build_chains(specification, G, O):
    probs = {}
    return [[chain(G, [O], sig_prob=probs, sampler=functools.partial(sampler, **kwargs))
             if sampler is not None else chain(G, [O], sig_prob=probs)
             for _ in range(num_chains)]
            for num_chains, chain, sampler, kwargs in specification]

def setup(nodes, chain_specification, burn_in=1000, calculate_actual_distribution=False):
    G, sigma = generate_g.generate_graph_and_settings(nodes)
    if G.min() < 0:
        print("Erroneous graph!")
        return

    O, path = generate_g.simulate_train(G, sigma, 10)

    chains = [[itertools.islice(chain, burn_in, None)
               for chain in chains]
              for chains in build_chains(chain_specification, G, O)]

    if calculate_actual_distribution:
        sigma_dist = calculate_sigma(G, O)
        return chains, sigma_dist
    else:
        return chains

def calculate_sigma(G, O):
    probabilities = collections.defaultdict(int)
    normalizer = -math.inf

    print("Calculating sigma...")
    for sigma in itertools.product((1, 2), repeat=G.shape[0]):
        sigma_arr = numpy.array(sigma)
        _, O_prob = state_probabilities.state_probabilities(G, sigma_arr, O)
        probabilities[sigma] = O_prob
        normalizer = numpy.logaddexp(normalizer, O_prob)

    for sigma, prob in probabilities.items():
        probabilities[sigma] = math.exp(prob - normalizer)

    print("Calculated sigma.")
    return probabilities

def convergence(chain_specification, data_file, nodes=8, window=200):
    chain_collection = setup(nodes, chain_specification, burn_in=0)
    all_Rs = []

    for chains in chain_collection:
        samples = [[] for chain in chains]
        delta_Rs = []
        try:
            for i in range(2000):
                for chain, sample in zip(chains, samples):
                    sample.append(next(chain))

                if i < 10 or i % 10 != 0:
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

                delta_R = max(abs(r - 1) for r in R_values)
                print(i, delta_R)
                delta_Rs.append(delta_R)
        except KeyboardInterrupt:
            print("Samples:", i)

        pyplot.plot(numpy.arange(11, len(delta_Rs)*10 + 11, 10), delta_Rs)
        all_Rs.append(delta_Rs)

    pyplot.show()

    with open(data_file, 'w') as f:
        for i, values in enumerate(zip(*all_Rs), 1):
            f.write("{}\t{}\n".format(i*10 + 1, "\t".join(map(str, values))))

def jensen_shannon(P, Q, num_P, num_Q):
    entropy = lambda p: - p * math.log2(p) if p > 0 else 0

    jsd = 0
    for sample in P.keys() | Q.keys():
        p = P[sample] / num_P
        q = Q[sample] / num_Q

        jsd += entropy((p + q) / 2) - (entropy(p) + entropy(q))/2

    return jsd

def samples_required(chain_specification, data_file, nodes=12, num_samples=50000):
    chains, sigma_dist = setup(nodes, chain_specification, burn_in=1000,
                               calculate_actual_distribution=True)

    all_jsds = []
    for test_chains, (num_chains, chain_type, _, kwargs) in zip(chains, chain_specification):
        jsds = []

        samples = collections.defaultdict(int)
        combined_test_chain = itertools.chain.from_iterable(zip(*test_chains))
        for i, sample in itertools.islice(enumerate(combined_test_chain, 1), 10000):
            samples[tuple(sample)] += 1
            jsd = jensen_shannon(samples, sigma_dist, i, 1)
            print(i, jsd)
            jsds.append(jsd)

        pyplot.plot(numpy.arange(1, len(jsds) + 1), jsds,
                    label="{} ({} chains{})".format(
                        chain_type.__name__, num_chains,
                        "; {} switches resampled".format(kwargs['switches_to_sample'])
                        if kwargs is not None and "switches_to_sample" in kwargs else ""))
        all_jsds.append(jsds)

    pyplot.legend()
    pyplot.show()

    with open(data_file, 'w') as f:
        for i, values in enumerate(zip(*all_jsds), 1):
            f.write("{}\t{}\n".format(i, "\t".join(map(str, values))))

if __name__ == '__main__':
    nodes = 8
    chains = [
        (1, mcmc.mcmc_chain, mcmc.sample_sigma_uniformly, {}),
        (1, mcmc.mcmc_chain, mcmc.sample_sigma_uniformly, {"switches_to_sample": 1}),
        (1, mcmc.mcmc_chain, mcmc.sample_sigma_uniformly, {"switches_to_sample": nodes//4}),
        (1, mcmc.mcmc_chain, mcmc.sample_sigma_uniformly, {"switches_to_sample": nodes//2}),
        (1, mcmc.mcmc_chain_2, None, None),
        (4, mcmc.mcmc_chain, mcmc.sample_sigma_uniformly, {}),
        (4, mcmc.mcmc_chain, mcmc.sample_sigma_uniformly, {"switches_to_sample": 1}),
        (4, mcmc.mcmc_chain, mcmc.sample_sigma_uniformly, {"switches_to_sample": nodes//4}),
        (4, mcmc.mcmc_chain, mcmc.sample_sigma_uniformly, {"switches_to_sample": nodes//2}),
        (4, mcmc.mcmc_chain_2, None, None)
    ]
    #convergence(chains, "burn_in_16.dat", nodes=nodes)
    samples_required(chains, "convergence_8.dat", nodes=nodes)
