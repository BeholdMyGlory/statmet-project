
import collections

import numpy

import generate_g
import mcmc
import state_probabilities
import matplotlib.pyplot as plot

def calculate_final_distribution(G, O):
    sigmas, sig_ind_prob, samples = mcmc.sig_mcmc(G, [O], 10000)

    # Sannolikhet för individuella sigman
    bar1 = sig_ind_prob/samples;
    bar2 = numpy.ones(G.shape[0]) - bar1;

    ind = numpy.arange(G.shape[0]);
    p1 = plot.bar(ind, bar1, 0.35, color='r');
    p2 = plot.bar(ind, bar2, 0.35, color='b', bottom=bar1);

    plot.ylabel('L/R')
    plot.title('Switch index')
    plot.legend((p1[0], p2[0]), ('L', 'R'))
    plot.show();

    # Sannolikhet för samling av sigman
    bar = numpy.zeros(len(sigmas));
    label = numpy.zeros(len(sigmas));
    ind = numpy.arange(len(sigmas));
    i = 0;
    for sigma, n in sigmas.items():
        bar[i] = n/samples;
        label[i] = mcmc.sigma_hash(sigmas);
        i += 1;
    p1 = plot.bar(ind, bar, 0.1, color='black');
    #plot.xticks(ind+0.1/2., tuple(label));
    plot.ylabel('Probability of sigma')
    plot.xlabel('Sigma index')
    plot.title('Joint probability of sigma')
    plot.show();

        
    s = numpy.zeros(G.shape)
    for sigma, n in sigmas.items():
        sigma = numpy.array(sigma)
        # s2 is already normalized, i.e. it is
        # p(s, O | G, sigma)/p(O | G, sigma) = p(s | G, O, sigma)
        s2, _ = state_probabilities.state_probabilities(G, sigma, O)
        # multiply s2 by p(sigma | G, O)
        s2 *= n / samples
        # sum over all sigmas
        s += s2

    return s

def run_sigma_experiment(nodes=20, observations=10, numObservations=10, trials=10000):
    G, sig, D = mcmc.generate_graph_and_paths(nodes, observations, numObservations);
    sigmas, sig_ind_prob, samples = mcmc.sig_mcmc(G, D, trials);

    print(sig);
    # Sannolikhet för individuella sigman
    bar1 = sig_ind_prob/samples;
    bar2 = numpy.ones(G.shape[0]) - bar1;

    ind = numpy.arange(G.shape[0]);
    p1 = plot.bar(ind, bar1, 0.35, color='r');
    p2 = plot.bar(ind, bar2, 0.35, color='b', bottom=bar1);

    plot.ylabel('L/R')
    plot.title('Switch index')
    plot.legend((p1[0], p2[0]), ('L', 'R'))
    plot.show();

    # Sannolikhet för samling av sigman
    bar = numpy.zeros(len(sigmas));
    label = numpy.zeros(len(sigmas));
    ind = numpy.arange(len(sigmas));
    i = 0;
    for sigma, n in sigmas.items():
        bar[i] = n/samples;
        label[i] = mcmc.sigma_hash(sigma);
        i += 1;
    p1 = plot.bar(ind, bar, 0.1, color='black');
    #plot.xticks(ind+0.1/2., tuple(label), rotation=90);
    plot.ylabel('Probability of sigma')
    plot.xlabel('Sigma index')
    plot.title('Joint probability of sigma')
    plot.show();

    true_sig_prob = 1;
    for i in range(0, nodes):
        if(sig[i] == 1):
            true_sig_prob *= bar1[i];
        else:
            true_sig_prob *= bar2[i];

    print('The probability of guessing the true sigma is {}'.format(true_sig_prob));
    print('posterior/prior = {}'.format(true_sig_prob/pow(0.5, G.shape[0])));
    
def run_experiment(nodes=20, observations=10):
    G, sigma = generate_g.generate_graph_and_settings(nodes)
    O, actual_path = generate_g.simulate_train(G, sigma, observations + 1)

    last_node = actual_path[-2]
    last_label = numpy.where(G[last_node] == actual_path[-1])[0][0]

    s = calculate_final_distribution(G, O)

    print("Guessed the correct stop position with probability {}.".format(
        s[last_node, last_label]))


def observations_needed(max_nodes=100, runs=50):
    """Calculates the number of observations needed
    to get a good estimate of the stop position."""

    for nodes in range(6, max_nodes + 1):
        total_observations_needed = 0
        failed_runs = 0

        i = 0
        while True:
            G, sigma = generate_g.generate_graph_and_settings(nodes)
            O, actual_path = generate_g.simulate_train(G, sigma, 200)

            for observations in range(1, len(O)):
                s, _ = state_probabilities.state_probabilities(
                    G, sigma, O[:observations])
                if s.max() > 0.90:
                    break
            else:
                print(s.max())
                failed_runs += 1
                continue

            total_observations_needed += observations
            i += 1
            if i >= runs:
                break

        print(nodes, total_observations_needed / runs)
        print("Failure probability: {}".format(
            failed_runs / (runs + failed_runs)))


if __name__ == '__main__':
    run_experiment()
    #observations_needed()
