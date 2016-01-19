
import collections

import numpy

import generate_g
import mcmc
import state_probabilities
import matplotlib.pyplot as plot

def calculate_final_distribution(G, O):
    sigmas, sig_ind_prob, samples = mcmc.sig_mcmc_2(G, [O], 10000, 100)

    bar1 = sig_ind_prob/samples;
    bar2 = numpy.ones(G.shape[0]) - bar1;

    ind = numpy.arange(G.shape[0]);
    p1 = plot.bar(ind, bar1, 0.35, color='r');
    p2 = plot.bar(ind, bar2, 0.35, color='b', bottom=bar1);

    plot.ylabel('L/R')
    plot.title('Switch index')
    plot.legend((p1[0], p2[0]), ('L', 'R'))
    plot.show();

    bar = numpy.zeros(len(sigmas));
    label = numpy.zeros(len(sigmas));
    ind = numpy.arange(len(sigmas));
    i = 0;
    for sigma, n in sigmas.items():
        bar[i] = n/float(samples);
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

def run_sigma_experiment_2(nodes=20, observations=10, numObservations=10, trials=10000, init=100, fileIndex=0):
    G, sig, D = mcmc.generate_graph_and_paths(nodes, observations, numObservations);
    sigmas, sig_ind_prob, samples = mcmc.sig_mcmc_2(G, D, trials, init);

    print(sig);
    bar1 = sig_ind_prob/samples;
    bar2 = numpy.ones(G.shape[0]) - bar1;

    ind = numpy.arange(G.shape[0]);
    p1 = plot.bar(ind, bar1, 0.35, color='r');
    p2 = plot.bar(ind, bar2, 0.35, color='b', bottom=bar1);

    plot.ylabel('L/R')
    plot.title('Switch index')
    plot.legend((p1[0], p2[0]), ('L', 'R'))
    plot.show();

    bar = numpy.zeros(len(sigmas));
    label = numpy.zeros(len(sigmas));
    ind = numpy.arange(len(sigmas));
    i = 0;
    for sigma, n in sigmas.items():
        bar[i] = n/float(samples);
        label[i] = mcmc.sigma_hash(sigma);
        i += 1;
    p1 = plot.bar(ind, bar, 0.1, color='black');
    #plot.xticks(ind+0.1/2., tuple(label), rotation=90);
    plot.ylabel('Probability of sigma')
    plot.xlabel('Sigma index')
    plot.title('Joint probability of sigma')
    plot.axis([0, len(sigmas), 0, max(bar)]);
    plot.show();

    true_sig_prob = 1;
    for i in range(0, nodes):
        if(sig[i] == 1):
            true_sig_prob *= bar1[i];
        else:
            true_sig_prob *= bar2[i];

    numpy.savetxt('test_results/sigma_experiment_2_ind_prob_L_{}_{}_{}_{}_{}.{}.txt'.format(nodes,observations,numObservations,trials,init,fileIndex), bar1);
    numpy.savetxt('test_results/sigma_experiment_2_search_space_{}_{}_{}_{}_{}.{}.txt'.format(nodes,observations,numObservations,trials,init,fileIndex), bar);
    numpy.savetxt('test_results/sigma_experiment_2_search_space_index_{}_{}_{}_{}_{}.{}.txt'.format(nodes,observations,numObservations,trials,init,fileIndex), label);

    print('Distribution error: {}'.format(mcmc.real_distribution_error(bar1, sig)));
    print('The probability of guessing the true sigma is {}'.format(true_sig_prob));
    print('posterior/prior = {}'.format(true_sig_prob/pow(0.5, nodes)));

def run_sigma_experiment_1(nodes=20, observations=10, numObservations=10, trials=10000, fileIndex=0):
    G, sig, D = mcmc.generate_graph_and_paths(nodes, observations, numObservations);
    sigmas, sig_ind_prob, samples = mcmc.sig_mcmc(G, D, trials);

    print(sig);
    bar1 = sig_ind_prob/samples;
    bar2 = numpy.ones(G.shape[0]) - bar1;

    ind = numpy.arange(G.shape[0]);
    p1 = plot.bar(ind, bar1, 0.35, color='r');
    p2 = plot.bar(ind, bar2, 0.35, color='b', bottom=bar1);

    plot.ylabel('L/R')
    plot.title('Switch index')
    plot.legend((p1[0], p2[0]), ('L', 'R'))
    plot.show();

    bar = numpy.zeros(len(sigmas));
    label = numpy.zeros(len(sigmas));
    ind = numpy.arange(len(sigmas));
    i = 0;
    for sigma, n in sigmas.items():
        bar[i] = n/float(samples);
        label[i] = mcmc.sigma_hash(sigma);
        i += 1;
    p1 = plot.bar(ind, bar, 0.1, color='black');
    #plot.xticks(ind+0.1/2., tuple(label), rotation=90);
    plot.ylabel('Probability of sigma')
    plot.xlabel('Sigma index')
    plot.title('Joint probability of sigma')
    plot.axis([0, len(sigmas), 0, max(bar)+0.05]);
    plot.show();

    true_sig_prob = 1;
    for i in range(0, nodes):
        if(sig[i] == 1):
            true_sig_prob *= bar1[i];
        else:
            true_sig_prob *= bar2[i];

    numpy.savetxt('test_results/sigma_experiment_1_ind_prob_L_{}_{}_{}_{}.{}.txt'.format(nodes,observations,numObservations,trials,fileIndex), bar1);
    numpy.savetxt('test_results/simga_experiment_1_search_space_index_{}_{}_{}_{}.{}.txt'.format(nodes,observations,numObservations,trials,fileIndex), label);
    numpy.savetxt('test_results/sigma_experiment_1_search_space_{}_{}_{}_{}.{}.txt'.format(nodes,observations,numObservations,trials,fileIndex), bar);

    print('Distribution error: {}'.format(mcmc.real_distribution_error(bar1, sig)));
    print('The probability of guessing the true sigma is {}'.format(true_sig_prob));
    print('posterior/prior = {}'.format(true_sig_prob/pow(0.5, nodes)));


def evaluate_sigma_mcmc(nodes=20, observations=10, numObservations=10, trials=10000, init=100, size=10):
    results = numpy.zeros(size);
    for i in range(0, size):
        G, sig, D = mcmc.generate_graph_and_paths(nodes, observations, numObservations);
        sigmas, sig_ind_prob, samples = mcmc.sig_mcmc_2(G, D, trials, init);
        bar1 = sig_ind_prob/samples;
        bar2 = numpy.ones(G.shape[0]) - bar1;
        true_sig_prob = 1;
        for j in range(0, nodes):
            if(sig[j] == 1):
                true_sig_prob *= bar1[j];
            else:
                true_sig_prob *= bar2[j];
        results[i] = true_sig_prob;
    return results, numpy.mean(results), sum(numpy.power(results - numpy.mean(results), 2));

def good_sigma_evaluator_1(nodes=20, observations=10, numObservations=10, trials=10000, size=10):
    results = numpy.zeros(size);
    for i in range(0, size):
        G, sig, D = mcmc.generate_graph_and_paths(nodes, observations, numObservations);
        sigmas, sig_ind_prob, samples = mcmc.sig_mcmc(G, D, trials);
        bar1 = sig_ind_prob/samples;
        results[i] = mcmc.real_distribution_error(bar1, sig);
    return results, numpy.mean(results), numpy.var(results);

def good_sigma_evaluator_2(nodes=20, observations=10, numObservations=10, trials=10000, init=100, size=10):
    results = numpy.zeros(size);
    for i in range(0, size):
        G, sig, D = mcmc.generate_graph_and_paths(nodes, observations, numObservations);
        sigmas, sig_ind_prob, samples = mcmc.sig_mcmc_2(G, D, trials, init);
        bar1 = sig_ind_prob/samples;
        results[i] = mcmc.real_distribution_error(bar1, sig);
    return results, numpy.mean(results), numpy.var(results);

def plot_sigma_observation_development(nodes=20, observations=10, numObservations=10, trials=10000, init=100, size=10, fileIndex=0):
	mean_1 = numpy.zeros(numObservations);
	var_1 = numpy.zeros(numObservations);
	mean_2 = numpy.zeros(numObservations);
	var_2 = numpy.zeros(numObservations);
	for i in range(0, numObservations):
		_, m, v = good_sigma_evaluator_1(nodes, observations, i, trials, size);
		mean_1[i] = m;
		var_1[i] = v;
		_, m, v = good_sigma_evaluator_2(nodes, observations, i, trials, init, size);
		mean_2[i] = m;
        var_2[i] = v;
	print('Simulations done!');
	plot.plot(range(1,numObservations+1), mean_1);
	plot.plot(range(1,numObservations+1), mean_2);
	plot.axis([1, numObservations, 0, 1]);
	#plot.legend((mean_1[0], mean_2[0]), ('mcmc_1', 'mcmc_2'))
	plot.ylabel('Mean Distribution error')
	plot.xlabel('Number of simulations')
	plot.show();
	plot.plot(range(1,numObservations+1), var_1);
	plot.plot(range(1,numObservations+1), var_2);
	#plot.legend((var_1[0], var_2[0]), ('mcmc_1', 'mcmc_2'))
	plot.ylabel('Distribution error Variance')
	plot.xlabel('Number of simulations')
	plot.show();
	numpy.savetxt('test_results/mcmc_mean_1_{}_{}_{}_{}_{}_{}.{}.txt'.format(nodes,observations,numObservations,trials,init,size, fileIndex), mean_1);
	numpy.savetxt('test_results/mcmc_mean_2_{}_{}_{}_{}_{}_{}.{}.txt'.format(nodes,observations,numObservations,trials,init,size, fileIndex), mean_2);
	numpy.savetxt('test_results/mcmc_var_1_{}_{}_{}_{}_{}_{}.{}.txt'.format(nodes,observations,numObservations,trials,init,size, fileIndex), var_1);
	numpy.savetxt('test_results/mcmc_var_2_{}_{}_{}_{}_{}_{}.{}.txt'.format(nodes,observations,numObservations,trials,init,size, fileIndex), var_2);	
	return mean_1, var_1, mean_2, var_2;

def run_convergence_experiment(nodes=20, observations=10, numObservations=10, trials=10000, init=100, chainSize=3):
    G, sig, D = mcmc.generate_graph_and_paths(nodes, observations, numObservations);
    chain_results_dict_array = [];
    sig_ind_prob_array = numpy.zeros((chainSize, nodes));
    for i in range(0, chainSize):
        sigmas, sig_ind_prob, samples = mcmc.sig_mcmc_2(G, D, trials, init);
        chain_results_dict_array.append(sigmas);
        sig_ind_prob_array[i,:] = sig_ind_prob;
        print('Chain {} processed...'.format(i+1));
    print('Calculating R...');
    result = mcmc.simple_convergence_checker(nodes, chain_results_dict_array, sig_ind_prob_array, samples);
    return result;
    
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

    failure_probabilities = []
    with open("observations_needed.dat", 'w') as obs_file:
        for nodes in range(6, max_nodes + 1, 2):
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

            obs_file.write("{}\t{}\n".format(nodes, total_observations_needed / runs))
            print(nodes, total_observations_needed / runs)
            failure_probability = failed_runs / (runs + failed_runs)
            print("Failure probability: {}".format(failure_probability))
            failure_probabilities.append(failure_probability)

    print("Average failure probability: {}".format(sum(failure_probabilities) / len(failure_probabilities)))


#if __name__ == '__main__':
    #run_experiment()
    #observations_needed()
