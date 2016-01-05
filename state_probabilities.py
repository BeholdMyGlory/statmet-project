
import functools
import math
import random

import numpy

def state_probabilities(G, sigma, O, p=0.05):
    """ Calculates the joint probability of the observations
    and the ending state.
    
    Args:
        G (numpy.array): An n x 3 matrix where G[i,j]
            is the neighbour of i at label j.
        sigma (numpy.array): A length n array where sigma[i]
            is the switch setting for node i (1 or 2).
        O (list): A list of observed signals (1, 2, 3 or 4).
        p (float): The probability of a corrupted signal.
    
    Returns:
        (numpy.array, float): A 2-tuple, where the first element
            is the distribution over ending states, and
            the second element is the observation probability.
        
        The distribution over states is given as an n x 3 matrix s,
        where s[i,j] is the probability of just having just exited
        node i through label j. The matrix is normalized with respect
        to the observation probability.
        
        The observation probability is the probability of seeing
        the observation sequence. The observation probability
        is given in (natural) log space.
    """
    
    nodes, edge_per_node = G.shape
    # N = number of edges * 2 possible directions = nodes * edge_per_node
    N = nodes * edge_per_node
    
    # s[i,j] is the probability of having exited node i through label j
    # at the current time step
    s = numpy.log(numpy.ones((nodes, edge_per_node)) / N)
    
    rows, columns = numpy.indices(G.shape)
    rows = rows[:,:,numpy.newaxis].repeat(2, axis=2)
    
    columns = columns[:,numpy.newaxis,:].repeat(3, axis=1)
    columns = columns[columns != [[0], [1], [2]]].reshape((nodes, 3, 2))
    
    # neighbours[i,j] are the neighbours of i, G[i], with j removed,
    # i.e. [u, w] when e = (i, *) has label j
    neighbours = G[rows,columns]
    
    x, y, z, labels = numpy.where(
        G[neighbours] == numpy.arange(nodes).reshape((nodes, 1, 1, 1)))
    exit_labels = numpy.zeros((nodes, 3, 2), dtype=int)
    
    # exit_labels[i,j,k] is the label of the edge
    # f = (neighbours[i,j,k], i) from the perspective
    # of neighbours[i,j,k]
    exit_labels[x,y,z] = labels
    
    # entry_labels[i,j,k] is the label of the edge
    # f = (neighbours[i,j,k], i) from the perspective
    # of i
    entry_labels = columns
    
    # current_labels[i,j,k] is the label j,
    # i.e. the exit label from i in the case
    # when exiting i via j and coming from neighbours[i,j,k]
    _, current_labels, _ = numpy.indices(columns.shape)
    
    # maps OL = 1, OR = 2, L0 = 3, R0 = 4
    signals = numpy.array([
        [-1, 1, 2],
        [3, -1, -1],
        [4, -1, -1]
    ])
    
    # signal_matrix[i,j,k] is the signal generated when
    # passing the labels entry_labels[i,j,k] and j
    signal_matrix = signals[entry_labels,current_labels]
    
    # a is the transition matrix, i.e. a[i,j,k] is the probability
    # of going from neighbours[i,j,k] and exiting i at label j --
    # 1 if coming from L or R and exiting through 0 (signal >= 3),
    # or if coming from 0 and exiting through L or R
    # (signal >= 1) and the exit label corresponds to sigma.
    # 0 otherwise
    a = (signal_matrix >= 3) | ((signal_matrix >= 1) &
                                (current_labels == sigma[rows]))
    # don't warn about -inf probabilities
    with numpy.errstate(divide='ignore'):
        a = numpy.log(a)
    
    for signal in O:
        # the probability of seeing signal given that the
        # actual signal is signal_matrix[i,j,k], i.e.
        # 1 - p if signal_matrix[i,j,k] == signal; p / 3 otherwise.
        # 0 if signal is impossible, i.e. RR, RL, LR, LL
        b = (signal_matrix == signal) * (1 - p) + \
            ((signal_matrix >= 1) & (signal_matrix != signal)) * p/3
        # don't warn about -inf probabilities
        with numpy.errstate(divide='ignore'):
            b = numpy.log(b)
        
        s = s[neighbours,exit_labels] + a + b
        # addition of the two cases (the two possible
        # previous states) in log space
        s = numpy.logaddexp(s[:,:,0], s[:,:,1])
    
    O_prob = functools.reduce(lambda x, y: numpy.logaddexp(x, y), s.flatten())
    normalized = numpy.exp(s - O_prob)
    
    return normalized, O_prob

if __name__ == '__main__':
    p = 0.05
    num_observations = 3
    
    edge_0 = [1, 4, 3, 2, 1, 4, 3, 4]
    edge_L = [3, 0, 0, 0, 7, 7, 5, 5]
    edge_R = [2, 2, 1, 6, 5, 6, 7, 6]
    G = numpy.array([edge_0, edge_L, edge_R]).T
    
    sigma = numpy.array([2, 1, 2, 2, 1, 2, 2, 2])
    
    O = []
    signals = {
        (0, 1): 1,
        (0, 2): 2,
        (1, 0): 3,
        (2, 0): 4
    }
    
    current_node, current_label = 4, 2 # after exiting node 4 at label 0
    for _ in range(num_observations):
        # the node connected to current_label at current_node
        next_node = G[current_node,current_label]
        # the corresponding label from next_node's perspective
        entry_label = numpy.where(G[next_node] == current_node)[0][0]
        # the label corresponding to next_node's exit
        exit_label = sigma[next_node] if entry_label == 0 else 0
        
        # add both labels; report the wrong label pair with probability p
        # (the wrong label pairs have equal probability)
        signal = signals[entry_label, exit_label]
        possible_errors = {1, 2, 3, 4} - {signal}
        O.append(signal if random.random() > p
                 else random.choice(list(possible_errors)))
        
        current_node, current_label = next_node, exit_label
    
    print("The final state was ({}, {}).".format(
        current_node, "0LR"[current_label]))
    
    s, O_prob = state_probabilities(G, sigma, O)
    
    predicted_node, predicted_label = numpy.unravel_index(
        s.argmax(), s .shape)
    print("The predicted final state is ({}, {}) with probability {}.".format(
        predicted_node, "0LR"[predicted_label], s.max()))
    
    print("The probability of the observation sequence " \
          "was {} (log space: {}).".format(math.exp(O_prob), O_prob))

