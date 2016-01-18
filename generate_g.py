import numpy as np
import random

def generate_graph(n):
    G = np.ones((n, 3), dtype=int) * (-1)
    for i in range(0, n - 1):
        for a in range(0, 3):
            c = 0
            if G[i, a] != -1:
                continue
            while True:
                c = c + 1
                b = random.randrange(i + 1, n)
                f = random.randrange(0, 3)

                q = G[i, 0] != b and G[i, 1] != b and G[i, 2] != b
                if G[b, f] == -1 and q:
                    G[i, a] = b
                    G[b, f] = i
                    break
                elif c == 1000:
                    print('Failed to generate graph');
                    return np.zeros(1)
    for a in range(0,3):
        if G[n-1, a] == -1:
            print('Failed to generate graph');
            return np.zeros(1);
    return G

def generate_graph_and_settings(n):
    G = np.zeros(1)
    while G.size == 1:
        G = generate_graph(n)
    # L = 1, R = 2
    sigma = np.random.randint(low=1, high=3, size=n)
    return G, sigma

def simulate_train(G, sigma, t):
    O = []
    actual_path = []
    s = random.randrange(0, G.shape[0])
    orientation = random.randrange(0, 3)
    actual_path.append(s)

    for i in range(0, t - 1):
        ext = -1
        obs = 0
        if orientation == 0:
            obs = sigma[s]
            ext = sigma[s]
        else:
            ext = 0
            if orientation == 1:
                obs = 3
            else:
                obs = 4

        prob = random.randrange(0, 100)
        if prob < 5:
            print('Some wrong observation made')
            wrongObs = obs
            while wrongObs == obs:
                wrongObs = random.randrange(1, 5)
            obs = wrongObs
        O.append(obs)

        s_old = s
        s = G[s, ext]
        for a in range(0, 3):
            if G[s, a] == s_old:
                orientation = a
                break
        actual_path.append(s)
    return O, actual_path

if __name__ == '__main__':
    G, sigma = generate_graph_and_settings(6)
    O, path = simulate_train(G, sigma, 20)
    print("Actual path:", path)
    print("Observations:", O)
    
    import state_probabilities
    s, O_prob = state_probabilities.state_probabilities(G, sigma, O)
    predicted_node, predicted_label = np.unravel_index(s.argmax(), s.shape)
    print("Last observation was generated after exiting " \
          "node {} at label {} with probability {}".format(
        predicted_node, "0LR"[predicted_label], s.max()))

