import numpy as np;
import random;

def generate_graph_1(n):
        G = np.zeros((n, n));
        degree = np.zeros(n);
        for a in range(1,4):
                for i in range(0,n-1):
                        c = 0;
                        if(degree[i] == a):
                                continue;
                        while True:
                                c = c + 1;
                                b = random.randrange(i+1, n);
                                if(degree[b] != a and degree[b] == degree[i] and G[i,b] == 0):
                                        G[i,b] = a;
                                        G[b,i] = a;
                                        degree[b] += 1;
                                        degree[i] += 1;
                                        break;
                                if c == 1000:
                                        print('Failed to generate graph');
                                        return np.zeros(1);
        return G;
	
def generate_graph_2(n):
        G = np.ones((n, 3))*(-1);
        degree = np.zeros(n);
        for a in range(0,3):
                for i in range(0,n-1):
                        c = 0;
                        if(degree[i] == a+1):
                                continue;
                        while True:
                                c = c + 1;
                                b = random.randrange(i+1, n);
                                q = G[i,0] != b and G[i,1] != b and G[i,2] != b;
                                if(degree[b] == degree[i] and q):
                                        G[i,a] = b;
                                        G[b,a] = i;
                                        degree[b] += 1;
                                        degree[i] += 1;
                                        break;
                                elif c == 1000:
                                        print('Failed to generate graph');
                                        return np.zeros(1);
        return G;

def generate_labels_randomly(n):
        l = np.zeros((n, 3));
        for a in range(0,3):
                for i in range(0,n):
                        while True:
                                b = random.randrange(1,4);
                                q = l[i,0] != b and l[i,1] != b and l[i,2] != b;
                                if(q):
                                        l[i,a] = b;
                                        break;
        return l;

def generate_settings_randomly(n):
        sig = np.zeros(n);
        for i in range(0, n):
                sig[i] = random.randrange(0,2);
        return sig;

def generate_graph_and_settings(n):
        G = np.zeros(1);
        while G.size == 1:
                G = generate_graph_2(n);
        l = generate_labels_randomly(n);
        sig = generate_settings_randomly(n);
        return (G, l, sig);
