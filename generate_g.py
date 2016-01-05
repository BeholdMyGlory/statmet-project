import numpy as np;
import random;

# Genererar grannmatris. Använd inte om vi inte måste
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

# Genererar grannlistor.	
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

# Genererar grannlistor där G[s,d] nästa nod från s åt håll d (s inte nödvändigtvis G[G[s,d],d])
def generate_graph_3(n):
        G = np.ones((n, 3))*(-1);
        for i in range(0,n-1):
                for a in range(0,3):
                        c = 0;
                        if(G[i,a] != -1):
                                continue;
                        while True:
                                c = c + 1;
                                b = random.randrange(i+1, n);
                                f = random.randrange(0,3);
                                
                                q = G[i,0] != b and G[i,1] != b and G[i,2] != b;
                                if(G[b, f] == -1 and q):
                                        G[i,a] = b;
                                        G[b,f] = i;
                                        break;
                                elif c == 1000:
                                        print('Failed to generate graph');
                                        return np.zeros(1);
        return G;

# Genererar grannlistor där G[s,d] nästa nod från s åt håll d (s = G[G[s,d],d])
def generate_graph_4(n):
        G = np.ones((n, 3))*(-1);
        for i in range(0,n-1):
                for a in range(0,3):
                        c = 0;
                        if(G[i,a] != -1):
                                continue;
                        while True:
                                c = c + 1;
                                b = random.randrange(i+1, n);
                                
                                q = G[i,0] != b and G[i,1] != b and G[i,2] != b;
                                if(G[b, a] == -1 and q):
                                        G[i,a] = b;
                                        G[b,a] = i;
                                        break;
                                elif c == 1000:
                                        print('Failed to generate graph');
                                        return np.zeros(1);
        return G;

# Genererar labels för switcharna. 1 = 0, 2 = L, 3 = R
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

# Genererar sigma. 0 = L, 1 = R
def generate_settings_randomly(n):
        sig = np.zeros(n);
        for i in range(0, n):
                sig[i] = random.randrange(0,2);
        return sig;

# Genererar graf och inställningar.
def generate_graph_and_settings(n):
        G = np.zeros(1);
        while G.size == 1:
                G = generate_graph_2(n);
        l = generate_labels_randomly(n);
        sig = generate_settings_randomly(n);
        return (n, G, l, sig);

# Genererar graf på annat sätt
# mode = 1 anger att s inte nödvändigtvis G[G[s,d],d], mode 2 anger s = G[G[s,d],d]
def generate_graph_and_settings_2(n, mode):
        G = np.zeros(1);
        while G.size == 1:
                if mode == 0:
                        G = generate_graph_3(n);
                elif mode == 1:
                        G = generate_graph_4(n);
        sig = generate_settings_randomly(n);
        return (n, G, sig);
                

# Simulerar tåg med given graf
def simulate_train(graph, t):
        O = [];
        actual_path = [];
        s = random.randrange(0,int(graph[0]));
        orientation = random.randrange(1,4);
        actual_path.append(s);
        
        for i in range(0, t-1):
                # Hitta nästa håll att åka mot
                # Samt gör observation av hur man åker
                # OL = 1, OR = 2, L0 = 3, R0 = 4
                ext = 0;
                obs = 0;
                if orientation == 1:
                        if graph[3][s] == 0:
                                obs = 1;
                                ext = 2;
                        else:
                                obs = 2;
                                ext = 3;
                else:
                        ext = 1;
                        if orientation == 2:
                                obs = 3;
                        else:
                                obs = 4;

                # Simulera felaktigheter i mätning
                prob = random.randrange(0, 100);
                if prob < 5:
                        print('Some wrong observation made');
                        wrongObs = obs;
                        while wrongObs == obs:
                                wrongObs = random.randrange(1,5);
                        obs = wrongObs;
                O.append(obs);
                        
                for a in range(0,3):
                        if graph[2][s,a] == ext:
                                s = graph[1][s,a];
                                actual_path.append(s);
                                orientation = graph[2][s,a];
                                break;
        return (O, actual_path);

# Simulerar tåg med given graf där G[s,d] nästa nod från s åt håll d
def simulate_train_2(graph, t):
        O = [];
        actual_path = [];
        s = random.randrange(0,int(graph[0]));
        orientation = random.randrange(0,3);
        actual_path.append(s);
        
        for i in range(0, t-1):
                # Hitta nästa håll att åka mot
                # Samt gör observation av hur man åker
                # obs: OL = 1, OR = 2, L0 = 3, R0 = 4
                # ext: 0 = 0, 1 = L, 2 = R
                ext = -1;
                obs = 0;
                if orientation == 0:
                        if graph[2][s] == 0:
                                obs = 1;
                                ext = 1;
                        else:
                                obs = 2;
                                ext = 2;
                else:
                        ext = 0;
                        if orientation == 1:
                                obs = 3;
                        else:
                                obs = 4;

                # Simulera felaktigheter i mätning
                prob = random.randrange(0, 100);
                if prob < 5:
                        print('Some wrong observation made');
                        wrongObs = obs;
                        while wrongObs == obs:
                                wrongObs = random.randrange(1,5);
                        obs = wrongObs;
                O.append(obs);

                s_old = s;
                s = graph[1][s,ext];
                for a in range(0,3):
                        if graph[1][s,a] == s_old:
                                orientation = a;
                                break;
                actual_path.append(s);
        return (O, actual_path);
