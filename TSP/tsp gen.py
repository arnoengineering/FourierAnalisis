import numpy as np


def do_crossover(s1, s2, m):
    s1, s2 = s1.copy(), s2.copy()
    c1 = s2.copy()
    for i in range(m, len(s1)): c1.remove(s1[i])
    for i in range(m, len(s1)): c1.append(s1[i])
    c2 = s1.copy()
    for i in range(m, len(s2)): c2.remove(s2[i])
    for i in range(m, len(s2)): c2.append(s2[i])
    return (c1, c2)


def do_mutation(s, m, n):
    i, j = min(m, n), max(m, n)
    s1 = s.copy()
    while i < j:
        s1[i], s1[j] = s1[j], s1[i]
        i += 1
        j -= 1
    return s1


def compute_fitness(G, s):
    l = 0
    for i in range(len(s) - 1):
        l += G[s[i]][s[i + 1]]
        l += G[s[len(s) - 1]][s[0]]
    return l


def get_elite(G, gen, k):
    gen = sorted(gen, key=lambda s: compute_fitness(G, s))
    return gen[:k]


def TSP_GA(G, k=20, ntrial=200):
    n_p = k
    mutation_prob = 0.1
    gen = []
    path = list(range(len(G)))
    while len(gen) < n_p:
        path1 = path.copy()
    np.random.shuffle(path1)
    if not path1 in gen:
        gen.append(path1)

    for trial in range(ntrial):
        gen = get_elite(G, gen, k)
    gen_costs = [(round(compute_fitness(G, s), 3), s) \
                 for s in gen]
    next_gen = []
    for i in range(len(gen)):
        for j in range(i + 1, len(gen)):
            c1, c2 = do_crossover(gen[i], gen[j], \
                                  np.random.randint(0, len(gen[i])))
            next_gen.append(c1)
            next_gen.append(c2)
        if np.random.rand() < mutation_prob:
            m = np.random.randint(0, len(gen[i]))
            while True:
                n = np.random.randint(0, len(gen[i]))
        if m != n:
            break
        c = do_mutation(gen[i], m, n)
        next_gen.append(c)


gen = next_gen