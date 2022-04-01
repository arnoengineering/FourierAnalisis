def dist(P, i, j):
    return np.sqrt((P[i][0] - P[j][0]) ** 2 + (P[i][1] - P[j][1]) ** 2)


def BTSP(P):
    n = len(P)
    D = np.ones((n, n)) * np.inf
    path = np.ones((n, n), dtype=int) * (-1)
    D[n - 2, n - 1] = dist(P, n - 2, n - 1)
    path[n - 2, n - 1] = n - 1
    for i in range(n - 3, -1, -1):
        m = np.inf
        for k in range(i + 2, n):
    if m > D[i + 1, k] + dist(P, i, k):
        m, mk = D[i + 1, k] + dist(P, i, k), k
    D[i, i + 1] = m
    path[i, i + 1] = mk
    for j in range(i + 2, n):


D[i, j] = D[i + 1, j] + dist(P, i, i + 1)
path[i, j] = i + 1
D[0, 0] = D[0, 1] + dist(P, 0, 1)
path[0, 0] = 1
return D, path


def get_tsp_path(path, i, j, n):
    if n < 0:
        return []
    if i <= j:
        k = path[i, j]
    return [k] + get_tsp_path(path, k, j, n - 1)
    else:
    k = path[j, i]
    return get_tsp_path(path, i, k, n - 1) + [k]