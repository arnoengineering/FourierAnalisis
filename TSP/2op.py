import numpy as np
import queue    def dfs(adj, x):
   visited = [False]*len(adj)
   stack = [x]
   visited[x] = True
   path = []
   while len(stack) > 0:
      u = stack.pop(-1)
      path.append(u)
      for v in adj[u]:
     if not visited[v]:
        stack.append(v)
        visited[v] = True
    return pathdef mst(adj):
   inf = np.inf
   c = [inf]*n
   s = 0
   c[s] = 0
   visited = [False]*n
   parent = [None]*n
   h = queue.PriorityQueue()
   for v in range(n):
      h.put((c[v], v))
   edges = []
   while not h.empty():
      w, u = h.get()
      if visited[u]: continue
           visited[u] = True
       if parent[u] != None:
        edges.append((parent[u], u))
        for v in range(n):
               if v == u: continue
           if (not visited[v]) and (c[v] > adj[u][v]):
            c[v] = adj[u][v]
            parent[v] = u
            h.put((c[v], v))
   adj = [[] for _ in range(n)]
   for i in range(n):
      if parent[i] != None:
           adj[parent[i]].append(i)
   path = dfs(adj, 0)
   path += [path[0]]
   return path