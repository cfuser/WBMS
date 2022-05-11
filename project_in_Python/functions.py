from ast import Lambda
import matplotlib
import math
import torch
import numpy as np

def K(x, y, w, h):
    # print(-torch.square(x - y) * w)
    # print(-torch.sum(torch.square(x - y) * w))
    # exit()
    return torch.exp(-torch.sum(torch.square(x - y) * w) / h)


def WBMS(X, h, _lambda = 1, tmax = 30):
    # print(id(X))
    n = X.shape[0]
    p = X.shape[1]
    K_matrix = torch.zeros(n, n)
    w = torch.full([p], 1 / p)
    # print(w)
    D = torch.zeros(p)
    X1 = X.clone()
    X2 = X.clone()
    for t in range(20):
        for i in range(n):
            for j in range(i, n):
                K_matrix[i, j] = K(X2[i, :], X2[j, :], w, h)

        for i in range(n):
            for j in range(i, n):
                K_matrix[j, i] = K_matrix[i, j]
        # print(K_matrix)
        # exit()

        for i in range(n):
            I = list(range(0, n))
            del I[i]
            s = torch.sum(K_matrix[I, i])
            for l in range(p):
                X1[i, l] = torch.sum(X2[I, l] * K_matrix[I, i])
            X1[i, :] = X1[i, :] / s

        D = torch.sum(torch.square(X - X1), dim = 0)
        w = torch.exp(-D / _lambda)
        w = w / torch.sum(w)
        X2 = X1.clone()
    # print('Turn One : ', X2)
    X1 = X.clone()
    X2 = X.clone()
    # print(id(X1))
    # print(id(X2))
    # print(X2)
    # exit()
    for t in range(tmax):
        for i in range(n):
            for j in range(i, n):
                K_matrix[i, j] = K(X2[i, :], X2[j, :], w, h)
        
        for i in range(n):
            for j in range(i, n):
                K_matrix[j, i] = K_matrix[i, j]
        # print(K_matrix)
        # exit()
        for i in range(n):
            I = list(range(0, n))
            del I[i]
            s = torch.sum(K_matrix[I, i])
            for l in range(p):
                # print(X2[I, l].shape)
                # print(K_matrix[I, i].shape)
                # print((X2[I, l] * K_matrix[I, i]).shape)
                # exit()
                X1[i, l] = torch.sum(X2[I, l] * K_matrix[I, i])
            X1[i, :] = X1[i, :] / s

        D = torch.sum(torch.square(X - X1), dim = 0)
        w = torch.exp(-D / _lambda)
        w = w / torch.sum(w)
        X2 = X1.clone()
    # print(X2)
    # print(id(X1))
    # print(id(X2))
    return X2, w

class graph_components:
    def __init__(self):
        pass
    def tr(self, G):
        #初始化翻转边的图GT
        GT = dict()
        for u in G.keys():
            GT[u] = GT.get(u,set())
        #翻转边
        for u in G.keys():
            for v in G[u]:
                GT[v].add(u)
        return GT

    #获取按节点遍历完成时间递减排序的顺序
    def topoSort(self, G):
        res=[]
        S=set()
        #dfs遍历图
        def dfs(G,u):
            if u in S:
                return
            S.add(u)
            for v in G[u]:
                if v in S:
                    continue
                dfs(G,v)
            res.append(u)
        #检查是否有遗漏的节点
        for u in G.keys():
            dfs(G,u)
        #返回拓扑排序后的节点列表
        res.reverse()
        return res

    #通过给定的起始节点，获取单个连通量
    def walk(self, G, s, S = None):
        if S is None:
            s = set()
        Q = []
        P = dict()
        Q.append(s)
        P[s] = None
        while Q:
            u = Q.pop()
            for v in G[u]:
                if v in P.keys() or v in S:
                    continue
                Q.append(v)
                P[v] = P.get(v,u)
        #返回强连通图
        return P

    def build_graph(self, A):
        G = dict()
        for i in range(A.shape[0]):
            # print(A[i, :])
            nonzero_idx = torch.nonzero(A[i, :])
            nonzero_idx = torch.squeeze(nonzero_idx, dim = 1)
            # print(nonzero_idx.numpy().tolist())
            G[i] = set(nonzero_idx.numpy().tolist())
        
        return G

    def components(self, G):
        #记录强连通分量的节点
        seen = set()
        #储存强强连通分量
        scc = []
        GT = self.tr(G)
        for u in self.topoSort(G):
            if u in seen :
                continue
            C = self.walk(GT,u,seen)
            seen.update(C)
            scc.append(sorted(list(C.keys())))

        print('scc', scc)
        return scc

    def build_set(self, clu):

        res = dict()
        id = 0
        for single_component in clu:
            for j in single_component:
                res[j] = id
            id += 1
        print('res', res)
        return res


def U2clus(U, epsa = 1e-5):
    n = U.shape[0]
    A = torch.zeros(n, n)
    print(U)
    for i in range(n):
        for j in range(n):
            # print(U[i, :])
            # print(U[j, :])
            # print(torch.norm(U[i, :] - U[j, :]))
            if (torch.norm(U[i, :] - U[j, :]) < epsa):
                A[i, j] = 1
    graph_component_solver = graph_components()
    g = graph_component_solver.build_graph(A)
    clu = graph_component_solver.components(g)
    res = graph_component_solver.build_set(clu)
    return res