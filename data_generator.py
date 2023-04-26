import networkx as nx
import numpy as np
import random
import json
import os
from networkx.readwrite import json_graph
from itertools import permutations


def generate_graph(N, max_q):
    """
    Generate a graph representing a switch scheduling 
    scenario in an (N x N) crossbar switch
    
    Parameters
    ----------
    N: Number of i/p and o/p ports of the switch
    max_q: max possible queue length in the graph 
        
    Returns
    -------
    A labeled directed graph representing the optimal 
    matching according to the MaxWeight algorithm
    """
    # create an empty directed graph structure
    G = nx.DiGraph()

    VOQ = [f'voq{i}{j}' for i in range(1, N+1) for j in range(1, N+1)]
    DQ = [f'dq{i}' for i in range(1,N+1)]

    # add N*N + N nodes to the graph
    G.add_nodes_from(VOQ)
    G.add_nodes_from(DQ)
    
    # set node attributes (??)
    nx.set_node_attributes(G, 0, 'src_dest')
    for voq in VOQ:
        G.nodes[voq]['src_dest'] = 1
    nx.set_node_attributes(G, 0, 'sp')
    nx.set_node_attributes(G, 'node', 'entity')
    
    # define the graph's weighted directed edges 
    edge_list = [(voq, dq, random.randint(0, max_q)) for i in range(N) for voq, dq in zip(VOQ[i*N:(i+1)*N], DQ)]
    G.add_weighted_edges_from(edge_list)
    
    # generate all permutation matrices of size N x N (represent a subset of the possible matchings)
    perms = permutations(np.eye(N))
    perm_matrices = np.zeros((np.math.factorial(N), N, N))
    for i, perm in enumerate(perms):
        perm_matrices[i] = np.array(perm).reshape(N, N)
    h = perm_matrices.shape[0]
    
    VOQ_mat = np.zeros(N*N)
    hadQM = np.zeros((h, N, N))
    wsum = np.zeros(h)
    for i, edge in enumerate(G.edges(data=True)):
        VOQ_mat[i] = edge[2]['weight']
    VOQ_mat = VOQ_mat.reshape(N, N)
    
    # MaxWeight algo.
    for j in range(h):
        # Hadamard product b/w VOQ matrix and (possible) matching matrix
        hadQM[j] = VOQ_mat * perm_matrices[j]
        wsum[j] = np.sum(hadQM[j])    
    M = hadQM[np.argmax(wsum)]
    M = np.where(M != 0, 1, M)
    
    M = M.reshape(N*N)
    for i, voq in enumerate(VOQ):
        G.nodes[voq]['sp'] = M[i]
     
    return nx.DiGraph(G)


def generate_dataset(file_name, num_samples, N=5, max_q=5):
    samples = []
    for _ in range(num_samples):
        G = generate_graph(N, max_q)
        samples.append(json_graph.node_link_data(G))

    with open(file_name, "w") as f:
        json.dump(samples, f)

        
root_dir = "./data"
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
if not os.path.exists(root_dir + "/train"):
    os.makedirs(root_dir + "/train")
if not os.path.exists(root_dir + "/validation"):
    os.makedirs(root_dir + "/validation")
if not os.path.exists(root_dir + "/test"):
    os.makedirs(root_dir + "/test")

generate_dataset("./data/train/data.json", 1000)
generate_dataset("./data/validation/data.json", 200)
generate_dataset("./data/test/data.json", 200)