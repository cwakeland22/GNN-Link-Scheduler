import numpy as np
from itertools import permutations

def genData(N,num_samples):
    """
    Generate data samples for training and testing a GNN for 
    switch scheduling assuming a crossbar switch is used.

    Args:
        N: Number of i/p and o/p ports of the switch
        num_samples: Number of desired data samples 

    Returns:
        VOQ: (num_samples, N, N) array of the number of 
             packets in each virtual output queue
        M: (num_samples, N, N) array with the corresponding 
           matching according to the MaxWeight algoirthm 
    """
    VOQ = np.zeros((num_samples, N, N))
    M = np.zeros((num_samples, N, N))
    # Generate all permutation matrices of size N x N (represent a subset of the possible matchings)
    perms = permutations(np.eye(N))
    perm_matrices = np.zeros((np.math.factorial(N), N, N))
    for i, perm in enumerate(perms):
        perm_matrices[i] = np.array(perm).reshape(N, N)
        
    h = perm_matrices.shape[0]
    for i in range(num_samples):
        # Randomly initialize the # of packets in each VOQ (takes values b/w 0 & 5) for each iteration
        VOQ_i = np.random.randint(low=0, high=6, size=(N,N))
        hadQM = np.zeros((h, N, N))
        wsum = np.zeros(h)
        # MaxWeight algo.
        for j in range(h):
            # Hadamard product b/w VOQ matrix and (possible) matching matrix
            hadQM[j] = VOQ_i * perm_matrices[j]
            wsum[j] = np.sum(hadQM[j])
        
        M_i = hadQM[np.argmax(wsum)]
        M_i = np.where(M_i != 0, 1, M_i)
        
        VOQ[i] = VOQ_i
        M[i] = M_i
        
    return VOQ, M
    
    
    
# if __name__ == '__main__':
# #     Generate 1000 data samples (MaxWeight was used for scheduling a 5 x 5 crossbar switch)
#     VOQ, M = genData(5, 1000)
#     np.save('VOQ_samples.npy', VOQ)
#     np.save('Matching_samples.npy', M)