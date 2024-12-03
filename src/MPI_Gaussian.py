from dataset1 import *
import numpy as np
from scipy.linalg import solve_triangular
from numpy.linalg import eigh
import torch
from hadamard_transform import hadamard_transform
import time
from mpi4py import MPI

def generate_gaussian_matrix(seed, shape):
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape)

# Initialize the MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()   # Process rank (ID)
size = comm.Get_size()    # Total number of processes
sqrt_p = int(np.sqrt(size))
n = 4
assert(n % sqrt_p == 0)
assert(sqrt_p**2 == size) #Assert perfect square
bloc_size = int(n/sqrt_p)



# Création d'une topologie cartésienne (sqrt_p x sqrt_p)
dims = [sqrt_p, sqrt_p]
periods = [False, False]  # Pas de périodicité
reorder = True
cart_comm = comm.Create_cart(dims, periods=periods, reorder=reorder)

# Obtenir les coordonnées (i, j) de chaque processeur
coords = cart_comm.Get_coords(rank)
row, col = coords
A_ij = getExpMatrixBloc(row,col,size,n=n,R=2)
good_conditionned = False
# Créer des communicateurs pour les lignes et les colonnes
row_comm = cart_comm.Sub([True, False])  # Communicateur pour la ligne (True pour la direction ligne)
col_comm = cart_comm.Sub([False, True])  # Communicateur pour la colonne (True pour la direction colonne)

# Le processeur de rang 0 génère des graines et les envoie
if rank == 0:

    base_seed_sequence = np.random.SeedSequence(12345)
    # Générer \sqrt{P} sous-séquences indépendantes
    sub_sequences = base_seed_sequence.spawn(sqrt_p)
    seed_values = [int(seq.generate_state(1)[0]) for seq in sub_sequences]
else:
    seed_values = None

seed_values = comm.bcast(seed_values, root=0)

# Chaque processus attribue la graine de sa ligne et de sa colonne
row_seed = seed_values[row]
col_seed = seed_values[col]
# Chaque processeur génère les matrices localement

omega_i = generate_gaussian_matrix(row_seed, (bloc_size, bloc_size))
omega_j = generate_gaussian_matrix(col_seed, (bloc_size, bloc_size))

C_ij = A_ij@omega_j
B_ij = omega_i.T@C_ij
C_j = None
if(row_comm.Get_rank()==0):
    C_j = np.empty(C_ij.shape)
row_comm.Reduce(C_ij,C_j,root=0)
B = None
if(rank==0):
    B=np.empty(B_ij.shape)
comm.Reduce(B_ij,B,root=0)

if good_conditionned:
    # Perform Cholesky decomposition of B
    L = np.linalg.cholesky(B)  # L is a lower triangular matrix
    # Solve for Z using the triangular solver
    Z_j = solve_triangular(L, C_j.T, lower=True).T  # Z = C @ L^-T --> ZL^T  = C -- > LZ^T = C^T
    comm.Gather(Z_j)

else: #if not good conditionned
    # Eigenvalue decomposition of B
    eig_v, V = eigh(B)  # eig_v: Eigenvalues, V: Eigenvectors
    # Truncate eigenvalues and eigenvectors to keep only positive eigenvalues
    truncate = eig_v > 0
    eig_v = eig_v[truncate]
    V = V[:, truncate]
    # Normalize using the eigenvalues to construct Z
    Z = C @ V @ np.diag(1 / np.sqrt(eig_v))  # Normalize columns of C
    # Perform QR decomposition of Z
    Q, R = np.linalg.qr(Z)  # Q: Orthogonal basis, R: Upper triangular matrix

    # Singular Value Decomposition (SVD) of R
    U, Sigma, Vh = np.linalg.svd(R)

    # Construct the low-rank approximation
    U_hat = Q @ U[:, :rank]  # Truncate U to the desired rank
    A_nyst_k = U_hat @ np.diag(Sigma[:rank]**2) @ U_hat.T  # Final approximation

