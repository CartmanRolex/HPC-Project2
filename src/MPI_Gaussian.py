from dataset1 import *
import numpy as np
from scipy.linalg import solve_triangular
import scipy
from numpy.linalg import eigh
from mpi4py import MPI
import math

###################################
# PARAMETERS (Change these as needed)
###################################

n = 1024  # Size of the square matrix A (must be divisible by sqrt_p)
Rate = 10  # Parameter for the exponential matrix generation (e.g., rate parameter)
good_conditionned = False  # Whether to assume B is well-conditioned (for Cholesky vs. eigenvalue decomposition)
sketch_size = 128  # Sketch size (number of columns in the random sketch matrices)
rank_k = 128 # Target rank for low-rank approximation

#####################
## FUNCTIONS
#####################

def getError(A, B):
    """Compute the normalized nuclear norm of the difference between A and B."""
    return np.linalg.norm(A - B, ord='nuc') / np.linalg.norm(A, ord='nuc')

def generate_gaussian_matrix(seed, shape):
    """Generate a Gaussian random matrix with the specified seed and shape."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape)

def getQR(A_local, comm):
    """
    Perform a distributed QR decomposition using the TSQR algorithm.

    Parameters:
    A_local : ndarray
        Local matrix on each process.
    comm : MPI communicator
        MPI communicator for communication between processes.

    Returns:
    Q_final : ndarray
        The final Q factor on each process.
    R_final : ndarray
        The final R factor (broadcasted to all processes).
    """
    local_rank = comm.Get_rank()
    m, n = A_local.shape
    size = comm.Get_size()
    logP = int(math.log(size, 2))  # Determine the number of levels in the binary tree
    assert 2**logP == size, "Number of processes must be a power of 2 for TSQR."

    # Initialize lists for local Q and R matrices at each level
    Q_local = [None] * (logP + 1)
    R_local = [None] * (logP + 1)

    # Perform QR decomposition on the local matrix A_local
    Q_local[logP], R_local[logP] = scipy.linalg.qr(
        A_local, mode='economic', check_finite=False
    )

    # Binary tree reduction to aggregate R matrices
    for k in range(logP - 1, -1, -1):  # From leaves to root
        # Determine if the process participates at this level
        if (local_rank) % (2 ** (logP - 1 - k)) != 0:
            break  # Non-participating processes exit the loop
        J = local_rank ^ (1 << (logP - 1 - k))  # Identify partner process using bitwise XOR
        if local_rank > J:
            # Send R to partner process
            comm.Send(R_local[k + 1], J, tag=k)
        else:
            # Receive R from partner process
            R_j = np.empty(R_local[k + 1].shape)
            comm.Recv(R_j, J, tag=k)
            # Compute QR of concatenated R matrices
            Q_local[k], R_local[k] = scipy.linalg.qr(
                np.concatenate((R_local[k + 1], R_j), axis=0),
                mode='economic',
                check_finite=False
            )

    ##################
    # Backward Pass: Reconstructing the Q matrix
    ##################

    # Initialize the final local Q matrix as identity (size n x n)
    Q_final = np.eye(n)

    # Traverse back down the binary tree to compute Q
    for k in range(0, logP):
        partner = local_rank ^ (1 << (logP - 1 - k))  # Compute partner rank at this level
        if (local_rank) % (2 ** (logP - 1 - k)) == 0:  # Filter processes that are needed at this level
            if local_rank < partner:
                # Partition Q_local[k] into top and bottom parts
                Q_top = Q_local[k][:n, :]        # Top half
                Q_bottom = Q_local[k][n:, :]     # Bottom half
                # Send Q_bottom @ Q_final to partner
                comm.Send(Q_bottom @ Q_final, dest=partner, tag=k + 200)
                # Update local Q_final with Q_top
                Q_final = Q_top @ Q_final
            else:
                # Receive Q_final from partner
                comm.Recv(Q_final, source=partner, tag=k + 200)

    # Final multiplication to complete Q for each process
    Q_final = Q_local[logP] @ Q_final

    # Broadcast R_final to all processes
    if local_rank != 0:
        R_final = np.empty((n, n))
    else:
        R_final = R_local[0]
    comm.Bcast(R_final, root=0)  # Broadcast so every process gets R

    return Q_final, R_final

###################################
# START OF THE SCRIPT
###################################

# Initialize MPI communicator and get process rank and size
comm = MPI.COMM_WORLD
rank = comm.Get_rank()   # Process rank (ID)
size = comm.Get_size()    # Total number of processes
sqrt_p = int(np.sqrt(size))  # Size of the processors grid (assumed square)
assert sqrt_p ** 2 == size, "Number of processes must be a perfect square."

# Create a Cartesian topology (sqrt_p x sqrt_p)
dims = [sqrt_p, sqrt_p]
periods = [False, False]  # No periodicity in the topology
cart_comm = comm.Create_cart(dims, periods=periods, reorder=False)

# Get the (row, col) coordinates of each processor in the grid
coords = cart_comm.Get_coords(rank)
row, col = coords

# Ensure matrix dimension n is divisible by sqrt_p
assert n % sqrt_p == 0, "Matrix dimension n must be divisible by sqrt_p."
bloc_size = int(n / sqrt_p)

# Generate local block of matrix A
A_ij = getExpMatrixBloc(row, col, size, n=n, R=Rate)  # Local block of matrix A

# Create communicators for rows and columns
row_comm = cart_comm.Sub([False, True])  # Row communicator
col_comm = cart_comm.Sub([True, False])  # Column communicator

# Rank 0 generates seeds for all sub-blocks of Omega
if rank == 0:
    base_seed_sequence = np.random.SeedSequence(12345)  # Base seed for reproducibility
    # Generate sqrt_p independent subsequences for rows and columns
    sub_sequences = base_seed_sequence.spawn(sqrt_p)
    seed_values = [int(seq.generate_state(1)[0]) for seq in sub_sequences]
else:
    seed_values = None

# Broadcast seed values to all processes
seed_values = comm.bcast(seed_values, root=0)

# Each process assigns its row and column seeds based on its coordinates
row_seed = seed_values[row]
col_seed = seed_values[col]

# Each process generates its local random matrices omega_i and omega_j
omega_i = generate_gaussian_matrix(row_seed, (bloc_size, sketch_size))
omega_j = generate_gaussian_matrix(col_seed, (bloc_size, sketch_size))

# Local computations for matrix multiplication
C_ij = A_ij @ omega_j  # Sketch along the row direction
B_ij = omega_i.T @ C_ij  # Sketch along the column direction

# Aggregate sketches along rows
C_i = None
if col == 0:  # Processes in the first column
    C_i = np.empty(C_ij.shape, dtype='float64')
row_comm.Reduce(C_ij, C_i, root=0)  # Sum over rows; result stored in C_i at col == 0

# Aggregate sketches along columns
B_i = None
B = None
if col == 0:
    B_i = np.empty(B_ij.shape)
    B = np.empty(B_ij.shape)
row_comm.Reduce(B_ij, B_i, root=0)  # Sum over rows; result stored in B_i at col == 0

if col == 0:  # Work only with processes in the first column
    # Allreduce B_i along the column; result stored in B
    col_comm.Allreduce(B_i, B)  # B is reduced over the first column and stored on all processes in col == 0

    if good_conditionned:
        # Perform Cholesky decomposition of B (assumes B is positive definite)
        L = np.linalg.cholesky(B)  # Lower triangular matrix
        # Solve for Z using the triangular solver
        Z_i = solve_triangular(L, C_i.T, lower=True).T  # Compute Z such that Z L^T = C
    else:
        # Perform eigenvalue decomposition of B
        eig_v, V = eigh(B)  # Eigenvalues and eigenvectors of B
        # Retain only positive eigenvalues
        truncate = eig_v > 0
        eig_v = eig_v[truncate]
        V = V[:, truncate]
        # Normalize using the eigenvalues to construct Z
        Z_i = C_i @ (V * (1 / np.sqrt(eig_v)))  # Normalize columns of C

    # Perform QR decomposition of Z
    Q_i, R = getQR(Z_i, col_comm)  # Orthogonal basis Q_i, upper triangular matrix R

    # Perform Singular Value Decomposition (SVD) of R
    U, Sigma, Vh = np.linalg.svd(R)

    # Construct the low-rank approximation
    U_hat_i = Q_i @ U[:, :rank_k]  # Truncate U to the desired rank k

    # Collect U_hat_i from all processes along the column
    U_hats = np.empty((U_hat_i.shape[0] * sqrt_p, U_hat_i.shape[1]))
    col_comm.Allgather(U_hat_i, U_hats)  # U_hats contains all U_hat_i stacked vertically

    # Reconstruct the approximated matrix block A_nyst_i
    A_nyst_i = (U_hat_i * Sigma[:rank_k]**2) @ U_hats.T

    # Gather the approximated matrix blocks to the root process
    A_nyst = None
    if rank == 0:
        A_original = getExpMatrix(n=n, R=Rate)  # Generate the original matrix at root
        A_nyst = np.empty(A_original.shape)
    col_comm.Gather(A_nyst_i, A_nyst, root=0)

    if rank == 0:
        # Compute and print the error
        error = getError(A_original, A_nyst)
        print("Normalized nuclear norm error:", error)
