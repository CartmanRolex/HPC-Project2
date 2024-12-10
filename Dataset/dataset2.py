import numpy as np

n = 10 ** 3
R = 5 # R = 5, 10, 20
xi = 10e-1 # signal-to-noise ratio, 10e-1, 10e-2 10e-4

def mnist_Data ():
    # TODO : this function will process mnist data.
    pass 

def polynomial_Decay ():
    # Polynomial Decay.
    p = 0.5 # p = 0.5, 1, 2 
    matrix2 = np.zeros((n, n)) 
    diagonalList = []
    for i in range(R):
        diagonalList.append(1) 
    for i in range(2, n - R + 2):
        diagonalList.append(i ** (- p))

    np.fill_diagonal(matrix2, diagonalList)
    A2 = matrix2
    print(f"diagonalList is \n{diagonalList}\n")
    print(f"A2 is \n{A2}\n")


def exponential_Decay ():
    # Exponential Decay.
    q = 0.1 # q > 0 controls the rate of exponential decay, q = 0.1,0.25,1
    diagonalList2 = []
    for i in range(R):
        diagonalList2.append(1)
    for i in range(1, n - R + 1):
        diagonalList2.append(10 ** ((-1) * i * q))
    A3 = np.zeros((n, n))
    np.fill_diagonal(A3, diagonalList2)

    print(f"diagonalList2 is \n{diagonalList2}\n")
    print(f"A3 is \n{A3}\n")