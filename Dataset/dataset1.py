import numpy as np

getPolMatrix(n=1e3,R=10,p=1):
    diag_elements = np.concatenate((np.ones(R), np.power(np.arange(2, n - R + 2, dtype=float), -p)))
    A = np.diag(diag_elements)
    return A
    
getExpMatrix(n=1e3,R=10,q=0.25):
    diag_elements = np.concatenate((np.ones(R), np.power(10, -q * np.arange(1, n - R + 1))))    
    A = np.diag(diag_elements)
    return A