import numpy as np

def getPolMatrix(n=1e3,R=10,p=1):
    diag_elements = np.concatenate((np.ones(R), np.power(np.arange(2, n - R + 2, dtype=float), -p)))
    A = np.diag(diag_elements)
    return A
    
def getExpMatrix(n=1e3,R=10,q=0.25):
    diag_elements = np.concatenate((np.ones(R), np.power(10, -q * np.arange(1, n - R + 1))))    
    A = np.diag(diag_elements)
    return A

def getExpMatrixBloc(i,j,p,n=1e3,R=10,q=0.25): #Get the correspondent bloc
    assert(n%np.sqrt(p)==0)
    bloc_size = int(n/ np.sqrt(p))
    if(i!=j):
        return np.zeros((bloc_size,bloc_size))
    if i==0:
        diag_elements = np.concatenate((np.ones(R), np.power(10, -q * np.arange(1, bloc_size - R + 1))))    
    else:
        diag_elements = np.power(10,-q*np.arange(i*bloc_size - R+1,(i+1)*bloc_size - R+1) )

    return np.diag(diag_elements)