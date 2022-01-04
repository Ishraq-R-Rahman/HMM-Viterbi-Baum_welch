import numpy as np
from scipy.linalg import eig 

def Stationary_Distribution( transition_matrix ):
    evals , evecs = eig( transition_matrix.T )
    evec1 = evecs[:,np.isclose(evals , 1 ) ]

    evec1 = evec1[:,0]

    stationary = evec1 / evec1.sum()
    stationary = stationary.real

    return stationary

def Read_Parameters( file ):
    with open(file) as f:
        hidden_states = int( next(f) )

        mat = []

        for i in range( hidden_states ):
            line = next(f)
            mat.append([float(x) for x in line.split()])
        
        gaussian_means = [float(x) for x in next(f).split()]
        variance = [float(x) for x in next(f).split()]
        transition_matrix = np.matrix( mat )

    return transition_matrix , gaussian_means , variance

if __name__=='__main__':
    transition_matrix , gaussian_means , variance = Read_Parameters('./Input/parameters.txt')

    initial_prob = Stationary_Distribution( transition_matrix )

    print( initial_prob )
    

    

    