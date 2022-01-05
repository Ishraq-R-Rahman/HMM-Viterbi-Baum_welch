import numpy as np
import math
from scipy.linalg import eig
from scipy.stats import norm
from sklearn.preprocessing import normalize

def Viterbi( emission_data , states , initial_prob , transition_matrix , emission_probabilties ):
    V = [[0 for x in range(len(states))] for y in range(len(emission_data))]
    path = [[0 for x in range(len(emission_data))] for y in range(len(states))]

    for s in states:
        V[0][s] = initial_prob[s] * emission_probabilties[s][0]
        path[s][0] = s
    
    
    for t in range( 1 , len( emission_data ) ):
        new_path = [[0 for x in range(len(emission_data))] for y in range(len(states))]

        for y in states:

            max_probability = -1
            
            for y0 in states:
                nprob = V[t-1][y0] * transition_matrix[y0][y] * emission_probabilties[y][t]
                
                if nprob > max_probability:
                    max_probability = nprob
                    state = y0
                    V[t][y] = max_probability

                    new_path[y][0:t+1] = path[state][0:t+1]

                    new_path[y][t] = y

        path = new_path
    
    probability = -1
    state = 0

    for y in states:
        if V[len(emission_data)-1][y] > probability:
            probability = V[len(emission_data)-1][y]
            state = y
    
    print(path[state])
    return path[state]


def Emission_Probability( emission_data , means , variances ):
    final = []

    for i in range( len(variances ) ):
        emission_probability = norm.pdf(emission_data , means[i] , math.sqrt(variances[i]))
        # normal = np.linalg.norm(emission_probability)
        # normal_array = emission_probability / normal
        final.append( emission_probability )
    
    return final


def Stationary_Distribution( transition_matrix ):
    evals , evecs = eig( np.matrix(transition_matrix).T )
    evec1 = evecs[:,np.isclose(evals , 1 ) ]

    evec1 = evec1[:,0]

    stationary = evec1 / evec1.sum()
    stationary = stationary.real

    return stationary

def Read_Parameters( file ):
    with open(file) as f:
        hidden_states = int( next(f) )

        states = [i for i in range(hidden_states)]

        mat = []

        for i in range( hidden_states ):
            line = next(f)
            mat.append([float(x) for x in line.split()])
        
        gaussian_means = [float(x) for x in next(f).split()]
        variance = [float(x) for x in next(f).split()]
        

    return states , mat , gaussian_means , variance

def Read_Data( file ):
    emission_data = []
    with open( file ) as f:
        for line in f:
            emission_data.append(float(line))
    return emission_data


if __name__=='__main__':
    states , transition_matrix , gaussian_means , variance = Read_Parameters('./Input/parameters.txt')
    emission_data = Read_Data('./Input/data.txt')

    initial_prob = Stationary_Distribution( transition_matrix )

    emission_probabilities = Emission_Probability( emission_data , gaussian_means , variance )

    # emission_data = [0,1,2]
    # states = [0,1]
    # initial_prob = [0.6,0.4]
    # transition_matrix = [[0.7,0.3],[0.4,0.6]]
    # emission_probabilities = [[0.1,0.4,0.5],[0.6,0.3,0.1]]


    Viterbi( emission_data , states , initial_prob , transition_matrix , emission_probabilities  )

    

    
    

    

    