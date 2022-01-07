import numpy as np
import math
from scipy.linalg import eig
from scipy.stats import norm

def Forward_Probability( emission_data , states , initial_probabilities , transition_matrix , emission_probabilities  ):

    observation_rows = len(emission_data)

    alpha_probabilities = [[0 for x in range( len(states) )] for y in range( observation_rows ) ]

    alpha_probabilities[0] = [ initial_probabilities[i] * emission_probabilities[i][0] for i in states]

    for t in range( 1 , observation_rows):
        for y in states:
            probability = 0

            for y0 in states:
                probability += alpha_probabilities[t-1][y0] * transition_matrix[y0][y]
            
            alpha_probabilities[t][y] = probability * emission_probabilities[y][t]

        # Normalize current probabilities, so that values won't be too low over time
        alpha_probabilities[t] /= np.sum(alpha_probabilities[t])
    
    return alpha_probabilities

def Backward_Probability( emission_data , states , initial_probabilities , transition_matrix , emission_probabilities ):

    observation_rows = len(emission_data)
    beta_probabilities = [[0 for x in range( len(states) )] for y in range( observation_rows ) ]

    beta_probabilities[observation_rows-1] = [1.0 for i in states]

    for t in reversed(list(range(observation_rows))[1:]):
        
        for y in states:
            probability = 0.0

            for y0 in states:
                
                probability += beta_probabilities[t][y0] * transition_matrix[y][y0] * emission_probabilities[y0][t]
            
            beta_probabilities[t-1][y] = probability
        
        beta_probabilities[t-1] /= np.sum(beta_probabilities[t-1])
    
    return beta_probabilities


def Baum_Welch( emission_data , states , initial_probability , transition_matrix , emission_probabilities ):

    observation_rows = len(emission_data)

    alpha_probabilities = Forward_Probability(emission_data , states , initial_probability , transition_matrix , emission_probabilities )
    beta_probabilities = Backward_Probability(emission_data , states , initial_probability , transition_matrix , emission_probabilities )


    pi_star = np.multiply( alpha_probabilities , beta_probabilities )

    for t in range(observation_rows):
        pi_star[t] /= np.sum(pi_star[t])

    
    probability_of_observation = sum(initial_probability[:] * beta_probabilities[0][:])

    pi_double_star = [[[0 for i in range(len(states))] for j in range( len(states) )] for k in range( observation_rows ) ]

    for t in range( observation_rows - 1):
        
        denominator = sum( alpha_probabilities[t][:] * beta_probabilities[t][:])


        for y in states:
            for y0 in states:
                
                pi_double_star[t][y][y0] = alpha_probabilities[t][y] * transition_matrix[y][y0] * emission_probabilities[y0][t+1] * beta_probabilities[t+1][y0]
    
    gamma = [[ 0 for y in range(len(states))] for x in range(observation_rows)]

    for t in range(observation_rows-1):
        for s in states:
            gamma[t][s] = np.sum(pi_double_star[t][s][:])

    new_initial_probability = gamma[0] / np.sum(gamma[0])

    new_transition_matrix = [[0 for x in range(len(states))] for y in range(len(states)) ]

    for y in states:
        for y0 in states:
            new_transition_matrix[y][y0] = np.sum( pi_double_star[:][y][y0] ) / np.sum( gamma[:][y] ) / probability_of_observation
    

    means = [ 0 for x in range(len(states) ) ]
    variances = [0 for x in range(len(states) ) ]

    sums_pi_star = sum(pi_star[:])

    for t in range(observation_rows):
        
        for s in states:
            means[s] += pi_star[t][s] * emission_data[t]
    
    means[:] /= sums_pi_star[:]
    print(means)

    return 0

def Another_Baum_Welch( emission_data , states , initial_probability , transition_matrix , emission_probabilities ):
    observation_rows = len(emission_data)

    alpha_probabilities = Forward_Probability(emission_data , states , initial_probability , transition_matrix , emission_probabilities )
    beta_probabilities = Backward_Probability(emission_data , states , initial_probability , transition_matrix , emission_probabilities )


    pi_star = np.multiply( alpha_probabilities , beta_probabilities )

    for t in range(observation_rows):
        pi_star[t] /= np.sum(pi_star[t])
    
        

    means = [ 0 for x in range(len(states) ) ]
    variances = [0 for x in range(len(states) ) ]

    sums_pi_star = sum(pi_star[:])

    for t in range(observation_rows):
        means[:] += pi_star[t][:] * emission_data[t]

    
    means[:] /= sums_pi_star[:]

    for t in range(observation_rows):
        for s in states:
            variances[s] += pi_star[t][s] * (( emission_data[t] - means[s] ) ** 2)

    variances[:] /= sums_pi_star[:]
    variances = np.sqrt(variances)


def Viterbi( emission_data , states , initial_prob , transition_matrix , emission_probabilities ):
    
    V = [[0 for x in range(len(states))] for y in range(len(emission_data))]
    path = [[0 for x in range(len(emission_data))] for y in range(len(states))]

    for s in states:
        V[0][s] = initial_prob[s] * emission_probabilities[s][0]
        path[s][0] = s
    
    
    for t in range( 1 , len( emission_data ) ):
        new_path = [[0 for x in range(len(emission_data))] for y in range(len(states))]

        for y in states:

            max_probability = -1
            
            for y0 in states:

                nprob = V[t-1][y0] * transition_matrix[y0][y] * emission_probabilities[y][t]

                if nprob > max_probability:
                    max_probability = nprob
                    state = y0
                    V[t][y] = max_probability

                    new_path[y][0:t+1] = path[state][0:t+1]

                    new_path[y][t] = y

        V[t] /= np.sum(V[t]) # normalizing to prevent the probabilities from getting too low

        path = new_path
    
    probability = -1
    state = 0

    for y in states:
        if V[len(emission_data)-1][y] > probability:
            probability = V[len(emission_data)-1][y]
            state = y
    return path[state]


def Emission_Probability( emission_data , means , variances ):
    final = []

    for i in range( len(variances) ):
        emission_probability = norm.pdf(emission_data , means[i] , math.sqrt(variances[i]))
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

def Write_Data( states , file ):
    with open( file , 'w') as f:
        for i in range(len(states)):
            f.write(str( '"La Nina"' if states[i] == 1 else '"El Nino"' ) + '\n')

if __name__=='__main__':
    states , transition_matrix , gaussian_means , variance = Read_Parameters('./Input/parameters.txt')
    emission_data = Read_Data('./Input/data.txt')

    initial_prob = Stationary_Distribution( transition_matrix )

    emission_probabilities = Emission_Probability( emission_data , gaussian_means , variance )


    # Viterbi without learning
    # sequence = Viterbi( emission_data , states , initial_prob , transition_matrix , emission_probabilities  )
    # Write_Data(sequence , './Output/viterbi_without_learning.txt' )

    Another_Baum_Welch(emission_data , states , initial_prob , transition_matrix , emission_probabilities )
