import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand



def init_J(N):
        J = rand.choice([-1,1], size = N) #Initialize random couplers
        return J
    
def energychange(spin, j_L,J_R,spin_L,spin_R):
        dE = 2*(j_L*spin*spin_L+J_R*spin*spin_R)
        return dE
    
def MCMH(out_size, N, J, beta = 1.0, verbose = False):
    """
    Performs the Metropolis-Coupled Markov Chain (MCMH) algorithm.

    :param out_size: The desired output size.
    :param N: The size of the state space.
    :param J: Coupling matrix.
    :param beta: Inverse temperature parameter, default is 1.0.
    :param verbose: Flag to enable verbose output, default is False.
    :return: Array of accepted states.
    """    
     # Initialize state randomly
    init_state = rand.choice([-1,1], size = N)
    
    # Verbose output for initial state and coupling matrix
    if verbose:
        print('J:',J)
        print('init state: ',init_state)
    
    accepted = np.zeros(N)
    # Loop until enough states are accepted
    while (accepted.size/N) <= out_size:
        
        index = rand.randint(N)
        
        n = init_state[index]
        # Get the current state and its left and right neighbors
        n_Left = init_state[(index-1)%N]
        J_Left = J[(index-1)%N]
        n_Right = init_state[(index+1)%N]
        J_Right = J[index]
        # Calculate the energy change for the potential state flip
        dE = energychange(n,J_Left,J_Right,n_Left,n_Right)
        
        # Accept or reject the new state based on Metropolis criteria
        if dE < 0:
            new_state = init_state
            new_state[index] *= -1
            accepted = np.concatenate((accepted,new_state))
            
            if verbose:
                print('New state: {} dE: {}'.format(new_state,dE))
        elif rand.random() < np.exp(-beta*dE):
            new_state = init_state
            new_state[index] *= -1
            
            accepted = np.concatenate((accepted,new_state))
            if verbose:
                print('New state: {} dE: {}'.format(new_state,dE))
        else:
            pass
    # Reshape the accepted states array and return it    
    return accepted.reshape((-1,N))[1:]


def expected(data):
    _ ,N = data.shape
    expected = np.zeros(N)
    for n in range(N):
        next_n = (n + 1) % N  # Get the index of the next element, wrapping around for the last element
        pairwise_product = data[:, n] * data[:, next_n]
        expected[n] = np.mean(pairwise_product)
    
    return expected



def calculateFVBM(matrix_data, batch_sz, seed, num_iterations, scaling_factor=1, lr=0.1):
    """
    Function to calculate FVBM.
    
    :param matrix_data: Input data matrix.
    :param batch_sz: Size of each batch.
    :param seed: Random seed for reproducibility.
    :param num_iterations: Number of iterations to run.
    :param scaling_factor: Beta scaling factor, default is 1.
    :param lr: Learning rate for the algorithm, default is 0.1.
    :return: Tuple of J matrix and KL divergence list.
    """

    random_generator = np.random.default_rng(seed=seed)
    total_rows, total_cols = matrix_data.shape
    total_batches = int(total_rows / batch_sz)
    initial_j_matrix = init_J(total_cols)
    j_matrix = np.zeros((num_iterations + 1, total_cols))
    j_matrix[0, :] = initial_j_matrix

    # Randomize order of data
    random_generator.shuffle(matrix_data)

    # Split data into batches
    data_batches = np.array_split(matrix_data, total_batches)
    kl_divergences = []

    for iteration in range(num_iterations):
        current_j = j_matrix[iteration, :]

        for batch_index in range(total_batches):
            current_batch_size, _ = data_batches[batch_index].shape
            expected_true = expected(data_batches[batch_index])
            generated_data = MCMH(current_batch_size, total_cols, current_j)
            expected_generated = expected(generated_data)

            kl_div = np.mean(-(expected_true * current_j - expected_generated * current_j))
            gradient = expected_true - expected_generated

            updated_j = current_j + lr * gradient
            j_matrix[iteration + 1, :] = updated_j

        kl_divergences.append(kl_div)
        print(f'Epoch {iteration + 1} of {num_iterations}')

    return j_matrix, kl_divergences