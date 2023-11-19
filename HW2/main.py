import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand
import time


def dataimporter(file):
    data = np.loadtxt(file, dtype=str)
    N = len(data[0])
    array = np.zeros([len(data), N], dtype=int)
    
    for i, row in enumerate(data):
        array[i] = [1 if spin == '+' else -1 for spin in row]
    return array


def Partition_func(data,beta = 1, ones = True):
    M,N = data.shape   
    if ones:
        J = np.ones(N)
        
    else:
        J = rand.choice([-1,1], size = N) #initialize random couplers
    Z = np.power(2,N)*np.power(np.cosh(-beta),N-1)
    return Z, J


def Probability(data, Z, J, beta=1):
    """
    Calculate the probability distribution of data configurations.

    Args:
        data (numpy.ndarray): Input data.
        Z (float): Partition function.
        J (numpy.ndarray): Coupling matrix.
        beta (float, optional): Inverse temperature parameter. Default is 1.

    Returns:
        list: A list containing the probabilities of each data configuration.
    """
    M, N = data.shape
    probabilities = []

    for m in range(M):
        En = 0

        for n in range(N - 1):
            En += -J[n] * data[m, n] * data[m, n + 1]

        En += -J[N - 1] * data[m, -1] * data[m, 0]

        Pn = np.exp(-beta * En) / Z
        probabilities.append(Pn)

    return probabilities




    
def init_J(N):
        J = rand.choice([-1,1], size = N) #initialize random couplers
        return J
    
def energychange(spin, j_L,J_R,spin_L,spin_R):
        dE = 2*(j_L*spin*spin_L+J_R*spin*spin_R)
        return dE
    
def MCMH(out_size, N, J, beta = 1.0, verbose = False):
        
        init_state = rand.choice([-1,1], size = N)
        if verbose:
            print('J:',J)
            print('init state: ',init_state)
        
        accepted = np.zeros(N)
        while (accepted.size/N) <= out_size:
            
            index = rand.randint(N)
            
            n = init_state[index]
            
            nleft = init_state[(index-1)%N]
            jleft = J[(index-1)%N]
            nright = init_state[(index+1)%N]
            J_Right = J[index]
            dE = energychange(n,jleft,J_Right,nleft,nright)
            
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



data = dataimporter('data/in.txt')
t0 = time.time()
J= np.array([-1,1,1,1])
J_new, KL = calculateFVBM(data,100,13,100, lr = 0.1)

# Refactor of the for loop for updating elements in J_new
final_j_vector = J_new[-1]  # Extracting the last row of J_new for processing

# Vectorized approach for conditionally updating elements
final_j_vector[final_j_vector < 0] = -1
final_j_vector[final_j_vector >= 0] = 1

# Assigning the updated vector back to the last row of J_new
J_new[-1] = final_j_vector

        
        
print(f"Coupling  of generated data set: {J_new[-1]}")
print(f"Coupling  of Original data set: {J}")
print(np.array_equal(J,J_new[-1]))
plt.plot(KL)
plt.xlabel('Epoch')  # Label for x-axis
plt.ylabel('KL Divergence')  # Label for y-axis
plt.title('KL Divergence over Epochs')  # Title for the plot
plt.show()



