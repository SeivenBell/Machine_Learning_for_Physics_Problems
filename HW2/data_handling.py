import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand


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
        J = rand.choice([-1,1], size = N) #Initialize random couplers
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