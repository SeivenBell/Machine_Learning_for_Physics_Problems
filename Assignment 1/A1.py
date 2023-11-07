import random
import numpy as np

def generate_binary_numbers(max_digits=8):
    return np.random.randint(0, 2, max_digits).tolist()

def one_hot_encode(binary_number):
    # Assuming binary_number is a list of integers (0 or 1)
    return np.eye(2)[binary_number]

def generate_dataset(train_size, test_size, seed):
    random.seed(seed)
    training_data = []
    testing_data = []
    for _ in range(train_size + test_size):
        a = generate_binary_numbers()
        b = generate_binary_numbers()
        a_int = int(''.join(str(x) for x in reversed(a)), 2)
        b_int = int(''.join(str(x) for x in reversed(b)), 2)
        product = a_int * b_int
        c = [int(x) for x in reversed(bin(product)[2:].zfill(16))]
        data_point = (one_hot_encode(a), one_hot_encode(b), one_hot_encode(c))
        if len(training_data) < train_size:
            training_data.append(data_point)
        else:
            testing_data.append(data_point)
    return training_data, testing_data
