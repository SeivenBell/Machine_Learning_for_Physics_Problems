import random as rand


def makedata(size, seed):
    """
    Generates two lists of random integers and their products, all in little endian binary format.

    :param size: Number of elements in each list.
    :param seed: Seed for random number generation.
    :return: Three lists containing binary representations of random integers and their products.
    """
    rand.seed(seed)
    A, B, C = [], [], []
    A_int, B_int = [], []

    # Generate random integers and their binary representation in little endian format
    for _ in range(size):
        a_int = rand.randint(0, 255)
        A_int.append(a_int)
        A.append(format(a_int, '08b')[::-1])  # Convert to binary and reverse for little endian

    rand.seed(seed - 1)
    for _ in range(size):
        b_int = rand.randint(0, 255)
        B_int.append(b_int)
        B.append(format(b_int, '08b')[::-1])  # Convert to binary and reverse for little endian

    # Generate the product list
    for i in range(size):
        product = A_int[i] * B_int[i]
        C.append(format(product, '016b')[::-1])  # Convert to binary and reverse for little endian

    return A, B, C