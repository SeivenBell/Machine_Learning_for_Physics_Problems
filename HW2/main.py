import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand
import time
import argparse
from data_handling import dataimporter
from model_training import calculateFVBM

def main(file_path):
    data = dataimporter(file_path)
    t0 = time.time()
    J = np.array([-1, 1, 1, 1])
    J_new, KL = calculateFVBM(data, 100, 13, 100, lr = 0.1)

    # Refactor of the for loop for updating elements in J_new
    final_j_vector = J_new[-1]  # Extracting the last row of J_new for processing

    # Vectorized approach for conditionally updating elements
    final_j_vector[final_j_vector < 0] = -1
    final_j_vector[final_j_vector >= 0] = 1

    # Assigning the updated vector back to the last row of J_new
    J_new[-1] = final_j_vector

    print(f"Coupling of Generated data set: {J_new[-1]}")
    print(f"Coupling of Original data set: {J}")
    print(np.array_equal(J, J_new[-1]))
    plt.plot(KL)
    plt.xlabel('Epoch')  # Label for x-axis
    plt.ylabel('KL Divergence')  # Label for y-axis
    plt.title('KL Divergence over Epochs')  # Title for the plot
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run FVBM on binary data')
    parser.add_argument('file_path', nargs='?', default='data\\in.txt', type=str, help='Path to the input data file')
    args = parser.parse_args()

    main(args.file_path)
    
    #python main.py data/in.txt

