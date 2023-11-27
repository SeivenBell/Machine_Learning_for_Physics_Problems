import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand
import time
import argparse
import json
from data_handling import dataimporter
from model_training import calculateFVBM

def main(file_path, batch_size, num_epochs, lr):
    data = dataimporter(file_path)
    t0 = time.time()
    J = np.array([-1, 1, 1, 1])
    J_new, KL = calculateFVBM(data, batch_size, 13, num_epochs, lr=lr)

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
    
    # Added arguments for hyperparameters
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for training')
    
    # Added argument for specifying JSON parameter file
    parser.add_argument('--param_file', type=str, default=None, help='Path to JSON parameter file')

    args = parser.parse_args()
    
    # If a JSON parameter file is provided, load and override hyperparameters
    if args.param_file:
        with open(args.param_file, 'r') as param_file:
            param_dict = json.load(param_file)
            # Override hyperparameters with values from the JSON file
            args.batch_size = param_dict.get('batch_size', args.batch_size)
            args.num_epochs = param_dict.get('num_epochs', args.num_epochs)
            args.lr = param_dict.get('lr', args.lr)

    main(args.file_path, args.batch_size, args.num_epochs, args.lr)
