import numpy as np
import argparse

# === Data Loader Module ===
# This function loads the data from a file and processes it into a usable format.
def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Convert '+' to 1 and '-' to -1, as per the Ising model representation
    data = [[1 if spin == '+' else -1 for spin in line.strip()] for line in lines]
    size = len(data[0])  # Determine the size of the Ising chain from the first row
    return data, size

# === Boltzmann Machine Module ===
# This class represents the Fully Visible Boltzmann Machine, specific to the Ising model.
class FullyVisibleBoltzmannMachine:
    def __init__(self, size):
        # Initialize the couplers randomly. They connect adjacent spins in the 1D chain.
        self.size = size
        self.couplers = {(i, (i + 1) % size): np.random.choice([-1, 1]) for i in range(size)}

    def train_step(self, data, learning_rate):
        # Calculate gradients for the couplers based on the training data.
        gradients = {key: 0 for key in self.couplers.keys()}
        for spin_config in data:
            for i in range(self.size):
                j = (i + 1) % self.size
                gradients[(i, j)] += spin_config[i] * spin_config[j]

        # Update the couplers using the calculated gradients and learning rate.
        for key in gradients:
            gradients[key] /= len(data)
            self.couplers[key] += learning_rate * gradients[key]

    def calculate_kl_divergence(self, data):
        # Calculate the Kullback-Leibler divergence to measure model performance.
        model_prob = self._calculate_model_probability(data)
        empirical_prob = 1.0 / len(data)  # Uniform probability for training data states
        kl_divergence = np.sum(model_prob * np.log(model_prob / empirical_prob))
        return kl_divergence

    def _calculate_model_probability(self, data):
        # Helper function to calculate the probability distribution of the model.
        total_energy = 0
        for spin_config in data:
            for i in range(self.size):
                j = (i + 1) % self.size
                total_energy += -self.couplers[(i, j)] * spin_config[i] * spin_config[j]

        # Calculate the model probability using the Boltzmann distribution.
        model_prob = np.exp(-total_energy)
        partition_function = np.sum(np.exp(-total_energy))
        return model_prob / partition_function

# === Training Module ===
# This function manages the training process of the Boltzmann Machine.
def train_model(boltzmann_machine, training_data, epochs, learning_rate, verbose):
    for epoch in range(epochs):
        boltzmann_machine.train_step(training_data, learning_rate)

        # If verbose, print the KL divergence for each epoch to track performance.
        if verbose:
            kl_divergence = boltzmann_machine.calculate_kl_divergence(training_data)
            print(f"Epoch {epoch}, KL Divergence: {kl_divergence}")

# === Prediction Module ===
# This function predicts the coupler values using the trained Boltzmann Machine.
def predict_couplers(boltzmann_machine):
    # Return the predicted couplers as a dictionary.
    return boltzmann_machine.couplers

# === Main Module ===
# Main function to orchestrate the data loading, training, and prediction process.
def main():
    parser = argparse.ArgumentParser(description="Train a FVBM on Ising Chain Data")
    parser.add_argument("file_path", help="Path to the training data file")
    args = parser.parse_args()

    # Load data and initialize the Boltzmann Machine.
    training_data, size = load_data(args.file_path)
    boltzmann_machine = FullyVisibleBoltzmannMachine(size)

    # Train the model and predict couplers.
    train_model(boltzmann_machine, training_data, epochs=100, learning_rate=0.01, verbose=True)
    coupler_predictions = predict_couplers(boltzmann_machine)

    # Output the predicted couplers.
    print(coupler_predictions)

if __name__ == "__main__":
    main()
