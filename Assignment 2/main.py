import numpy as np
import argparse

# Data Loader Module
def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = [[1 if spin == '+' else -1 for spin in line.strip()] for line in lines]
    size = len(data[0])
    return data, size

# Boltzmann Machine Module
class FullyVisibleBoltzmannMachine:
    def __init__(self, size):
        self.size = size
        self.couplers = {(i, (i + 1) % size): np.random.choice([-1, 1]) for i in range(size)}

    def train_step(self, data, learning_rate):
        gradients = {key: 0 for key in self.couplers.keys()}

        for spin_config in data:
            for i in range(self.size):
                j = (i + 1) % self.size
                gradients[(i, j)] += spin_config[i] * spin_config[j]

        for key in gradients:
            gradients[key] /= len(data)
            self.couplers[key] += learning_rate * gradients[key]

    def calculate_kl_divergence(self, data):
        model_prob = self._calculate_model_probability(data)
        empirical_prob = 1.0 / len(data)
        kl_divergence = np.sum(model_prob * np.log(model_prob / empirical_prob))
        return kl_divergence

    def _calculate_model_probability(self, data):
        total_energy = 0
        for spin_config in data:
            for i in range(self.size):
                j = (i + 1) % self.size
                total_energy += -self.couplers[(i, j)] * spin_config[i] * spin_config[j]

        model_prob = np.exp(-total_energy)
        partition_function = np.sum(np.exp(-total_energy))
        return model_prob / partition_function

# Training Module
def train_model(boltzmann_machine, training_data, epochs, learning_rate, verbose):
    for epoch in range(epochs):
        boltzmann_machine.train_step(training_data, learning_rate)

        if verbose:
            kl_divergence = boltzmann_machine.calculate_kl_divergence(training_data)
            print(f"Epoch {epoch}, KL Divergence: {kl_divergence}")

# Prediction Module
def predict_couplers(boltzmann_machine):
    return boltzmann_machine.couplers

# Main Module
def main():
    parser = argparse.ArgumentParser(description="Train a FVBM on Ising Chain Data")
    parser.add_argument("file_path", help="Path to the training data file")
    args = parser.parse_args()

    training_data, size = load_data(args.file_path)
    boltzmann_machine = FullyVisibleBoltzmannMachine(size)
    train_model(boltzmann_machine, training_data, epochs=100, learning_rate=0.01, verbose=True)
    coupler_predictions = predict_couplers(boltzmann_machine)
    print(coupler_predictions)

if __name__ == "__main__":
    main()
