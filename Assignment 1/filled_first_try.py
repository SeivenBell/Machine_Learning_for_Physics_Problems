import numpy as np
import random
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import json

# Function to convert integer to binary array
def int_to_bin_array(num, num_bits):
    return [int(digit) for digit in bin(num)[2:].zfill(num_bits)][::-1]

# Data Generation
def generate_data(size, num_bits, seed):
    np.random.seed(seed)
    max_num = 2**num_bits - 1
    data = []
    for _ in range(size):
        a = np.random.randint(max_num + 1)
        b = np.random.randint(max_num + 1)
        c = a * b
        a_bin = int_to_bin_array(a, num_bits)
        b_bin = int_to_bin_array(b, num_bits)
        c_bin = int_to_bin_array(c, 2 * num_bits)
        input_seq = []
        for a_bit, b_bit in zip(a_bin, b_bin):
            input_seq.extend([a_bit, b_bit])
        #input_seq.append(0)  # Padding to match the output length
        data.append((input_seq, c_bin))
        #print(data)
    return data

import torch.nn as nn

# RNN Model
class BinaryMultiplicationRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(BinaryMultiplicationRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x)#, h0)
        out = torch.sigmoid(self.fc(out))  # Get the output of the last time step
        return out

train_losses = []
train_accuracies = []

# Training Function
def train(model, train_data, optimizer, criterion, num_epochs, batch_size, device, train_losses, train_accuracies):
    model.train()
    for epoch in range(num_epochs):
        random.shuffle(train_data)
        total_loss = 0
        total_accuracy = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            x_batch, y_batch = zip(*batch)
            x_tensor = torch.tensor(x_batch, dtype=torch.float).to(device)
            y_tensor = torch.tensor(y_batch, dtype=torch.float).to(device)

            optimizer.zero_grad()
            pred = model(x_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += binary_accuracy(pred, y_tensor)

        epoch_loss = total_loss / len(train_data)
        epoch_accuracy = total_accuracy / len(train_data)
        
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')


# Evaluation Function
def evaluate(model, test_data, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in test_data:
            x_tensor = torch.tensor([x], dtype=torch.float).to(device)
            y_tensor = torch.tensor([y], dtype=torch.float).to(device)

            pred = model(x_tensor)
            loss = criterion(pred, y_tensor)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_data)
    return avg_loss

# Loss Calculation with Swapped Inputs
def loss_with_swapped_inputs(model, data, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in data:
            swapped_x = [x[i+1] if i % 2 == 0 else x[i-1] for i in range(len(x)-1)] + [0]
            x_tensor = torch.tensor([swapped_x], dtype=torch.float).to(device)
            y_tensor = torch.tensor([y], dtype=torch.float).to(device)

            pred = model(x_tensor)
            loss = criterion(pred, y_tensor)
            total_loss += loss.item()

    avg_loss = total_loss / len(data)
    return avg_loss
def binary_accuracy(y_pred, y_true):
    # Applying sigmoid function to convert logits to probabilities
    y_pred = torch.sigmoid(y_pred)
    # Rounding to get binary predictions
    predicted = y_pred.round()
    # Calculating accuracy
    correct = (predicted == y_true).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy.item()


import json
import argparse
import torch

def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description='Train an RNN for binary integer multiplication')
    parser.add_argument('--param', type=str, help='file containing hyperparameters', required=True)
    parser.add_argument('--train-size', type=int, help='size of the generated training set', required=True)
    parser.add_argument('--test-size', type=int, help='size of the generated test set', required=True)
    parser.add_argument('--seed', type=int, help='random seed used for creating the datasets', required=True)
    args = parser.parse_args()

    # Load Hyperparameters
    with open(args.param, 'r') as f:
        hyperparams = json.load(f)

    # Generate Data
    train_data = generate_data(args.train_size, hyperparams['data']['max_digits'], args.seed)
    test_data = generate_data(args.test_size, hyperparams['data']['max_digits'], args.seed)

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = hyperparams['model']
    model = BinaryMultiplicationRNN(
        input_size=16, 
        hidden_size=model_config['rnn_units_per_layer'], 
        output_size=16,  # 16 bits for 8-bit multiplication
        num_layers=model_config['rnn_layers']
    ).to(device)

    training_config = hyperparams['training']  # This line gets the 'training' section

    # Use training_config to access 'optimizer' and 'loss_function'
    if training_config['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config['learning_rate'])
    else:
        raise ValueError("Unsupported optimizer type provided.")

    if training_config['loss_function'].lower() == 'bcewithlogitsloss':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError("Unsupported loss function provided.")
    
    train_losses = []
    train_accuracies = []

    # Train the Model
    train(model, train_data, optimizer, criterion, hyperparams['training']['epochs'], 
          hyperparams['training']['batch_size'], device, train_losses, train_accuracies)

    # Evaluate the Model
    eval_loss = evaluate(model, test_data, criterion, device)
    print(f'Evaluation Loss: {eval_loss:.4f}')

    # Calculate Loss with Swapped Inputs
    swap_loss = loss_with_swapped_inputs(model, test_data, criterion, device)
    print(f'Loss with Swapped Inputs: {swap_loss:.4f}')
    
    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    main()
