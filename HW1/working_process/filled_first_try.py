import numpy as np
import random
from random import randint
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import json


#python filled_first_try.py --param param/param.json --train-size 10000 --test-size 1000 --seed 1234
# Function to convert integer to binary array

def dataGeneration(size,seed):
    # genrates 2 random lists of  integer between 0,255 (inclusive) converts to the binary representation in little endian format
    A = []
    A_int = []
    random.seed(seed)
    for i in range(size):
        randa = random.randint(0,255)
        A_int.append(randa)
        a = bin(randa)[2:]
        la = len(a)
        a =  str(0) * (8 - la) + a  #big endian format
        a = a[::-1] #little endian format
        A.append(a)
    random.seed(seed-1) #this ensures no matter size  b_i = b_i for seed N
    B = []
    B_int = []
    for i in range(size):
        randb = random.randint(0,255)
        B_int.append(randb)
        b = bin(randb)[2:]
        lb = len(b)
        b =  str(0) * (8 - lb) + b #big endian
        b = b[::-1] #little endian
        B.append(b)
    #creates a list of the product of A_i and B_i in little endian format  
    C = []
    for i in range(size):
        c = bin(A_int[i]*B_int[i])[2:]
        lc = len(c)
        c= str(0) * (16-lc) + c #big endian
        c = c[::-1] #little endian
        C.append(c)
    
    return A,B,C

# each binary digit becomes an input feature, so the input tensor shape for each 
# sample should be [sequence_length, input_dim] where sequence_length is 8 for inputs
# and 16 for outputs.

# RNN Model
########################################################################################### 
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(RNN, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first = True)#, dropout = 0.3)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        
    def forward(self, inputs):
        # Forward propagate through the RNN layer
        rnn_outputs, _ = self.rnn(inputs)
        
        # Pass the output of the RNN through the fully-connected layer
        final_outputs = self.fc(rnn_outputs)
        
        # Transpose the final output to match the expected dimensions
        return final_outputs.transpose(1, 2)

    def reset(self):
        self.rnn.reset_parameters()
        self.fc.reset_parameters()
        
        
###########################################################################################

# Training Function
def train(model, train_data, optimizer, criterion, num_epochs, batch_size, device, train_losses, train_accuracies):
    model.train()
    for epoch in range(num_epochs):
        random.shuffle(train_data)
        total_loss = 0
        total_accuracy = 0

        for i in range(0, len(train_data), batch_size):
            # Extracting batches
            batch = train_data[i:i + batch_size]
            x_batch, y_batch = zip(*batch)

            # Convert binary strings to tensor
            x_tensor = torch.tensor([[int(bit) for bit in num] for num in x_batch], dtype=torch.float).to(device)
            y_tensor = torch.tensor([[int(bit) for bit in num] for num in y_batch], dtype=torch.float).to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            pred = model(x_tensor)
            loss = criterion(pred, y_tensor)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += binary_accuracy(pred, y_tensor)

        # Calculate average loss and accuracy
        epoch_loss = total_loss / len(train_data)
        epoch_accuracy = total_accuracy / len(train_data)

        # Print epoch stats
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        


# Evaluation Function
def evaluate(model, test_data, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in test_data:
            # Convert binary strings to tensor
            x_tensor = torch.tensor([[int(bit) for bit in num] for num in x], dtype=torch.float).to(device)
            y_tensor = torch.tensor([[int(bit) for bit in num] for num in y], dtype=torch.float).to(device)

            # Forward pass
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
            # Swap the inputs
            swapped_x = [x[i+1] if i % 2 == 0 else x[i-1] for i in range(len(x)-1)] + ['0']
            
            # Convert binary strings to tensor
            x_tensor = torch.tensor([[int(bit) for bit in num] for num in swapped_x], dtype=torch.float).to(device)
            y_tensor = torch.tensor([[int(bit) for bit in num] for num in y], dtype=torch.float).to(device)

            # Forward pass
            pred = model(x_tensor)
            loss = criterion(pred, y_tensor)
            total_loss += loss.item()

    avg_loss = total_loss / len(data)
    return avg_loss


def binary_accuracy(y_pred, y_true):
    predicted = (y_pred > 0.5).float()  # Threshold at 0.5
    correct = (predicted == y_true).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy.item()


#####################################################################################
#####################################################################################

# Main Function
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
    train_data = dataGeneration(args.train_size, args.seed)
    test_data = dataGeneration(args.test_size, args.seed)

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    model_config = hyperparams['model']
    model = RNN(
        input_dim=model_config['input_dim'], 
        hidden_dim=model_config['hidden_dim'], 
        num_layers=model_config['num_layers'],
        output_dim=model_config['output_dim']
    ).to(device)
    
    # Optimizer and Loss Function
    training_config = hyperparams['training']
    if training_config['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config['learning_rate'])
    else:
        raise ValueError("Unsupported optimizer type provided.")

    if training_config['loss_function'].lower() == 'mse':
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError("Unsupported loss function provided.")

    # Initialize lists to track losses and accuracies
    train_losses = []
    train_accuracies = []

    # Train the Model
    train(model, train_data, optimizer, criterion, training_config['epochs'], 
          training_config['batch_size'], device, train_losses, train_accuracies)

    # Evaluate the Model
    eval_loss = evaluate(model, test_data, criterion, device)
    print(f'Evaluation Loss: {eval_loss:.4f}')

    # Calculate Loss with Swapped Inputs
    swap_loss = loss_with_swapped_inputs(model, test_data, criterion, device)
    print(f'Loss with Swapped Inputs: {swap_loss:.4f}')

# print("Training Losses:", train_losses)
# print("Training Accuracies:", train_accuracies)

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
    
# param.json:
    
# {
#     "model": {
#         "input_dim": 1, 
#         "hidden_dim": 64,
#         "num_layers": 2,
#         "output_dim": 16,
#         "dropout_rate": 0.2
#     },
#     "training": {
#         "optimizer": "adam",
#         "learning_rate": 0.03,
#         "batch_size": 32,
#         "epochs": 50,
#         "loss_function": "BCEWithLogitsLoss"
#     }
# }


# python filled_first_try.py --param param/param.json --train-size 10000 --test-size 1000 --seed 1337