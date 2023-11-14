import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import json
import random

# Set device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to convert integer to binary array
def int_to_bin_array(num, num_bits):
    return [int(digit) for digit in bin(num)[2:].zfill(num_bits)][::-1]

# 1. Data Generation
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
        data.append((a_bin, b_bin, c_bin))
    return data

# 2. RNN Model Creation
class BinaryMultiplicationRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(BinaryMultiplicationRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x should be of shape (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Get the output of the last time step
        return out



# 3. Training
def train(model, train_data, optimizer, criterion, num_epochs, batch_size):
    model.train()
    for epoch in range(num_epochs):
        random.shuffle(train_data)
        total_loss = 0
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            optimizer.zero_grad()

            for a, b, c in batch:
                x = torch.tensor([[a_i, b_i] for a_i, b_i in zip(a, b)], dtype=torch.float).to(device).unsqueeze(0)

                y = torch.tensor(c, dtype=torch.float).to(device)

                pred = model(x)
                loss = criterion(pred.view(-1), y)
                total_loss += loss.item()
                loss.backward()

            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_data):.4f}')

# 4. Evaluation
def evaluate(model, test_data, criterion):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for a, b, c in test_data:
            x = torch.tensor([[a_i, b_i] for a_i, b_i in zip(a, b)], dtype=torch.float).to(device).unsqueeze(0)
            y = torch.tensor(c, dtype=torch.float).to(device)
            pred = model(x)
            loss = criterion(pred.view(-1), y)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_data)
    return avg_loss

# 5. Loss Calculation with Swapped Inputs
def loss_with_swapped_inputs(model, data, criterion):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for a, b, c in data:
            x = torch.tensor([[b_i, a_i] for a_i, b_i in zip(a, b)], dtype=torch.float).to(device)
            y = torch.tensor(c, dtype=torch.float).to(device)
            pred = model(x)
            loss = criterion(pred.view(-1), y)
            total_loss += loss.item()
    avg_loss = total_loss / len(data)
    return avg_loss

# 6. Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train an RNN for binary integer multiplication')
    parser.add_argument('--param', type=str, help='file containing hyperparameters', required=True)
    parser.add_argument('--train-size', type=int, help='size of the generated training set', required=True)
    parser.add_argument('--test-size', type=int, help='size of the generated test set', required=True)
    parser.add_argument('--seed', type=int, help='random seed used for creating the datasets', required=True)
    return parser.parse_args()

# 7. Reporting and Main Function
def main():
    args = parse_arguments()
    
    # Load hyperparameters from the JSON file
    with open(args.param, 'r') as f:
        hyperparams = json.load(f)
    
    # Generate data
    train_data = generate_data(args.train_size, hyperparams['data']['max_digits'], args.seed)
    test_data = generate_data(args.test_size, hyperparams['data']['max_digits'], args.seed)
    
    # Initialize the model
    model_config = hyperparams['model']
    model = BinaryMultiplicationRNN(
        input_size=2, 
        hidden_size=model_config['rnn_units_per_layer'], 
        output_size=1, 
        num_layers=model_config['rnn_layers']
    ).to(device)

    # Training setup
    training_config = hyperparams['training']
    if model_config['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=model_config['learning_rate'])
    else:
        raise ValueError("Unsupported optimizer type provided.")

    if training_config['loss_function'].lower() == 'bcewithlogitsloss':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError("Unsupported loss function provided.")

    # Train the model
    train(
        model, 
        train_data, 
        optimizer, 
        criterion, 
        training_config['epochs'], 
        training_config['batch_size']
    )

    # Evaluate the model
    eval_loss = evaluate(model, test_data, criterion)
    print(f'Evaluation Loss: {eval_loss:.4f}')

    # Calculate loss with swapped inputs
    swap_loss = loss_with_swapped_inputs(model, test_data, criterion)
    print(f'Loss with Swapped Inputs: {swap_loss:.4f}')

if __name__ == '__main__':
    main()

