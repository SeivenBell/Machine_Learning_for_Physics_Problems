import nntplib
import numpy as np
import torch


def int_to_bin_array(num, num_bits):
    return [int(digit) for digit in bin(num)[2:].zfill(num_bits)][::-1]

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

class BinaryMultiplicationRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(BinaryMultiplicationRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nntplib.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out

def evaluate(model, test_data, criterion):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for a, b, c in test_data:
            x = torch.tensor([[a_i, b_i] for a_i, b_i in zip(a, b)], dtype=torch.float).to(device)
            y = torch.tensor(c, dtype=torch.float).to(device)
            pred = model(x)
            loss = criterion(pred.view(-1), y)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_data)
    return avg_loss

def loss_with_swapped_inputs(model, data, criterion):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for a, b, c in data:
            # Swap a and b
            x = torch.tensor([[b_i, a_i] for a_i, b_i in zip(a, b)], dtype=torch.float).to(device)
            y = torch.tensor(c, dtype=torch.float).to(device)
            pred = model(x)
            loss = criterion(pred.view(-1), y)
            total_loss += loss.item()
    avg_loss = total_loss / len(data)
    return avg_loss

