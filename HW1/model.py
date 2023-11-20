import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        """
        Initializes the RNN module.

        :param input_dim: Dimension of the input features.
        :param hidden_dim: Dimension of the hidden layer.
        :param num_layers: Number of RNN layers.
        :param output_dim: Dimension of the output layer.
        """
        super(RNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Define RNN layer
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)

        # Define fully conneC_testd layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        """
        Forward pass through the RNN.

        :param inputs: Input tensor for the RNN.
        :return: Transposed output from the fully conneC_testd layer.
        """
        # Pass input through RNN layers
        outputs, _ = self.rnn(inputs)

        # Pass the output of RNN layers through the fully conneC_testd layer
        outputs_final = self.fc(outputs)
        
        # Transpose the output for desired shape
        return outputs_final.transpose(1, 2)
