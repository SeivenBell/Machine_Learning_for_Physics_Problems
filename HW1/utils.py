import torch
import time

def runtime(t_i):
    t_r = time.time() - t_i
    return t_r

def binary_to_one_hot(binary_string):
    """
    Converts a binary string into a one-hot encoded PyTorch tensor.
    
    :param binary_string: A string containing binary digits.
    :return: A 2D PyTorch tensor representing the one-hot encoding of the binary string.
    """
    num_bits = 2  # Since the binary system has two bits: 0 and 1
    one_hot_tensor = torch.zeros(len(binary_string), num_bits)

    # Mapping for binary digits to tensor indices
    binary_to_index = {'0': 0, '1': 1}

    for idx, digit in enumerate(binary_string):
        one_hot_tensor[idx][binary_to_index[digit]] = 1

    return one_hot_tensor


def input_creation(l_1, l_2, index_size):
    """
    Creates a tensor from two lists of binary strings. Each tensor element is a one-hot encoded 
    representation of concatenated elements from the two lists, with an additional '0'.

    :param l_1: First list of binary strings.
    :param l_2: Second list of binary strings.
    :param index_size: The size of the index for the one-hot encoding.
    :return: A tensor of shape (N, 2L+1, H) where L is the length of the binary strings.
    """
    tensor = torch.zeros(len(l_1), 2 * len(l_1[0]) + 1, index_size)
    list_AB = []

    # Concatenating binary strings from both lists and adding a '0' at the end
    for value_1, value_2 in zip(l_1, l_2):
        concatenated = value_1 + value_2 + '0'
        list_AB.append(concatenated)

    # Converting concatenated strings to one-hot encoded format and storing in the tensor
    for idx, value in enumerate(list_AB):
        tensor[idx] = binary_to_one_hot(value)

    return tensor



def target_creation(binary_list, index_size):
    """
    Creates a tensor from a list of binary strings. Each tensor element is a one-hot encoded
    representation of the binary strings prefixed with '0'.

    :param binary_list: List of binary strings.
    :param index_size: The size of the index for the one-hot encoding.
    :return: A tensor of shape (N, C, L+1), where N is the number of binary strings and L is the length of each string.
    """
    num_strings = len(binary_list)
    string_length = len(binary_list[0]) + 1  # +1 for the prefixed '0'

    targetsOH = torch.zeros(num_strings, index_size, string_length)

    for idx, value in enumerate(binary_list):
        prefixed_value = '0' + value
        one_hot_line = binary_to_one_hot(prefixed_value)
        one_hot_line = one_hot_line.transpose(0, 1)
        targetsOH[idx] = one_hot_line

    return targetsOH
   

# def batch(inputs, targets,  batchsize):
#     """
#     Splits inputs and targets tensors into batches.

#     :param inputs: Input tensor of shape (N, L, H).
#     :param targets: Target tensor of shape (N, 1, L).
#     :param batch_size: Size of each batch.
#     :return: Tuple containing batches of inputs, targets, and the number of batches.
#     """
#     num_samples = inputs.size(0) 
#     # Calculating the number of batches based on the batch size
#     num_batches = int((num_samples - (num_samples%batchsize))/batchsize)
#     # Splitting the tensors into batches
#     in_batches = inputs.tensor_split(num_batches)
        
#     target_batches = targets.tensor_split(num_batches)
        
#     return in_batches, target_batches, num_batches

def batch(inputs, targets, batch_size):
    """
    Splits inputs and targets tensors into batches.

    :param inputs: Input tensor of shape (N, L, H).
    :param targets: Target tensor of shape (N, 1, L).
    :param batch_size: Size of each batch.
    :return: Tuple containing batches of inputs, targets, and the number of batches.
    """
    num_samples = inputs.size(0)
    num_batches = num_samples // batch_size + (num_samples % batch_size > 0)

    in_batches = inputs.split(batch_size)
    target_batches = targets.split(batch_size)

    return in_batches, target_batches, num_batches

def one_hotToBinary(one_hot_tensor):
    """
    Converts a one-hot encoded tensor to its binary representation.

    :param one_hot_tensor: A tensor in one-hot encoded format.
    :return: A tensor in binary format.
    """
    num_samples, _, seq_length = one_hot_tensor.size()
    binary_tensor = torch.zeros(num_samples, seq_length, device=one_hot_tensor.device)

    # Convert one-hot encoding to binary representation
    binary_tensor = torch.argmax(one_hot_tensor, dim=1)
    
    return binary_tensor
