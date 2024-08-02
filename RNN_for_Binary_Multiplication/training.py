import torch
import torch.nn.functional as f
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from model import RNN
from utils import input_creation, target_creation, batch, one_hotToBinary, runtime
from data_generation import makedata

def train_model(train_size, test_size, seed, params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Extract hyperparameters from params
    dim_hidden = params['model']['hidden_dim']
    lr = params['training']['learning_rate']
    batch_size = params['training']['batch_size']
    num_epochs = params['training']['num_epochs']

    # Generate training and testing data
    A_train, B_train, C_train = makedata(train_size, seed)
    A_test, B_test, C_test = makedata(test_size, seed + 1)

    # Prepare input and target data for training and testing
    AB_train = input_creation(A_train, B_train, 2).to(device)
    BA_train = input_creation(B_train, A_train, 2).to(device)
    C_train = target_creation(C_train, 2).to(device)
    AB_test = input_creation(A_test, B_test, 2).to(device)
    BA_test = input_creation(B_test, A_test, 2).to(device)
    C_test = target_creation(C_test, 2).to(device)

    # Initialize the RNN model
    model = RNN(2, dim_hidden, 1, 2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Training loop variables
    loss_vals = []
    test_loss_vals = []
    t0 = time.time()

    for epoch in range(num_epochs):
        # Alternate training data between AB and BA sequences each epoch
        current_input, current_target = (AB_train, C_train) if epoch % 2 == 0 else (BA_train, C_train)
        
        # Batch creation for current epoch
        x, targets, num_batches = batch(current_input, current_target, batch_size)
        display_epochs = 1
        epoch_loss = 0
        pred_list = []
        truevalues = []

        # Training loop for each batch
        for batch_num in range(num_batches):
            out = model(x[batch_num])
            loss = criterion(out, targets[batch_num])
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)

            prediction = one_hotToBinary(f.softmax(out, dim=1)).type(torch.LongTensor).to(device)
            pred_list.append(prediction)
            truevalues.append(one_hotToBinary(targets[batch_num]).type(torch.LongTensor).to(device))

        epoch_loss /= num_batches
        loss_vals.append(epoch_loss)

        # Testing with the unused sequence
        with torch.no_grad():
            test_inputs = AB_test if epoch % 2 == 0 else BA_test
            outte = model(test_inputs)
            test_loss = criterion(outte, C_test)
            test_loss_vals.append(test_loss.item())

            test_preds = one_hotToBinary(f.softmax(outte, dim=1)).type(torch.LongTensor).to(device)
            truetestvalues = one_hotToBinary(C_test).type(torch.LongTensor).to(device)

        # Display training and test results
        if (epoch+1) % display_epochs == 0:
            train_count, train_correct, test_count, test_correct = 0, 0, 0, 0

            # Calculating training accuracy
            for n in range(len(targets)):
                for k in range(min(batch_size, len(pred_list[n]))):  # Adjust for last batch size
                    for l in range(len(pred_list[n][k])-1):
                        train_count += 1
                        if pred_list[n][k][l+1] == truevalues[n][k][l+1]:
                            train_correct += 1

            # Calculating testing accuracy
            for n in range(test_preds.size(0)):
                for k in range(test_preds.size(1)-1):
                    test_count += 1
                    if test_preds[n][k+1] == truetestvalues[n][k+1]:
                        test_correct += 1

            # Calculating percentages for display
            percent_train = (train_correct / train_count) * 100 if train_count > 0 else 0
            percent_test = (test_correct / test_count) * 100 if test_count > 0 else 0
            t = runtime(t0)

            # Displaying epoch, training loss, test loss, and accuracies
            print(f'Epoch [{epoch+1}/{num_epochs}] - Training Data: {"AB" if epoch % 2 == 0 else "BA"}')
            print(f'\tTrain Loss: {epoch_loss:.4f}\tAccuracy (Train): {percent_train:.2f}%')
            print(f'\tTest Data: {"BA" if epoch % 2 == 0 else "AB"}')
            print(f'\tTest Loss: {test_loss.item():.4f}\tAccuracy (Test): {percent_test:.2f}%\tTotal Time: {t/60:.2f} mins')
            print('_'*85)

    # Plotting training and testing loss over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), loss_vals, label='Train Loss')
    plt.plot(range(num_epochs), test_loss_vals, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss over Epochs')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
