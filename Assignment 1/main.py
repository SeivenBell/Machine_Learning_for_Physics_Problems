import torch as torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import random as rand
import time
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#device = 'cpu'
def runtime(t_i):
    t_r = time.time() - t_i
    return t_r

def makedata(size,seed):
    # genrates 2 random lists of  integer between 0,255 (inclusive) converts to the binary representation in little endian format
    A = []
    Aint = []
    rand.seed(seed)
    for i in range(size):
        randa = rand.randint(0,255)
        Aint.append(randa)
        a = bin(randa)[2:]
        la = len(a)
        a =  str(0) * (8 - la) + a  #big endian format
        a = a[::-1] #little endian format
        A.append(a)
    rand.seed(seed-1) #this ensures no matter size  b_i = b_i for seed N
    B = []
    Bint = []
    for i in range(size):
        randb = rand.randint(0,255)
        Bint.append(randb)
        b = bin(randb)[2:]
        lb = len(b)
        b =  str(0) * (8 - lb) + b #big endian 
        b = b[::-1] #little endian
        B.append(b)
    #creates a list of the product of A_i and B_i in little endian format    
    C = []
    for i in range(size):
        c = bin(Aint[i]*Bint[i])[2:]
        lc = len(c)
        c= str(0) * (16-lc) + c #big endian
        c = c[::-1] #little endian
        C.append(c)
    
    return A,B,C

    
def binary_to_onehot(line):
    #turns a line of binary into a num_bits x C tensor
    index = '01'
    tensor = torch.zeros(len(line),len(index))
    for count, number in enumerate(line):
        tensor[count][index.find(number)] = 1
        
    return tensor



def createinput(l1,l2,index_size):
     # creates a tensor holding the lines of binary from to inputs
     # in size N x L x H
    listAB =[]
    tensor = tensor = torch.zeros(len(l1),2*len(l1[0])+1,index_size)
    for count, value in enumerate(l1):
        element = value+l2[count] + '0'
        listAB.append(element)
    for count, value in enumerate(listAB):
        line = binary_to_onehot(value)
        tensor[count] = line
        
    return tensor
    

def createtargets(list1,index_size):
    # output is in shape N X C X 16 
    targetsOH = torch.zeros(len(list1),index_size, len(list1[0]) + 1)
   
    
    for count, value in enumerate(list1):
        
        value = '0' + value
        line = binary_to_onehot(value)
        line = line.transpose(0,1)
        targetsOH[count] = line
            
    
    return targetsOH       


def batch(inputs, targets,  batchsize):
    #inputs of size N x L X H 
    #Targets in size N X 1 X L
    num_samples = inputs.size(0) 
    
    num_batches = int((num_samples - (num_samples%batchsize))/batchsize)
    
    
    in_batches = inputs.tensor_split(num_batches)
        
        
    target_batches = targets.tensor_split(num_batches)
        
    return in_batches, target_batches, num_batches
        
        
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(RNN, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first = True)#, dropout = 0.3)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        
    def forward(self, inputs):
        
        #one time step
        outputs ,_ = self.rnn(inputs)
        
        #outputs1 = outputs.reshape(outputs.shape[0], -1)  
        outputsfinal = self.fc(outputs)
        return outputsfinal.transpose(1,2)
    
    def reset(self):
        self.rnn.reset_parameters()
        self.fc.reset_parameters()
    
        
criterion = nn.MSELoss()

t0 = time.time()
Atr,Btr,Ctr = makedata(10000,5874)
Ate, Bte, Cte = makedata(1000,6456)
print(runtime(t0))
ABtr = createinput(Atr,Btr,2).to(device)
Ctr = createtargets(Ctr,2)
Ctr = Ctr.to(device)

ABte = createinput(Ate,Bte,2).to(device)
Cte = createtargets(Cte,2)
Cte = Cte.to(device)
print(runtime(t0))

t = time.time()
dim_hidden = 256
lr = 0.001
batch_size = 32
x , targets, num_batches = batch(ABtr, Ctr, batch_size)
print(runtime(t))
model = RNN(2,dim_hidden,1,2).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr = lr)#,
                                #alpha=0.99, 
                                #eps=1e-08, 
                                #weight_decay=0, 
                                #momentum=0.9)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
def oneHotToBinary(out):
    N , C, L = out.size()
    pred = torch.zeros(N,L).to(device)
    
    for n in range(N):
        m = torch.argmax(out[n], dim = 0)
        pred[n] = m 
    return pred
print(targets[0].size())
print(x[0].size())




loss_vals = []
test_loss_vals = []
num_epochs = 25
display_epochs = 25
t0 = time.time()
print(''*85)

for epoch in range(num_epochs):

    epoch_loss = 0
    pred_list = []
    truevalues = []

    for batch_num in range(num_batches):
        out = model.forward((x[batch_num]))
        loss = criterion(out, (targets[batch_num]))
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)

        prediction = oneHotToBinary(f.softmax(out, dim=1)).type(torch.LongTensor).to(device)
        pred_list.append(prediction)
        truevalues.append(oneHotToBinary(targets[batch_num]).type(torch.LongTensor).to(device))

    epoch_loss /= num_batches
    loss_vals.append(epoch_loss)

    with torch.no_grad():
        outte = model.forward(ABte)
        test_loss = criterion(outte, Cte)
        test_loss_vals.append(test_loss.item())

        test_preds = oneHotToBinary(f.softmax(outte, dim=1)).type(torch.LongTensor).to(device)
        truetestvalues = oneHotToBinary(Cte).type(torch.LongTensor).to(device)

    if (epoch+1) % display_epochs == 0:
        trcount = 0
        trcorrect = 0
        
        for n in range(len(targets)):
            for k in range(batch_size):
                for l in range(len(pred_list[n][k])-1):
                    trcount += 1 
                    if pred_list[n][k][l+1] == truevalues[n][k][l+1]:
                        trcorrect  += 1
        tecount = 0
        tecorrect = 0
        
        for n in range(test_preds.size(0)):
            for k in range(test_preds.size(1)-1):
            
                tecount += 1 
                if test_preds[n][k+1] == truetestvalues[n][k+1]:
                        tecorrect  += 1
        percent = (trcorrect/trcount)*100
        percent_test = (tecorrect/tecount)*100
        t = runtime(t0)
        
        print('Epoch [{}/{}]\tTrain Loss:{:.4f}\tPercent correct {:.2f} % '.format(epoch+1, num_epochs, loss.item(),percent))
        print('Epoch [{}/{}]\tTest Loss:{:.4f}\tPercent correct {:.2f} %\tTotal time: {:.2f} mins'.format(epoch+1, num_epochs, test_loss.item(),percent_test,t/60))
        print('_'*85 )
   # loss_vals.append(loss.item())

    
plt.plot(range(num_epochs), loss_vals, label='Train')
plt.plot(range(num_epochs), test_loss_vals, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()
    

