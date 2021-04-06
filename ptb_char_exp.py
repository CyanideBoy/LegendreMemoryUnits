import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from dataloaders.ptb import ptb_char
import models
import torch.optim as optim
import time

LEARNINGRATE = 1e-2
GAMMA = 0.98
LEN_TOTAL = 500
BATCHSIZE = 256
NUMEPOCHS = 50
MODEL = 'LMU'
base_pth = 'results/ptb/'


def get_random_chunks(inp, target):
    start = torch.randint(0, inp.size(0) - LEN_TOTAL, size=(BATCHSIZE,))
    inputs = torch.zeros(BATCHSIZE, LEN_TOTAL, inp.size(1))
    targets = torch.zeros(BATCHSIZE, LEN_TOTAL)
    for i, s in enumerate(start):
        inputs[i] = inp[s: s+LEN_TOTAL]
        targets[i] = target[s: s+LEN_TOTAL]

    return inputs, targets


a = ptb_char(path='datasets/ptbdataset')
data = a.process_all()
# convert to tensors
for split in ['train', 'test', 'val']:
    data[split] = (torch.tensor(data[split][0]), torch.tensor(data[split][1]), data[split][2])
char_to_idx = data['train'][2]
vocab_size = len(char_to_idx)
# print("idx_to_char:", char_to_idx)
idx_to_char = {v: k for k, v in char_to_idx.items()}

print('Loaded datasets....')
print('Using Model', MODEL)
if MODEL == 'LSTM':
    model = models.LSTM_general(inp_features=vocab_size)
elif MODEL == 'LMU':
    model = models.LMU_general(inp_features=vocab_size)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on',device)
print('Building model..')	
model.to(device)
print('Model Built.')

print('Initializing optimizer and scheduler..')
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNINGRATE)             # OR RAdam/DiffGrad/etc
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = GAMMA)

min_val_loss = float('inf')

loss_values_train = []
loss_values_val = []

'''
def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
'''

for epoch in range(1, NUMEPOCHS+1):
    start_time = time.time()
    
    model.train()
    runloss = 0.0
    hidden = model.init_hidden(BATCHSIZE,LEN_TOTAL)

    data_input, data_output = get_random_chunks(data['train'][0], data['train'][1])    
    data_input = data_input.transpose(0,1).to(device, dtype=torch.float)  
    data_output = data_output.transpose(0,1).to(device, dtype=torch.long)
    
    optimizer.zero_grad()
    output, hidden = model(data_input)

    output = output.view(-1, output.size(2))
    data_output = data_output.flatten()
    loss = criterion(output, data_output)
    runloss = loss.item()
    loss_values_train.append(runloss)
    loss.backward()
    optimizer.step()
        
    model.eval()
    
    val_loss = 0
    hidden = model.init_hidden(BATCHSIZE,LEN_TOTAL)
    with torch.no_grad():
        data_input, data_output = get_random_chunks(data['val'][0], data['val'][1])
        data_input = data_input.transpose(0,1).to(device, dtype=torch.float)  
        data_output = data_output.transpose(0,1).to(device, dtype=torch.long)

        output, hidden = model(data_input)
        output = output.view(-1, output.size(2))
        data_output = data_output.flatten()
        loss = criterion(output, data_output) 
        val_loss = loss.item()

    loss_values_val.append(val_loss)

    if val_loss <= min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), base_pth+MODEL+'_best_val_quicksave.pt')
    
    stop_time = time.time()
    time_el = int(stop_time-start_time)
    
    print('epoch [{}/{}], loss:{:.7f}, val loss:{:.7f}, in {}h {}m {}s'.format(epoch, NUMEPOCHS,
                                                                                runloss, val_loss,  time_el//3600,
                                                                                (time_el%3600)//60,
                                                                                time_el%60))
    scheduler.step()

fig = plt.figure()
plt.plot(np.array(loss_values_train), 'b')
plt.plot(np.array(loss_values_val), 'r')
plt.legend(['Train','Val'])
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Loss')
fig.savefig(base_pth+MODEL+'_train_curve.png')

# hidden = model.init_hidden(BATCHSIZE,LEN_TOTAL)
# test_loss = 0.0
# with torch.no_grad():
#     for data_input, data_output in test_loader:
#         data_input = data_input.transpose(0,1).transpose(0,2).to(device, dtype=torch.float)
#         data_output = data_output.transpose(0,1).transpose(0,2).to(device, dtype=torch.float)
#         output, hidden = model(data_input)
#         #hidden = repackage_hidden(hidden)
#         #output, hidden = model(data_input,hidden)
#         loss = torch.sqrt(torch.mean(torch.square((output[LEN_INIT:]-data_output[LEN_INIT:])))/torch.mean(torch.square(output[LEN_INIT:])))
#         test_loss += loss.item()*BATCHSIZE
# test_loss = test_loss/len(test_loader.dataset)
# print('Test loss ',test_loss)



np.save(base_pth+MODEL+'_data.npy',(loss_values_train,loss_values_val))