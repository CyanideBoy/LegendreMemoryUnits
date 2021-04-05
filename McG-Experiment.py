from __future__ import print_function, division
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from MG.mg_generator import MGDataset
from custom.models import McG_LSTM, McG_LMU
import torch.optim as optim
import time

LEARNINGRATE = 1e-2
GAMMA = 0.95
BATCHSIZE = 256
LEN_INIT = 32
LEN_TOTAL = 256 + LEN_INIT
NUMEPOCHS = 100
PREDICT = 15
SPLIT = [4096,256,256]
MODEL = 'LSTM'


MG = MGDataset('.', LEN_TOTAL, PREDICT, SPLIT)
data = MG.get_dataset() 
train_data = TensorDataset(data['train'][0],data['train'][1])
val_data = TensorDataset(data['val'][0],data['val'][1])
test_data = TensorDataset(data['test'][0],data['test'][1])

train_loader = DataLoader(train_data,batch_size=BATCHSIZE,shuffle=True)
val_loader = DataLoader(val_data,batch_size=BATCHSIZE,shuffle=True)
test_loader = DataLoader(test_data,batch_size=BATCHSIZE,shuffle=True)

print('Loaded datasets....')
print('Using Model ',MODEL)
if MODEL == 'LSTM':
    model = McG_LSTM(BSIZE=BATCHSIZE,LEN=LEN_TOTAL)

elif MODEL == 'LMU':
    model = McG_LMU(BSIZE=BATCHSIZE,LEN=LEN_TOTAL)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on',device)
print('Building model..')	
model.to(device)
print('Model Built.')

print('Initializing optimizer and scheduler..')
criterion = torch.nn.MSELoss()
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

for epoch in range(1,NUMEPOCHS+1):
    start_time = time.time()
    
    model.train()
    runloss = 0.0
    hidden = model.init_hidden(BATCHSIZE,LEN_TOTAL)

    for data_input, data_output in train_loader:    
        optimizer.zero_grad()
        data_input = data_input.transpose(0,1).transpose(0,2).to(device, dtype=torch.float)  
        data_output = data_output.transpose(0,1).transpose(0,2).to(device, dtype=torch.float)
        
        output, hidden = model(data_input)
        #hidden = repackage_hidden(hidden)
        #output, hidden = model(data_input,hidden) 
        loss = criterion(output[LEN_INIT:], data_output[LEN_INIT:])
        #print(torch.sqrt(torch.mean(torch.square((output[LEN_INIT:]-data_output[LEN_INIT:])))/torch.mean(torch.square(output[LEN_INIT:]))))
        runloss += loss.item()*BATCHSIZE

        loss.backward()
        optimizer.step()
        runloss += loss.item()

    runloss /= len(train_loader.dataset)
    loss_values_train.append(runloss)
    
    model.eval()
    
    val_loss = 0
    hidden = model.init_hidden(BATCHSIZE,LEN_TOTAL)
    with torch.no_grad():
        for data_input, data_output in val_loader:
            data_input = data_input.transpose(0,1).transpose(0,2).to(device, dtype=torch.float)
            data_output = data_output.transpose(0,1).transpose(0,2).to(device, dtype=torch.float)

            output, hidden = model(data_input)
            #hidden = repackage_hidden(hidden)
            #output, hidden = model(data_input,hidden)
            print(torch.sqrt(torch.mean(torch.square((output[LEN_INIT:]-data_output[LEN_INIT:])))/torch.mean(torch.square(output[LEN_INIT:]))))
            loss = criterion(output[LEN_INIT:], data_output[LEN_INIT:])
            val_loss += loss.item()*BATCHSIZE

    val_loss /= len(val_loader.dataset)
    loss_values_val.append(val_loss)

    if val_loss <= min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), 'MG/weights/'+MODEL+'_best_val_quicksave.pt')
    
    stop_time = time.time()
    time_el = int(stop_time-start_time)
    
    print('epoch [{}/{}], loss:{:.7f}, val loss:{:.7f}, in {}h {}m {}s'.format(epoch, NUMEPOCHS,
                                                                                runloss, val_loss,                                                                                time_el//3600,
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
fig.savefig('MG/'+MODEL+'train_curve.png')

hidden = model.init_hidden(BATCHSIZE,LEN_TOTAL)
test_loss = 0.0
with torch.no_grad():
    for data_input, data_output in test_loader:
        data_input = data_input.transpose(0,1).transpose(0,2).to(device, dtype=torch.float)
        data_output = data_output.transpose(0,1).transpose(0,2).to(device, dtype=torch.float)
        output, hidden = model(data_input)
        #hidden = repackage_hidden(hidden)
        #output, hidden = model(data_input,hidden)
        print(torch.sqrt(torch.mean(torch.square((output[LEN_INIT:]-data_output[LEN_INIT:])))/torch.mean(torch.square(output[LEN_INIT:]))))
        loss = torch.sqrt(torch.mean(torch.square((output[50:]-data_output[50:])/output[50:])))
        test_loss += loss.item()*BATCHSIZE
test_loss = test_loss/len(test_loader.dataset)
print('Test loss ',test_loss)