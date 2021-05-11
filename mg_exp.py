import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from dataloaders.mg_generator import MGDataset
import models
import torch.optim as optim
import time
import sys

LEARNINGRATE = 1e-2
GAMMA = 0.98
BATCHSIZE = 256
LEN_INIT = 0
LEN_TOTAL = 256 + LEN_INIT
NUMEPOCHS = 100
PREDICT = 15
SPLIT = [4096,256,256]
MODEL = 'LSTM'
base_pth = 'results/mg/'


print('Loaded datasets....')

MG = MGDataset('datasets', LEN_TOTAL, PREDICT, SPLIT)
data = MG.get_dataset() 
train_data = TensorDataset(data['train'][0],data['train'][1])
val_data = TensorDataset(data['val'][0],data['val'][1])
test_data = TensorDataset(data['test'][0],data['test'][1])

train_loader = DataLoader(train_data,batch_size=BATCHSIZE,shuffle=True)
val_loader = DataLoader(val_data,batch_size=BATCHSIZE,shuffle=True)
test_loader = DataLoader(test_data,batch_size=BATCHSIZE,shuffle=False)

print('Using Model',MODEL)

if MODEL == 'LSTM':
    model = models.LSTM_general(inp_features=1)

elif MODEL == 'LMU':
    model = models.LMU_general(inp_features=1,name='LMU')

elif MODEL == 'Hybrid':
    model = models.Hybrid_general()

elif MODEL == 'BMU':
    model = models.LMU_general(inp_features=1,name='BMU')

elif MODEL == 'ASSVMU':
    model = models.LMU_general(inp_features=1,name='ASSVMU')

hidden = model.init_hidden(BATCHSIZE,LEN_TOTAL)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on',device)
print('Building model..')	
model.to(device)
print('Model Built.')

print('Initializing optimizer and scheduler..')
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNINGRATE)             # OR RAdam/DiffGrad/etc
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = GAMMA)

min_test_loss = float('inf')
loss_values_train = []
loss_values_test = []
epoch_times = []

actual = 0
predicted = 0

for epoch in range(1,NUMEPOCHS+1):
    start_time = time.time()
    
    model.train()
    runloss = 0.0

    for data_input, data_output in train_loader:    

        optimizer.zero_grad()
        data_input = data_input.transpose(0,1).transpose(0,2).to(device, dtype=torch.float)  
        data_output = data_output.transpose(0,1).transpose(0,2).to(device, dtype=torch.float)
        
        output, hidden = model(data_input)
        #loss = torch.sqrt(torch.mean(torch.square((output[LEN_INIT:]-data_output[LEN_INIT:])))/torch.mean(torch.square(output[LEN_INIT:])))
        loss = criterion(output[LEN_INIT:], data_output[LEN_INIT:])
        runloss += loss.item()*BATCHSIZE

        loss.backward()
        optimizer.step()
        runloss += loss.item()

    stop_time = time.time()
    time_el = int(stop_time-start_time)
    epoch_times.append(time_el)

    runloss /= len(train_loader.dataset)
    loss_values_train.append(runloss)
    
    model.eval()
    
    test_loss = 0.0
    with torch.no_grad():
        for data_input, data_output in test_loader:
            data_input = data_input.transpose(0,1).transpose(0,2).to(device, dtype=torch.float)
            data_output = data_output.transpose(0,1).transpose(0,2).to(device, dtype=torch.float)
            output, hidden = model(data_input)
            loss = torch.sqrt(torch.mean(torch.square((output[LEN_INIT:]-data_output[LEN_INIT:])))/torch.mean(torch.square(data_output[LEN_INIT:])))
            test_loss += loss.item()*BATCHSIZE
            
            actual = data_output.cpu().numpy()[:,0,0]
            predicted = output.cpu().numpy()[:,0,0]

    test_loss = test_loss/len(test_loader.dataset)
    #print('Test NRMSE :',test_loss)
    loss_values_test.append(test_loss)

    if test_loss <= min_test_loss:
        min_test_loss = test_loss
        torch.save(model.state_dict(), base_pth+MODEL+'_best_test.pt')
    
    print('epoch [{}/{}], loss:{:.7f}, test loss:{:.7f}, in {}h {}m {}s'.format(epoch, NUMEPOCHS,
                                                                                runloss, test_loss, time_el//3600,
                                                                                (time_el%3600)//60,
                                                                                time_el%60))
    scheduler.step()

print("------ TRAINING LOOP FINISHED -------")
print('Mean Epoch Time :',sum(epoch_times)/len(epoch_times))

fig = plt.figure()
plt.plot(list(range(len(loss_values_train))),np.array(loss_values_train), 'b')
plt.plot(list(range(len(loss_values_test))),np.array(loss_values_test), 'r')
plt.legend(['Train','Test'])
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Loss')
fig.savefig(base_pth+MODEL+'_train_curve.png')
np.save(base_pth+MODEL+'_data.npy',(loss_values_train,loss_values_test,actual,predicted))