import torch
import torch.nn as nn
import torch.nn.functional as F
from .lmu_layer import LMU 

class McG_LSTM(nn.Module):
    def __init__(self,size=25,BSIZE=32,LEN=512):
        super(McG_LSTM, self).__init__()
        
        self.size = size
        self.lstm_1 = nn.LSTMCell(1,self.size,True)
        self.lstm_2 = nn.LSTMCell(self.size,self.size,True)
        self.lstm_3 = nn.LSTMCell(self.size,self.size,True)
        self.lstm_4 = nn.LSTMCell(self.size,self.size,True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.size, 1)

    def forward(self, x):#, hidden):
        #self.ht1, self.ct1, self.ht2, self.ct2, self.ht3, self.ct3, self.ht4, self.ct4 = hidden 
        self.y = self.y.zero_().detach()
        
        self.ht1 = self.ht1.zero_().detach()
        self.ht2 = self.ht2.zero_().detach()
        self.ht3 = self.ht3.zero_().detach()
        self.ht4 = self.ht4.zero_().detach()
        self.ct1 = self.ct1.zero_().detach()
        self.ct2 = self.ct2.zero_().detach()
        self.ct3 = self.ct3.zero_().detach()
        self.ct4 = self.ct4.zero_().detach()
        
        for i in range(x.size(0)):
            self.ht1, self.ct1 = self.lstm_1(x[i],(self.ht1,self.ct1))
            self.ht2, self.ct2 = self.lstm_2(self.ht1,(self.ht2,self.ct2))
            self.ht3, self.ct3 = self.lstm_3(self.ht2,(self.ht3,self.ct3))
            self.ht4, self.ct4 = self.lstm_4(self.ht3,(self.ht4,self.ct4))
            #self.ht4 = self.dropout(self.ht4)
            self.y[i] = self.fc1(self.ht4)

        return self.y, (self.ht1, self.ct1, self.ht2, self.ct2, self.ht3, self.ct3, self.ht4, self.ct4)
    
    def init_hidden(self,BSIZE,LEN):
        self.register_buffer('ht1', torch.zeros(BSIZE,self.size))
        self.register_buffer('ct1', torch.zeros(BSIZE,self.size))
        self.register_buffer('ht2', torch.zeros(BSIZE,self.size))
        self.register_buffer('ct2', torch.zeros(BSIZE,self.size))
        self.register_buffer('ht3', torch.zeros(BSIZE,self.size))
        self.register_buffer('ct3', torch.zeros(BSIZE,self.size))
        self.register_buffer('ht4', torch.zeros(BSIZE,self.size))
        self.register_buffer('ct4', torch.zeros(BSIZE,self.size))
        self.register_buffer('y', torch.zeros(LEN,BSIZE,1))

        return (self.ht1, self.ct1, self.ht2, self.ct2, self.ht3, self.ct3, self.ht4, self.ct4)



class McG_LMU(nn.Module):
    def __init__(self,n=49,d=4,BSIZE=32,LEN=512):
        super(McG_LSTM, self).__init__()
        
        self.n = n
        self.d = d
        self.lstm_1 = nn.LMU(1,self.n,self.d,4)
        self.lstm_2 = nn.LMU(self.n,self.n,self.d,4)
        self.lstm_3 = nn.LMU(self.n,self.n,self.d,4)
        self.lstm_4 = nn.LMU(self.n,self.n,self.d,4)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.n, 1)

    def forward(self, x):#, hidden):
        #self.ht1, self.ct1, self.ht2, self.ct2, self.ht3, self.ct3, self.ht4, self.ct4 = hidden 
        self.y = self.y.zero_().detach()
        
        self.ht1 = self.ht1.zero_().detach()
        self.ht2 = self.ht2.zero_().detach()
        self.ht3 = self.ht3.zero_().detach()
        self.ht4 = self.ht4.zero_().detach()
        self.ct1 = self.ct1.zero_().detach()
        self.ct2 = self.ct2.zero_().detach()
        self.ct3 = self.ct3.zero_().detach()
        self.ct4 = self.ct4.zero_().detach()
        
        for i in range(x.size(0)):
            self.ht1, self.ct1 = self.lstm_1(x[i],(self.ht1,self.ct1))
            self.ht2, self.ct2 = self.lstm_2(self.ht1,(self.ht2,self.ct2))
            self.ht3, self.ct3 = self.lstm_3(self.ht2,(self.ht3,self.ct3))
            self.ht4, self.ct4 = self.lstm_4(self.ht3,(self.ht4,self.ct4))
            #self.ht4 = self.dropout(self.ht4)
            self.y[i] = self.fc1(self.ht4)

        return self.y, (self.ht1, self.ct1, self.ht2, self.ct2, self.ht3, self.ct3, self.ht4, self.ct4)
    
    def init_hidden(self,BSIZE,LEN):
        self.register_buffer('ht1', torch.zeros(BSIZE,self.size))
        self.register_buffer('ct1', torch.zeros(BSIZE,self.size))
        self.register_buffer('ht2', torch.zeros(BSIZE,self.size))
        self.register_buffer('ct2', torch.zeros(BSIZE,self.size))
        self.register_buffer('ht3', torch.zeros(BSIZE,self.size))
        self.register_buffer('ct3', torch.zeros(BSIZE,self.size))
        self.register_buffer('ht4', torch.zeros(BSIZE,self.size))
        self.register_buffer('ct4', torch.zeros(BSIZE,self.size))
        self.register_buffer('y', torch.zeros(LEN,BSIZE,1))

        return (self.ht1, self.ct1, self.ht2, self.ct2, self.ht3, self.ct3, self.ht4, self.ct4)