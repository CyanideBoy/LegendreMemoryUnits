import torch
import torch.nn as nn
import torch.nn.functional as F
from nengolib.signal import Identity, cont2discrete
from nengolib.synapses import LegendreDelay
import numpy as np


class LMU(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, theta, nonlinearity='sigmoid', A_learnable = False, B_learnable = False):
        super(LMU, self).__init__()
        
        ### SIZE
        self.k = input_size
        self.n = hidden_size
        self.d = memory_size

        ### PARAMETERS
        self.Wx = nn.Parameter(torch.Tensor(self.n,self.k))
        self.Wh = nn.Parameter(torch.Tensor(self.n,self.n))
        self.Wm = nn.Parameter(torch.Tensor(self.n,self.d))
        self.ex = nn.Parameter(torch.Tensor(1,self.k))
        self.eh = nn.Parameter(torch.Tensor(1,self.n))
        self.em = nn.Parameter(torch.Tensor(1,self.d))

        ### A,B MATRIX
        realizer = Identity()
        self._realizer_result = realizer(LegendreDelay(theta=theta, order=self.d))
        self._ss = cont2discrete(self._realizer_result.realization, dt=1., method='zoh')
        self._A = self._ss.A
        self._B = self._ss.B
        self.AT = torch.Tensor(self._A)
        self.BT = torch.Tensor(self._B)
        if A_learnable:
            self.AT = nn.Parameter(self.AT)
        if B_learnable:
            self.BT = nn.Parameter(self.BT)

        ### NON-LINEARITY
        self.nl = nonlinearity
        if self.nl == 'sigmoid':
            self.act = nn.Sigmoid()
        elif self.nl == 'tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.ReLU()

        ### INITIALIZATION
        torch.nn.init.xavier_normal_(self.Wm)    ###???
        torch.nn.init.xavier_normal_(self.Wx)
        torch.nn.init.xavier_normal_(self.Wh)
        torch.nn.init.zeros_(self.em)
        torch.nn.init.uniform_(self.ex, -np.sqrt(3/self.d), np.sqrt(3/self.d))
        torch.nn.init.uniform_(self.eh, -np.sqrt(3/self.d), np.sqrt(3/self.d))
        
    def forward(self,x,hm):
        h,m = hm 
        '''
        x = (B,self.k) 
        h = (B,self.n)
        m = (B,self.d) 
        '''

        u = F.linear(x,self.ex)+F.linear(h,self.eh)+F.linear(m,self.em)
        new_m = F.linear(m,self.AT) + F.linear(u,self.BT)
        new_h = self.act(F.linear(x,self.Wx)+F.linear(h,self.Wh)+F.linear(new_m,self.Wm))

        return new_h,new_m