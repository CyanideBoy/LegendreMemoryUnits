import torch
import torch.nn as nn
import torch.nn.functional as F
from nengolib.signal import Identity, cont2discrete
from nengolib.synapses import LegendreDelay
import numpy as np


class LMU(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, theta, discretizer = 'zoh',nonlinearity='sigmoid', A_learnable = False, B_learnable = False):
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

        ### A,B MATRIX ----- FIX??
        realizer = Identity()
        self._realizer_result = realizer(LegendreDelay(theta=theta, order=self.d))
        self._ss = cont2discrete(self._realizer_result.realization, dt=1., method=discretizer)
        self._A = self._ss.A
        self._B = self._ss.B
        '''
        Q = np.arange(order, dtype=np.float64)
        R = (2 * Q + 1)[:, None] / theta
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R
        C = np.ones((1, order))
        D = np.zeros((1,))
        self._A, self._B, _, _, _ = cont2discrete((A, B, C, D), dt=1.0, method="zoh")
        '''
        '''
        self.AT = torch.Tensor(self._A)
        self.BT = torch.Tensor(self._B)
        if A_learnable:
            self.AT = nn.Parameter(self.AT)
        if B_learnable:
            self.BT = nn.Parameter(self.BT)
        '''
        ### NON-LINEARITY
        self.nl = nonlinearity
        if self.nl == 'sigmoid':
            self.act = nn.Sigmoid()
        elif self.nl == 'tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.ReLU()

        ### INITIALIZATION
        torch.nn.init.xavier_normal_(self.Wm)    ##### FIGURE THIS OUT!!
        torch.nn.init.xavier_normal_(self.Wx)
        torch.nn.init.xavier_normal_(self.Wh)
        torch.nn.init.zeros_(self.em)
        torch.nn.init.uniform_(self.ex, -np.sqrt(3/self.d), np.sqrt(3/self.d))
        torch.nn.init.uniform_(self.eh, -np.sqrt(3/self.d), np.sqrt(3/self.d))


        #### TRIAL
        self.register_buffer('AT', torch.Tensor(self._A))
        self.register_buffer('BT', torch.Tensor(self._B))
        if A_learnable:
            self.AT = nn.Parameter(self.AT)
        if B_learnable:
            self.BT = nn.Parameter(self.BT)

        
    def forward(self,x,hm):
        '''
        x shape: (batch_size, input_size) 
        h shape: (batch_size, hidden_size)
        m shape: (batch_size, memory_size) 
        '''

        h,m = hm 
        u = F.linear(x,self.ex)+F.linear(h,self.eh)+F.linear(m,self.em)
        new_m = F.linear(m,self.AT) + F.linear(u,self.BT)
        new_h = self.act(F.linear(x,self.Wx)+F.linear(h,self.Wh)+F.linear(new_m,self.Wm))

        return new_h,new_m




class BhadwaLMU(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, theta, discretizer = 'zoh',nonlinearity='sigmoid', 
                        A_learnable = False, B_learnable = False, include_both=False, activate=False):
        super(BhadwaLMU, self).__init__()
        
        ### SIZE
        self.k = input_size
        self.n = hidden_size
        self.d = memory_size

        ###
        self.include_both = include_both
        

        ### PARAMETERS
        self.Wx = nn.Parameter(torch.Tensor(self.n,self.k))
        self.Wh = nn.Parameter(torch.Tensor(self.n,self.n))
        self.Wm = nn.Parameter(torch.Tensor(self.n,self.d))
        self.ex = nn.Parameter(torch.Tensor(1,self.k))
        self.eh = nn.Parameter(torch.Tensor(1,self.n))
        self.em = nn.Parameter(torch.Tensor(1,self.d))

        ### A,B MATRIX ----- FIX??
        realizer = Identity()
        self._realizer_result = realizer(LegendreDelay(theta=theta, order=self.d))
        self._ss = cont2discrete(self._realizer_result.realization, dt=1., method=discretizer)
        self._A = self._ss.A
        self._B = self._ss.B

        ### NON-LINEARITY
        self.nl = nonlinearity
        if self.nl == 'sigmoid':
            self.act = nn.Sigmoid()
        elif self.nl == 'tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.ReLU()

        ### NN
        if include_both:
            self.fc = nn.Linear(self.n+self.d,self.n+self.d)
        else:
            self.fc = nn.Linear(self.n,self.n)

        if activate:
            self.nn_act = self.act
        else:
            self.nn_act = nn.LeakyReLU(1.0) #Identity Function

        ### INITIALIZATION
        torch.nn.init.xavier_normal_(self.Wm)    ##### FIGURE THIS OUT!!
        torch.nn.init.xavier_normal_(self.Wx)
        torch.nn.init.xavier_normal_(self.Wh)
        torch.nn.init.zeros_(self.em)
        torch.nn.init.uniform_(self.ex, -np.sqrt(3/self.d), np.sqrt(3/self.d))
        torch.nn.init.uniform_(self.eh, -np.sqrt(3/self.d), np.sqrt(3/self.d))


        #### TRIAL
        self.register_buffer('AT', torch.Tensor(self._A))
        self.register_buffer('BT', torch.Tensor(self._B))
        if A_learnable:
            self.AT = nn.Parameter(self.AT)
        if B_learnable:
            self.BT = nn.Parameter(self.BT)

    def forward_both(self,x,hm):
        '''
        x shape: (batch_size, input_size) 
        h shape: (batch_size, hidden_size)
        m shape: (batch_size, memory_size) 
        '''

        h,m = hm 
        u = F.linear(x,self.ex)+F.linear(h,self.eh)+F.linear(m,self.em)
        new_m = F.linear(m,self.AT) + F.linear(u,self.BT)
        new_h = self.act(F.linear(x,self.Wx)+F.linear(h,self.Wh)+F.linear(new_m,self.Wm))
        new_temp = self.nn_act(self.fc(torch.cat((new_h,new_m),1)))
        return new_temp[:,self.n],new_temp[:,self.n:]
        
    def forward_onlyh(self,x,hm):
        '''
        x shape: (batch_size, input_size) 
        h shape: (batch_size, hidden_size)
        m shape: (batch_size, memory_size) 
        '''

        h,m = hm 
        u = F.linear(x,self.ex)+F.linear(h,self.eh)+F.linear(m,self.em)
        new_m = F.linear(m,self.AT) + F.linear(u,self.BT)
        new_h = self.act(F.linear(x,self.Wx)+F.linear(h,self.Wh)+F.linear(new_m,self.Wm))
        new_h = self.nn_act(self.fc(new_h))
        return new_h,new_m

    def forward(self,x,hm):
        '''
        x shape: (batch_size, input_size) 
        h shape: (batch_size, hidden_size)
        m shape: (batch_size, memory_size) 
        '''

        h,m = hm 
        u = F.linear(x,self.ex)+F.linear(h,self.eh)+F.linear(m,self.em)
        new_m = F.linear(m,self.AT) + F.linear(u,self.BT)
        new_h = self.act(F.linear(x,self.Wx)+F.linear(h,self.Wh)+F.linear(new_m,self.Wm))
        if self.include_both:
            new_temp = self.nn_act(self.fc(torch.cat((new_h,new_m),1)))
            return new_temp[:,self.n],new_temp[:,self.n:]
        else:
            new_temp = self.nn_act(self.fc(new_h))
            return new_temp,new_m