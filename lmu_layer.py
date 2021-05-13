# Some of the code taken from
# https://github.com/nengo/keras-lmu/blob/master/keras_lmu/layers.py
# https://github.com/bjkomer/pytorch-legendre-memory-unit/blob/master/lmu.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from nengolib.signal import cont2discrete
import numpy as np
from scipy.special import comb


class LMU(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, theta, matrix_type='pl',
                discretizer = 'zoh',nonlinearity='sigmoid', A_learnable = False, B_learnable = False):
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

        if matrix_type=='pl':   #For Legendre Memory Unit
            order=self.d
            Q = np.arange(order, dtype=np.float64)
            R = (2 * Q + 1)[:, None] / theta
            j, i = np.meshgrid(Q, Q)
            A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
            B = (-1.0) ** Q[:, None] * R
            C = np.ones((1, order))
            D = np.zeros((1,))
            self._ss = cont2discrete((A, B, C, D), dt=0.01, method=discretizer)
            self._A = self._ss.A
            self._B = self._ss.B
        elif matrix_type=='p':  #For Pade Memory Unit
            order=self.d
            Q=np.arange(order,dtype=np.float64)
            V=(order+Q+1)*(order-Q)/(Q+1)/theta
            A=np.zeros([order,order],dtype=np.float64)
            B=np.zeros([order,1],dtype=np.float64)
            A[0,:]=-V[0]
            A[1:order,0:order-1]=np.diag(V[1:order])
            B[0]=V[0]
            C = np.ones((1, order))
            D = np.zeros((1,))
            self._ss = cont2discrete((A, B, C, D), dt=0.01, method=discretizer)
            self._A = self._ss.A
            self._B = self._ss.B
        elif matrix_type=='pb':  #For Bernstein Memory Unit
            order=self.d
            Q = np.arange(order, dtype=np.float64)
            R = (2 * Q + 1)[:, None] / theta
            j, i = np.meshgrid(Q, Q)
            A_leg = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
            B_leg = (-1.0) ** Q[:, None] * R
            C = np.ones((1, order))
            D = np.zeros((1,))
            M=np.zeros([order,order],dtype=np.float64)
            M_inv=np.zeros([order,order],dtype=np.float64)
            n=order-1 #degree of polynomial
            for j in range(0,n+1):
              for k in range(0,n+1):
                ll=max(0,j+k-n)
                ul=min(j,k)+1
                sum=0.0
                for i in range(ll,ul):
                  sum=sum+((-1.0)**(k+i))*(comb(k,i)**2)*comb(n-k,j-i)
                M[j,k]=sum/comb(n,j)

                sum=0.0
                for i in range(0,j+1):
                  sum=sum+(-1.0)**(j+i)*comb(j,i)**2/comb(n+j,k+i)
                M_inv[j,k]=(2*j+1)/(n+j+1)*comb(n,k)*sum

            M=10*np.tanh(M/10)
            M_inv=10*np.tanh(M_inv/10)

            A_1=np.matmul(M,A_leg)
            A=np.matmul(A_1,M_inv)
            B=np.matmul(M,B_leg)

            self._ss = cont2discrete((A, B, C, D), dt=0.01, method=discretizer)
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

        ### INITIALIZATION
        torch.nn.init.xavier_normal_(self.Wm)    ### UNDOCUMENTED CHANGE!
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



class ASSVMU(nn.Module):
    '''
    Also known as Linear LMU
    '''
    def __init__(self, input_size, hidden_size, memory_size, theta, name='garbage', discretizer = 'zoh',nonlinearity='sigmoid', 
                        A_learnable = False, B_learnable = False, activate=False):
        super(ASSVMU, self).__init__()
        
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
        order=self.d
        Q = np.arange(order, dtype=np.float64)
        R = (2 * Q + 1)[:, None] / theta
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R
        C = np.ones((1, order))
        D = np.zeros((1,))
        self._ss = cont2discrete((A, B, C, D), dt=0.01, method=discretizer)
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
        new_h = self.nn_act(self.fc(new_h))
        return new_h,new_m
