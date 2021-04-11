import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import random
import torch
import gc

np.random.seed(7)
random.seed(7)

def mackey_glass(length=20000, x0=None, a=0.2, b=0.1, c=10.0, e=10, tau=17):
    a = 0.2
    b = 0.8
    c = 0.9
    e = 10
    x = np.empty(length)
    x[:len(x0)] = x0
        
    for i in range(len(x0), length):
        x[i] = c * x[i-1] + a * x[i -1 - tau] / (b + x[i -1 - tau] ** e) 
    return x

class MGDataset:

    def __init__(self, path, seq_len, future_predict=15, number_of_series=[2, 2, 2], x0 = None):
        """
        path: where to dump pickle file so that series need not be generated again and again
        seq_len: length of each time series. To test memory retention, it should be as high as the GPU allows ???
        future_predict: how many time steps in the future are we asking the RNN to predict? 15 used in paper
        number_of_series: how many different series to generate in the form [train, test, val].

        Use NRMSE loss for this task
        """

        self.db_path = path
        self.seq_len = seq_len
        self.future_predict = future_predict
        if x0 is None:
            self.x0 = np.array([0.9697, 0.9699, 0.9794, 1.0003, 
                     1.0319, 1.0703, 1.1076, 1.1352, 
                     1.1485,1.1482, 1.1383, 1.1234, 
                     1.1072, 1.0928, 1.0820, 1.0756, 
                     1.0739, 1.0759])
        else:
            self.x0 = x0
        self.number_of_series=number_of_series
        if not os.path.exists(self.db_path):
            os.mkdir(self.db_path)
        if not os.path.exists(os.path.join(self.db_path, 'mg')):
            os.mkdir(os.path.join(self.db_path, 'mg'))
        self.pkl_name = os.path.join(self.db_path, 'mg')+'/'+str(seq_len)+'_'+ str(future_predict)+'_'+ ('-').join([str(x) for x in number_of_series])+'.pkl'
        print(self.pkl_name)
    
    def get_dataset(self):
        
        if os.path.exists(self.pkl_name):
            print("Found pickle dump")
            with open(self.pkl_name, 'rb') as f:
                return pickle.load(f)

        self.sequence = mackey_glass(sum(self.number_of_series)*(self.seq_len+self.future_predict), self.x0)
        self.seq_train = self.sequence[0:self.number_of_series[0]*(self.seq_len+self.future_predict)]
        self.seq_val = self.sequence[self.number_of_series[0]*(self.seq_len+self.future_predict):sum(self.number_of_series[:2])*(self.seq_len+self.future_predict)]
        self.seq_test = self.sequence[sum(self.number_of_series[:2])*(self.seq_len+self.future_predict):]

        del self.sequence
        gc.collect()

        self.x_train = np.zeros([self.number_of_series[0],1,self.seq_len])
        self.y_train = np.zeros([self.number_of_series[0],1,self.seq_len])
        self.x_val = np.zeros([self.number_of_series[1],1,self.seq_len])
        self.y_val = np.zeros([self.number_of_series[1],1,self.seq_len])
        self.x_test = np.zeros([self.number_of_series[2],1,self.seq_len])
        self.y_test = np.zeros([self.number_of_series[2],1,self.seq_len])
        
        for i in range(self.number_of_series[0]): 
            self.x_train[i,0,:] = self.seq_train[i*(self.seq_len+self.future_predict):i*(self.seq_len+self.future_predict)+self.seq_len]
            self.y_train[i,0,:] = self.seq_train[i*(self.seq_len+self.future_predict)+self.future_predict:(i+1)*(self.seq_len+self.future_predict)]
        
        for i in range(self.number_of_series[1]):
            self.x_val[i,0,:] = self.seq_val[i*(self.seq_len+self.future_predict):i*(self.seq_len+self.future_predict)+self.seq_len]
            self.y_val[i,0,:] = self.seq_val[i*(self.seq_len+self.future_predict)+self.future_predict:(i+1)*(self.seq_len+self.future_predict)]
        
        for i in range(self.number_of_series[2]): 
            self.x_test[i,0,:] = self.seq_test[i*(self.seq_len+self.future_predict):i*(self.seq_len+self.future_predict)+self.seq_len]
            self.y_test[i,0,:] = self.seq_test[i*(self.seq_len+self.future_predict)+self.future_predict:(i+1)*(self.seq_len+self.future_predict)]
        
        del self.seq_train
        del self.seq_val
        del self.seq_test
        gc.collect()

        data = {}
        data['train'] = (torch.from_numpy(self.x_train),torch.from_numpy(self.y_train))
        data['val'] = (torch.from_numpy(self.x_val),torch.from_numpy(self.y_val))
        data['test'] = (torch.from_numpy(self.x_test),torch.from_numpy(self.y_test))
        

        with open(self.pkl_name, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("Dumped pickle")
        
        return data


if __name__ == '__main__':
    a = MGDataset('.', 200, 15)
    data = a.get_dataset()
    plt.plot(list(range(200)),data['train'][0][0,0,:], c='r')
    plt.plot(list(range(200)),data['val'][0][0,0,:], c='b')
    plt.plot(list(range(15,215)),data['val'][1][0,0,:], c='black')
    plt.show()