import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import random

np.random.seed(7)
random.seed(7)


def mackey_glass(length=10000, x0=None, a=0.2, b=0.1, c=10.0, tau=17.0,
                 n=1000, sample=0.46, discard=250):
    """Generate time series using the Mackey-Glass equation.
    Generates time series using the discrete approximation of the
    Mackey-Glass delay differential equation described by Grassberger &
    Procaccia (1983).
    Parameters
    ----------
    length : int, optional (default = 10000)
        Length of the time series to be generated.
    x0 : array, optional (default = random)
        Initial condition for the discrete map.  Should be of length n.
    a : float, optional (default = 0.2)
        Constant a in the Mackey-Glass equation.
    b : float, optional (default = 0.1)
        Constant b in the Mackey-Glass equation.
    c : float, optional (default = 10.0)
        Constant c in the Mackey-Glass equation.
    tau : float, optional (default = 23.0)
        Time delay in the Mackey-Glass equation.
    n : int, optional (default = 1000)
        The number of discrete steps into which the interval between
        t and t + tau should be divided.  This results in a time
        step of tau/n and an n + 1 dimensional map.
    sample : float, optional (default = 0.46)
        Sampling step of the time series.  It is useful to pick
        something between tau/100 and tau/10, with tau/sample being
        a factor of n.  This will make sure that there are only whole
        number indices.
    discard : int, optional (default = 250)
        Number of n-steps to discard in order to eliminate transients.
        A total of n*discard steps will be discarded.
    Returns
    -------
    x : array
        Array containing the time series.
    """
    sample = int(n * sample / tau)
    grids = n * discard + sample * length
    x = np.empty(grids)

    if not x0:
        x[:n] = 0.5 + 0.05 * (-1 + 2 * np.random.random(n))
    else:
        x[:n] = x0

    A = (2 * n - b * tau) / (2 * n + b * tau)
    B = a * tau / (2 * n + b * tau)

    for i in range(n - 1, grids - 1):
        x[i + 1] = A * x[i] + B * (x[i - n] / (1 + x[i - n] ** c) +
                                   x[i - n + 1] / (1 + x[i - n + 1] ** c))
    return x[n * discard::sample]


class mg():

    def __init__(self, path, seq_len, future_predict=15):
    	"""
    	path: where to dump pickle file so that series need not be generated again and again
    	seq_len: length of each time series. To test memory retention, it should be as high as the GPU allows
    	future_predict: how many time steps in the future are we asking the RNN to predict? 15 used in paper

    	Use MSE loss for this task
    	Will need to add an extra dimension before feeding to RNN since our features are scalars. Transform it from (batch, timesteps) to (batch, timesteps, 1) using torch.unsqueeze(-1)

    	"""

        self.db_path = path
        self.seq_len = seq_len
        self.future_predict = future_predict
        self.pkl_name = os.path.join(self.db_path, 'mg', str(seq_len), str(future_predict), '.pkl')

    #Generate and store pickle dump
    def process_one(self, num_sequences):
        print("Generating {} sequences".format(num_sequences))
        inputs = []
        labels = []

        for i in range(num_sequences):
            series = mackey_glass(self.seq_len)
            target = series[self.future_predict:]
            series = series[:-self.future_predict]
            inputs.append(series)
            labels.append(target)

        # inputs and labels both of shape number_of_seq x timesteps
        return inputs, labels


    def process_all(self, number_of_series=[100, 20, 20]):
    	"""
		numebr_of_series: how many different series to generate in the form [train, test, val]. Each series will be different due to random init conditions
    	"""
        if os.path.exists(self.pkl_name):
            print("Found pickle dump")
            with open(self.pkl_name, 'rb') as f:
                return pickle.load(f)

        else:
            final = {}            
            final['train'] = self.process_one(number_of_series[0])
            final['test'] = self.process_one(number_of_series[1])
            final['val'] = self.process_one(number_of_series[2])

            with open(self.pkl_name, 'wb') as f:
                pickle.dump(final, f)
                print("Dumped pickle")

            return final

    def __getitem__(self, idx):


if __name__ == '__main__':
    a = mg('.', 100000, 15)
    a = a.process_all()
    plt.plot(a['train'][0][1][:100], c='r')
    plt.plot(a['train'][1][1][:100], c='b')
    plt.show()