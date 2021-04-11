import numpy as np
import matplotlib.pyplot as plt

lmu_train,lmu_val,lmu_act,lmu_pred = np.load('LMU_data.npy',allow_pickle=True)
lstm_train,lstm_val,lstm_act,lstm_pred = np.load('LSTM_data.npy',allow_pickle=True)
hybrid_train,hybrid_val,hybrid_act,hybrid_pred = np.load('Hybrid_data.npy',allow_pickle=True)



fig = plt.figure()
plt.plot(lmu_act[:120], 'black')
plt.plot(lstm_pred[:120], ':',color='blue',linewidth='1.3')
plt.plot(lmu_pred[:120], ':',color='red',linewidth='1.3')
plt.plot(hybrid_pred[:120], ':',color='orange',linewidth='1.3')

plt.legend(['True','LSTM','LMU','Hybrid'])
plt.grid(True)
plt.xlabel('Time Step')
plt.ylabel('Mackey Glass Curve')
fig.savefig('CombinedGraph.png')