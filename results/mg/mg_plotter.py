import numpy as np
import matplotlib.pyplot as plt

lmu_train,lmu_test,lmu_act,lmu_pred = np.load('LMU_data.npy',allow_pickle=True)
lstm_train,lstm_test,lstm_act,lstm_pred = np.load('LSTM_data.npy',allow_pickle=True)
hybrid_train,hybrid_test,hybrid_act,hybrid_pred = np.load('Hybrid_data.npy',allow_pickle=True)
bmu_train,bmu_test,bmu_act,bmu_pred = np.load('BMU_data.npy',allow_pickle=True)
assvmu_train,assvmu_test,assvmu_act,assvmu_pred = np.load('ASSVMU_data.npy',allow_pickle=True)


fig = plt.figure()
plt.plot(lmu_act[:120], 'black')
plt.plot(lstm_pred[:120], ':',color='#d7191c',linewidth='1.3')
plt.plot(lmu_pred[:120], ':',color='#fdae61',linewidth='1.3')
plt.plot(hybrid_pred[:120], ':',color='green',linewidth='1.3')
plt.plot(bmu_pred[:120], ':',color='#abd9e9',linewidth='1.3')
plt.plot(assvmu_pred[:120], ':',color='#2c7bb6',linewidth='1.3')

plt.legend(['True','LSTM','LMU','Hybrid','BMU','ASSVMU'])
plt.grid(True)
plt.xlabel('Time Step')
plt.ylabel('Mackey Glass Curve')
fig.savefig('CombinedGraph.png', dpi=800)