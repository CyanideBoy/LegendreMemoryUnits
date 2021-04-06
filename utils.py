import numpy as np
import matplotlib.pyplot as plt

dataset = 'ptb'
res_path = 'results/{}/'.format(dataset)

lstm_train, lstm_val = np.load(res_path+'lstm_data.npy')
lmu_train, lmu_val = np.load(res_path+'lmu_data.npy')

fig = plt.figure()
plt.plot(np.array(lstm_train), 'b', marker='x', label="LSTM train")
plt.plot(np.array(lstm_val), 'r', marker='x', label="LSTM val")
plt.plot(np.array(lmu_train), 'b', marker='o', label="LMU train")
plt.plot(np.array(lmu_val), 'r', marker='o', label="LMU val")
plt.legend()
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Loss')
fig.savefig(res_path+'combined_train_curve.png')

