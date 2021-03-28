import numpy as np
import pickle
import os
import string

#Ignore DS_Store files found on Mac
def listdir(pth):
	return [x for x in os.listdir(pth) if x != '.DS_Store']

def one_hot(index, length):	
	a = np.zeros((length), dtype=np.uint8)
	a[index] = 1
	return a

class ptb_char():

	def __init__(self, path='../datasets/ptbdataset'):

		self.db_path = path
		self.pkl_name = os.path.join(self.db_path, 'ptb_char.pkl')
		self.exclude_list = ['\\', '&', '*']
		self.exclude_char = '*'

	#Generate and store pickle dump
	def process_one(self, mode, exclude_char, exclude_list=None, char_to_idx=None):		

		print("Generating pickle dump for")

		with open(os.path.join(self.db_path, 'ptb.char.' + mode + '.txt'), 'r') as f:
			text = f.readlines()

		corpus = ''.join(text).lower().replace(' ', '')
		
		if char_to_idx is None:
			for c in exclude_list:
				corpus = corpus.replace(c, exclude_char)
			unique_chars = set(corpus)
			unique_chars.add(exclude_char)
			print("Length of corpus for {} is {}".format(mode, len(corpus)))
			
			char_to_idx = {ch: n for n, ch in enumerate(unique_chars)}
			print("Mapping: {} ".format(char_to_idx))
		
		vocab_size = len(char_to_idx)
		idx_to_char = {val: key for key, val in char_to_idx.items()}

		inputs = []
		labels = []
		i = 0
		for i, c in enumerate(corpus[:-1]):
			if c not in char_to_idx.keys():
				c = exclude_char
			inputs.append(one_hot(char_to_idx[c], vocab_size))
			labels.append(char_to_idx[corpus[i+1]])

		return inputs, labels, char_to_idx


	def process_all(self):
		if os.path.exists(self.pkl_name):
			print("Found pickle dump")
			with open(self.pkl_name, 'rb') as f:
				return pickle.load(f)

		else:
			final = {}
			tr_inp, tr_lab, char_to_idx = self.process_one('train', self.exclude_char, self.exclude_list)
			final['train'] = (tr_inp, tr_lab, char_to_idx)
			final['test'] = self.process_one('test', self.exclude_char, char_to_idx)
			final['val'] = self.process_one('valid', self.exclude_char, char_to_idx)

			with open(self.pkl_name, 'wb') as f:
				pickle.dump(final, f)
				print("Dumped pickle")

			return final

if __name__ == '__main__':

	a = ptb_char()
	final = a.process_all()
	# print(final['train'][0][:10])
	print(final['train'][2])
	# print(sample_to_frame(63488))