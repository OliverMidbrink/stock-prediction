from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np
import random

def scalar_augment(X_, min_scalar=0.5, max_scalar=2):
	scalar = min_scalar + random.random() * (max_scalar - min_scalar)
	return X_ * scalar


class CustomSequence(Sequence):
	def __init__(self, X_, Y_, batch_size, augment):
		self.X, self.Y = X_, Y_
		self.batch_size = batch_size
		self.augment = augment

	def __len__(self):
		return int(np.ceil(len(self.X) / float(self.batch_size)))

	def __getitem__(self, idx):
		while True:
			batch_x = []
			batch_y = []
			
			for i in range(len(self.X)):					
				batch_x.append(self.augment(self.X[i]))	# Append augmented X
				batch_y.append(self.Y[i])
					
				if len(batch_x) == self.batch_size:
					yield np.array(batch_x), np.array(batch_y)
					batch_x = []
					batch_y = []						
						