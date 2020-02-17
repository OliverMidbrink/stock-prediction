from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np
import random

'''
def scalar_augment(X_, min_scalar=0.5, max_scalar=2):
	scalar = min_scalar + random.random() * (max_scalar - min_scalar)
	return X_ * scalar
'''

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
				X_, Y_ = self.augment(self.X[i], self.Y[i])				
				batch_x.append(X_)	# Append augmented X
				batch_y.append(Y_)
					
				if len(batch_x) == self.batch_size:
					yield np.array(batch_x), np.array(batch_y)
					batch_x = []
					batch_y = []						
						





'''			TO ANALYZE THE TRAINING DATA
sum_values = np.zeros((len(X_train), 2))
for x in range(len(X_train)):
		sum_values[x][0] = np.sum(np.sum(X_train[x]) / (6 * hist_time_steps + 6 * pred_time_steps))
		sum_values[x][1] = np.sum(np.sum(Y_train[x]) / pred_time_steps)

explore_df = pd.DataFrame(data=sum_values, columns=['Sum X', 'Sum Y'])
print(explore_df)
print('X avg mean: {}.'.format(explore_df['Sum X'].mean()))
print('Y avg mean: {}.'.format(explore_df['Sum Y'].mean()))
print('X avg min: {}.'.format(explore_df['Sum X'].min()))
print('Y avg min: {}.'.format(explore_df['Sum Y'].min()))
print('X avg max: {}.'.format(explore_df['Sum X'].max()))
print('Y avg max: {}.'.format(explore_df['Sum Y'].max()))
'''