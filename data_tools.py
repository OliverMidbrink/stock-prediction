from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np
import random
import os, time, sys, h5py, sklearn
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# Use: https://stackoverflow.com/questions/25655588/incremental-writes-to-hdf5-with-h5py?rq=1

'''
def scalar_augment(X_, min_scalar=0.5, max_scalar=2):
	scalar = min_scalar + random.random() * (max_scalar - min_scalar)
	return X_ * scalar
'''
def download_symbols(symbols_list, start_date, end_date):
	batch_size = 100
	df_list = [None] * int(len(symbols_list)/batch_size + 1)

	idx = -1 
	for x in range(0, len(symbols_list), batch_size):
		idx += 1
		to_download = ""
		for sym in symbols_list[x:x+batch_size]:
			to_download += sym + " "

		print('{} Symbols have been iterated. Downloading {} additional symbols.'.format(x, len(symbols_list[x:x+batch_size])))
		try:
			df_list[idx] = yf.download(to_download, start=start_date, end=end_date)
		except Exception as e:
			print(e)
		print('Done.')

	full_df = pd.concat(df_list, axis=1, sort=False)
	return full_df


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
						
class RandomSequence(Sequence):
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

			random_indices = list(range(len(self.X)))
			random.shuffle(random_indices)
			
			for i in random_indices:
				X_, Y_ = self.augment(self.X[i], self.Y[i])				
				batch_x.append(X_)	# Append augmented X
				batch_y.append(Y_)
					
				if len(batch_x) == self.batch_size:
					yield np.array(batch_x), np.array(batch_y)
					batch_x = []
					batch_y = []


def trim_zeros(X_, Y_):
	idx = len(X_)
	print('Before cut: {}.'.format(idx))
	searching = True
	
	while searching:
		idx += -1
		if np.count_nonzero(X_[idx]) > 0 and np.count_nonzero(Y_[idx]) > 0:
			print('End Found at Index: {}.'.format(idx))
			X_ = X_[:idx+1]
			Y_ = Y_[:idx+1]
			searching = False

		if idx == 0:
			print('Dataset seems to only contain zeros')
			searching = False

	print('After cut: {}.'.format(len(X_)))





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

# model should predict: 
	# (1 market days) - might be better to create a new day_trading model instead of interday model
	# 3 market days - half a week
	# 5 market days - a week
	# 10 market days - around two weeks
	# 20 market days - around a month
	# 65 market days - around a quarter
	# 260 market days - around a year


def create_sliding_hdf5(output_filename, raw_dataset, hist_time_steps=500, stride=30, pred_time_steps=[1, 3, 5, 10, 20, 65]): #Create a sliding training dataset. Latest time period will be added to the validation set.  
	start_time = time.time()																								  #network therefore starts training at t_now - longest_pred_time_step - stride (could be 85 market days after specified time if 65 pred and 20 stride).

	print('Reading data file...')
	df = pd.read_hdf(raw_dataset, 'df')
	df.columns = df.columns.swaplevel(0, 1)
	print('Done.')

	# Data Variables - key order for df is SYMBOL, VARIABLE, DATE 
	variables = df.columns.get_level_values(1).unique()
	symbols = df.columns.get_level_values(0).unique()

	# Get Normalized Data
	n_file = raw_dataset[:-3] + '-NormalizedAndLevelChange.h5'
	n_df = None

	if not os.path.isfile(n_file):
		print('Normalizing data...')
		n_df = df.copy()
		n_df = n_df.fillna(method='ffill')
		n_df = n_df.fillna(0)
		for sym in symbols:	#iterate through stocks and normalize data for that stock
			v_df = n_df[sym]
			nv_df=(v_df-v_df.min())/(v_df.max()-v_df.min())
			n_df[sym] = nv_df

		n_df.to_hdf(n_file, 'df', mode='w', format='fixed')
		print('Normalized data file saved.')
	else:
		print('Loading normalized data, since it already exists.')
		n_df = pd.read_hdf(n_file, 'df')
		print('Done.')

	del df # To save memory

	max_length = int(len(n_df) / (stride) * len(symbols) * 1.1)	# Total amount of data periods 

	chunk_size = 1000

	hf = h5py.File(output_filename, 'w')
	X_train = hf.create_dataset('X_train', (0, hist_time_steps, 6), maxshape=(None, hist_time_steps, 6), dtype='float64', chunks=(chunk_size, hist_time_steps, 6))
	Y_train = hf.create_dataset('Y_train', (0, len(pred_time_steps)), maxshape=(None, len(pred_time_steps)), dtype='float64', chunks=(chunk_size, len(pred_time_steps)))

	X_val = hf.create_dataset('X_val', (0, hist_time_steps, 6), maxshape=(None, hist_time_steps, 6), dtype='float64', chunks=(chunk_size, hist_time_steps, 6))
	Y_val = hf.create_dataset('Y_val', (0, len(pred_time_steps)), maxshape=(None, len(pred_time_steps)), dtype='float64', chunks=(chunk_size, len(pred_time_steps)))

	X_test = hf.create_dataset('X_test', (0, hist_time_steps, 6), maxshape=(None, hist_time_steps, 6), dtype='float64', chunks=(chunk_size, hist_time_steps, 6))
	Y_test = hf.create_dataset('Y_test', (0, len(pred_time_steps)), maxshape=(None, len(pred_time_steps)), dtype='float64', chunks=(chunk_size, len(pred_time_steps)))

	batch_x_train = []
	batch_y_train = []

	batch_x_val = []
	batch_y_val = []

	batch_x_test = []
	batch_y_test = []

	split_pattern = [0]	# Train: 0, val: 1, test: 2
	
	n_periods = int(	(len(n_df) - hist_time_steps - np.amax(pred_time_steps) - 1) / stride	) + 1

	'''
	split_pattern = [0] * n_periods
	val_idx = int(n_periods * 0.8)
	split_pattern[val_idx:] = [1] * (len(split_pattern) - val_idx)

	test_idx = int(n_periods * 0.9)
	split_pattern[test_idx:] = [2] * (len(split_pattern) - test_idx)
	'''

	# DEBUG
	n_x_non_finite = 0
	n_y_non_finite = 0
	n_x_contain_zero = 0 
	n_y_contain_zero = 0
	n_elements = 0
	n_flat = 0

	period_idx = 0
	for t in range(len(n_df) - hist_time_steps - np.amax(pred_time_steps) - 1, 0, -stride): # Iterate through time sections of the full dataset
		time_period_df = n_df[t:t + hist_time_steps + np.amax(pred_time_steps) + 1]	# Get time period from full df (dataframe)
		period_split = split_pattern[period_idx%len(split_pattern)]
		
		if period_idx == 0:	# Always make latest time period validation data
			period_split = 1
		
		period_idx+=1

		per_n_x_non_finite = n_x_non_finite
		per_n_y_non_finite = n_y_non_finite
		per_n_x_contain_zero = n_x_contain_zero
		per_n_y_contain_zero = n_y_contain_zero
		per_n_elements = n_elements
		per_n_flat = n_flat

		for sym in symbols:
			sym_df = time_period_df[sym]

			x_p = sym_df[:hist_time_steps].values


			# Filter X data
			if not np.isfinite(x_p).all():	# If not all the values are finite, don't add element
				#print('X element conains non finite value.')
				n_x_non_finite+=1
				continue

			if np.count_nonzero(x_p==0) > hist_time_steps:
				#print('X element contained zeros.')
				n_x_contain_zero+=1
				continue

			if np.sum(x_p) / (6 * hist_time_steps) < 0.003:	# Get average value for X
				#print('X mean value is {} (abnormal), removing'.format(np.sum(x_p) / (6 * hist_time_steps)))
				continue


			y_p = np.empty(len(pred_time_steps))

			for pred_idx in range(len(y_p)):	# Get the labels from sym_df
				y_idx = hist_time_steps + pred_time_steps[pred_idx]		# index for certain label in time_period_df
				y_p_elem = sym_df[y_idx:y_idx + 1]['Close'].values

				if len(y_p_elem) == 0: 
					print('Empty label value: {}, y_idx: {}'.format(sym_df[y_idx:y_idx + 1], y_idx))
				else:
					y_p[pred_idx] = y_p_elem[0]				

			n_elements+=1

			# Filter Y data
			if not np.isfinite(y_p).all():	# If not all the values are finite, don't add element
				#print('Y element conains non finite value.')
				n_y_non_finite+=1
				continue

			if np.count_nonzero(y_p==0) > 0:
				#print('Y element contained zeros')
				n_y_contain_zero+=1
				continue

			if np.sum(y_p) / len(pred_time_steps) < 0.003 or np.sum(y_p) / len(pred_time_steps) > 0.99: # Get average value for y
				#print('Y mean value is {} (abnormal), removing'.format(np.sum(y_p) / len(pred_time_steps)))
				continue

			if np.mean(np.diff(x_p, axis=0)[-3:], axis=0)[1] == 0:
				#print('Data is flat at the end. Company might have shut down leaving lack of data.')
				n_flat+=1
				continue

			if period_split == 0:	# Add to traing set
				batch_x_train.append(x_p)
				batch_y_train.append(y_p)

				if len(batch_x_train) == chunk_size:
					X_train.resize(X_train.shape[0] + chunk_size, axis=0)
					Y_train.resize(Y_train.shape[0] + chunk_size, axis=0)

					X_train[-chunk_size:] = np.array(batch_x_train)
					Y_train[-chunk_size:] = np.array(batch_y_train)

					batch_x_train = []
					batch_y_train = []

					print(X_train.shape)


			if period_split == 1:	# Add to validation set
				batch_x_val.append(x_p)
				batch_y_val.append(y_p)

				if len(batch_x_val) == chunk_size:
					X_val.resize(X_val.shape[0] + chunk_size, axis=0)
					Y_val.resize(Y_val.shape[0] + chunk_size, axis=0)

					X_val[-chunk_size:] = np.array(batch_x_val)
					Y_val[-chunk_size:] = np.array(batch_y_val)

					batch_x_val = []
					batch_y_val = []

					print(X_val.shape)

			if period_split == 2:	# Add to validation set
				batch_x_test.append(x_p)
				batch_y_test.append(y_p)

				if len(batch_x_test) == chunk_size:
					X_test.resize(X_test.shape[0] + chunk_size, axis=0)
					Y_test.resize(Y_test.shape[0] + chunk_size, axis=0)

					X_test[-chunk_size:] = np.array(batch_x_test)
					Y_test[-chunk_size:] = np.array(batch_y_test)

					batch_x_test = []
					batch_y_test = []

					print(X_test.shape)

		print('Running average. Non finite X, Y: {}, {}. Elements that contained too many zeros: {}, {}. Flat {}'.format(
			(n_x_non_finite - per_n_x_non_finite) / (n_elements - per_n_elements), (n_y_non_finite - per_n_y_non_finite) / (n_elements - per_n_elements), 
			(n_x_contain_zero - per_n_x_contain_zero) / (n_elements - per_n_elements), (n_y_contain_zero - per_n_y_contain_zero) / (n_elements - per_n_elements),
			(n_flat - per_n_flat) / (n_elements - per_n_elements) ))
		print('{:0.4f}% of periods completed. \n\nETA: {:0.2f} minutes. Elapsed: {}. Iterated: {}\n'.format(period_idx/n_periods * 100, (time.time()-start_time) / 60 / (period_idx/n_periods), (time.time()-start_time) / 60, n_elements))

	print('Filtered. Non finite X, Y: {}, {}. Elements that contained too many zeros: {}, {}. Flat {}'.format(n_x_non_finite / n_elements, n_y_non_finite / n_elements, n_x_contain_zero / n_elements, n_y_contain_zero / n_elements, n_flat / n_elements))
	

	if len(batch_x_train) > 0:	# if there are elements not added to dataset, add them
		X_train.resize(X_train.shape[0] + len(batch_x_train), axis=0)
		Y_train.resize(Y_train.shape[0] + len(batch_y_train), axis=0)
		
		X_train[-len(batch_x_train):] = np.array(batch_x_train)
		Y_train[-len(batch_y_train):] = np.array(batch_y_train)

	if len(batch_x_val) > 0:	# if there are elements not added to dataset, add them
		X_val.resize(X_val.shape[0] + len(batch_x_val), axis=0)
		Y_val.resize(Y_val.shape[0] + len(batch_y_val), axis=0)
		
		X_val[-len(batch_x_val):] = np.array(batch_x_val)
		Y_val[-len(batch_y_val):] = np.array(batch_y_val)

	if len(batch_x_test) > 0:	# if there are elements not added to dataset, add them
		X_test.resize(X_test.shape[0] + len(batch_x_test), axis=0)
		Y_test.resize(Y_test.shape[0] + len(batch_y_test), axis=0)
		
		X_test[-len(batch_x_test):] = np.array(batch_x_test)
		Y_test[-len(batch_y_test):] = np.array(batch_y_test)

	# Shuffle when generating batches for training instead

	hf.close()

	print('Done. Process took {:.2f} minutes and {} data elements were added.'.format((time.time() - start_time) / 60, len(X_train) + len(X_val) + len(X_test)))