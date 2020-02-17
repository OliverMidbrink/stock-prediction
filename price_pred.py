import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, h5py, random, data_tools, pickle, os, platform

if platform.system() == 'Windows':
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	set_session(sess)

	# --- Get Symbols ---
'''
import requests
from bs4 import BeautifulSoup
symbols_list = set([])
symbols_text = ""
for n_page in range(40):
	print('Page {}'.format(n_page))
	# URL FOR MID-LARGE CAP INT. url = 'https://finance.yahoo.com/screener/unsaved/2d100c4f-9eab-403f-ab91-8de0b917e380?offset={}&count=100'.format(n_page * 100)
	url = 'https://finance.yahoo.com/screener/unsaved/758d0d96-0a87-4cf9-ada9-9f8d77f87ae5?count=100&offset={}'.format(n_page*100)
	page = requests.get(url)
	soup = BeautifulSoup(page.content, 'html.parser')

	sym_tags = soup.find_all('a', {'class':'Fw(600)'})

	for sym in sym_tags:
		symbols_list.add(sym.text)

for sym in symbols_list:
	if len(sym)<1:
		print('Empty sym text')
		continue
	symbols_text += sym + " "


	# --- Download and Save ---
df = yf.download(symbols_text, start="2000-01-01", end="2020-02-10")
df.to_hdf(os.path.join('original_dfs', 'top10000-part1.h5'), 'df', mode='w', format='fixed')

sys.exit(0)
'''


	# --- Create HDF5 Dataset ---
def create_hdf5(output_filename, raw_dataset, hist_time_steps=30, pred_time_steps=7):
	df = pd.read_hdf(raw_dataset, 'df')
	df.columns = df.columns.swaplevel(0, 1)

	# Data Variables - key order for df is SYMBOL, VARIABLE, DATE 
	variables = df.columns.get_level_values(1).unique()
	symbols = df.columns.get_level_values(0).unique()


	n_df = df.copy()
	n_df = n_df.fillna(method='ffill')
	n_df = n_df.fillna(0)


	for sym in symbols:	#iterate through stocks and normalize data
		v_df = n_df[sym]
		nv_df=(v_df-v_df.min())/(v_df.max()-v_df.min())
		n_df[sym] = nv_df

	max_length = int(len(n_df) / (hist_time_steps + pred_time_steps) * len(symbols) * 1.02)

	X = np.zeros((max_length, hist_time_steps, 6))
	Y = np.zeros((max_length, pred_time_steps))		# The reason for choosing 7 timesteps is that it
								# eliminates some of the "noise" in the stock data

	print('Creating dataset... Expecting max {} elements'.format(max_length))
	count = -1
	display = 1
	tot_elements=0

	sym_count = -1
	for sym in symbols:
		sym_count+=1
		display+=1
		if display%31 == 0:
			display=1
			print('{} percent completed {}.'.format(round(sym_count/len(symbols)*100, 2), sym))

		sym_df = n_df[sym]
		
		start = 0
		diff = np.diff(sym_df['Close'])
		#test = np.where(np.diff(np.where(sym_df['Close'] == 0)) > 3)[0][0] 	Could be useful for VOLVO (where there is much space and lonely entries in close history)
		
		try:
			if sym_df['Close'][0] == 0:
				start = np.where(diff > 0)[0][0]+1 #start when company is introduced to market
			#print(sym_df[max(start-5,0):start+5])
		except Exception as e:
			print('Error {}. Start is {}'.format(e, start))
			print(sym_df)


		end = len(sym_df)-1
		t_df = sym_df[start:end]
		tot_elements += int((end-start)/(hist_time_steps+pred_time_steps))

		for i in range(0, len(t_df), hist_time_steps+pred_time_steps):
			x_p = t_df[i:i+hist_time_steps].values	# get range from i to i + 37
			#y_p = np.zeros((7, 1))
			y_p = t_df[i+hist_time_steps:i+(hist_time_steps+pred_time_steps)]["Close"].values
			
			if i+hist_time_steps+pred_time_steps >= len(t_df):
				continue

			if np.sum(x_p) / (6 * hist_time_steps + 6 * pred_time_steps) < 0.01:	# Get average value for X
				print('X mean value is abnormal, removing')
				continue

			if np.sum(y_p) / pred_time_steps < 0.01 or np.sum(y_p) / pred_time_steps > 0.99: # Get average value for y
				print('Y mean value is abnormal, removing')
				continue

			if not np.isfinite(x_p).all():	# If not all the values are finite, don't add element
				print('X element conains non finite value')
				continue

			if not np.isfinite(y_p).all():	# If not all the values are finite, don't add element
				print('Y element conains non finite value')
				continue

			count+=1
			X[count] = x_p
			Y[count] = y_p

	idx = len(X)
	searching = True
	n_empty = 0
	n_full = 0
	
	while searching:
		idx += -1
		if np.count_nonzero(X[idx]) > 0 and np.count_nonzero(Y[idx]) > 0:
			print('End Found: {}. Total expected sets: {}.'.format(idx, tot_elements))
			X = X[:idx+1]
			Y = Y[:idx+1]
			searching = False


	X, Y = shuffle(X, Y)
	train_frac = 0.6
	val_frac = 0.2
	X_train = X[:int(len(X) * train_frac)]
	Y_train = Y[:int(len(Y) * train_frac)]
	X_val = X[int(len(X) * train_frac) : int(len(X) * (train_frac + val_frac))]
	Y_val = Y[int(len(Y) * train_frac) : int(len(Y) * (train_frac + val_frac))]
	X_test = X[int(len(X) * (train_frac + val_frac)):]
	Y_test = Y[int(len(Y) * (train_frac + val_frac)):]


	hf = h5py.File(output_filename, 'w')
	hf.create_dataset('X_train', data=X_train)
	hf.create_dataset('Y_train', data=Y_train)

	hf.create_dataset('X_val', data=X_val)
	hf.create_dataset('Y_val', data=Y_val)

	hf.create_dataset('X_test', data=X_test)
	hf.create_dataset('Y_test', data=Y_test)

	hf.close()

	print('Done.')


#create_hdf5(os.path.join('datasets', '90Day-part1-ffill.h5'), os.path.join('original_dfs','top10000-part1.h5'), hist_time_steps=90)
#df = pd.read_hdf('top10000-part1.h5', 'df')
#df.to_csv('top10000-part1.csv')
#sys.exit(0)

def load_data(filename):
	hf = h5py.File(filename, 'r')
	ret = (np.array(hf.get('X_train')), np.array(hf.get('Y_train')),
		np.array(hf.get('X_val')), np.array(hf.get('Y_val')),
		np.array(hf.get('X_test')),np.array(hf.get('Y_test')))

	return ret


# Available Datasets
# 'Top-700-20-year-Swe.h5'
# 'Top-700-20-year-Swe-120Day.h5'	# New
# 'Top-100-20-year.h5'
# '700Swe-20Year-30Day.h5'		# New
dataset = os.path.join('datasets', '90Day-part1-ffill.h5')
X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(dataset)
hist_time_steps = 90
pred_time_steps = 7

	# --- Model ---
model = keras.models.Sequential()
model.add(keras.layers.GRU(120, return_sequences=True, input_shape=(None,6)))	# None for any number of timesteps
model.add(keras.layers.GRU(120, return_sequences=True))	# None for any number of timesteps
model.add(keras.layers.GRU(60, return_sequences=False))
model.add(keras.layers.Dense(60))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(pred_time_steps))


filepath = os.path.join('checkpoints', 'weights-improvement-{epoch:02d}-{val_loss:.6f}.h5')
check = keras.callbacks.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor=0.5, verbose=1, patience=3, min_lr = 0.00001)

opt = keras.optimizers.Adam(lr=0.01)
model.compile(loss='mse', optimizer=opt)
model.summary()


def scalar_augment(X_elem, Y_elem, min_scalar=0.5, max_scalar=1.5):
	scalar = min_scalar + random.random() * (max_scalar - min_scalar)
	#noise = np.random.normal(0, 0.001, X_elem.shape)
	return X_elem * scalar, Y_elem * scalar

data_gen = data_tools.CustomSequence(X_train, Y_train, 128, scalar_augment)


#model = keras.models.load_model(os.path.join('models', 'new_data_modelMSE.h5'))

history = model.fit_generator(next(iter(data_gen)), steps_per_epoch=len(data_gen), 
	validation_data=(X_val, Y_val), epochs=40, callbacks=[reduce_lr, check])

model.save(os.path.join('models', 'dual_aug.h5'))

with open(os.path.join('trainHistoryDict', 'mse_history.txt'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

#sys.exit(0)


def visualize2(X_, Y_, n_plots=2):
	for n_plots in range(n_plots): # Show n_plots
		i = random.randint(0, len(X_) - 1)	#pick random selection of data
		
		pred = model.predict(np.array([X_[i]])) # Make prediction of X_
		y = np.array([None]*37)		# y values of graph

		for j in range(37):	# Add 7 more timesteps to fill graph length (37)
			if j<30:
				y[j] = X_[i][j][1]
			else:
				y[j] = Y_[i][j-30]
			
		plt.plot(range(37), y, label="True Prices") # Display input data and true/correct output
		plt.plot(range(37), np.append([None] * 30, pred[0]), label="Predicted") # Display predictions
		plt.axvline(x=29.5)
		plt.show()


def visualize3(X_, Y_, hist_time_steps=30, pred_time_steps=7, start=0):
	fig1, f1_axes = plt.subplots(ncols=7, nrows=7, sharex='col', tight_layout=True)
	idx_total = set(range(len(X_)-1))

	for row in range(len(f1_axes)):
		for col in range(len(f1_axes[0])):
			if len(idx_total) < 0:
				print('Data has run out. Total dataset is: {}.'.format(len(X_)))

			index = random.sample(idx_total, 1)[0]
			idx_total.remove(index)

			pred = model.predict(np.array([X_[index]]))
			y = np.array([None]*(hist_time_steps + pred_time_steps))		# y values of graph

			for t in range(hist_time_steps + pred_time_steps):	# Add 7 more timesteps to fill graph length (37)
				if t<hist_time_steps:
					y[t] = X_[index][t][1]
				else:
					y[t] = Y_[index][t-hist_time_steps]

			f1_axes[row][col].plot(range(hist_time_steps + pred_time_steps), y)
			f1_axes[row][col].plot(range(hist_time_steps + pred_time_steps), np.append([None] * hist_time_steps, pred[0]))
			f1_axes[row][col].axvline(x=(hist_time_steps-0.5))


	plt.show()

def evaluate(X_, Y_, n_):
	idx_total = set(range(len(X_)-1))

	tp=0
	tn=0
	fp=0
	fn=0

	for x in range(n_):
		i = random.sample(idx_total, 1)[0]
		idx_total.remove(i)

		previous_close = X_[i][-1][1]
		pred = model.predict(np.array([X_[i]]))[0]
		true = Y_[i]

		avg_pred = -previous_close	# How will the true and pred stock prices differ from the last close price in X
		avg_true = -previous_close
		for j in range(len(pred)):
			avg_true+=true[j]/len(true)
			avg_pred+=pred[j]/len(pred)

		if avg_pred >= 0:
			if avg_true >= 0:
				tp+=1
			elif avg_true < 0:
				fp+=1
		else:
			if avg_true >= 0:
				fn+=1
			elif avg_true < 0:
				tn+=1

	print('True Positive: {}, TN {}, FP {}, FN {}.'.format(tp, tn, fp, fn))
	return tp, tn, fp, fn




#visualize2(X_val, Y_val, 5)

visualize3(X_val, Y_val, hist_time_steps=hist_time_steps, pred_time_steps=pred_time_steps)

tp, tn, fp, fn = evaluate(X_val, Y_val, 1000)	# 53.4% Accuracy (TP + TN)/(TP + TN + FP + FN)
									# 49.8% of stock data increased in price
									# 50.1% of stock data decreased in price

Accuracy = (tp + tn)/(tp+tn+fp+fn)
Percent_up = (tp + fn)/(tp+tn+fp+fn)
Percent_down = (tn + fp)/(tp+tn+fp+fn)
print('Taking the mean of prediction values and comparing that to the mean of the label values. \n Accuracy, (TP + TN)/(TP + TN + FP + FN), for dataset {} was: {}, Percent of stocks that went up: {}, percent of stocks that went down {}. TP: {}, TN: {}, FP:{}, FN: {}.'.format(dataset, Accuracy, Percent_up, Percent_down, tp, tn, fp, fn))



# //TODO 
# 1. Update the evaluation function to take into account how much the stock changed. 
	# Also, make an array of all the errors and explor that. For instance the
	# standard deviation of each pred_time_step. This is useful to show the 
	# uncertainty in the data. 
# 2. 
# 3. Reinforcement learning?



# Future thoughts
# * Add S&P500 (Or other index) to the input data
# * Add news articles to the data somehow. Maybe add sentiment along the input






