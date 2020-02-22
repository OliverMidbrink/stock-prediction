import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, h5py, random, data_tools, pickle, os, platform, time
from data_tools import trim_zeros

if platform.system() == 'Windows':
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	set_session(sess)

	# --- Get Symbols ---
'''
import requests
from bs4 import BeautifulSoup
symbols_set = set([])
symbols_text = ""
for n_page in range(160):
	print('Page {}'.format(n_page))
	# URL FOR MID-LARGE CAP INT. url = 'https://finance.yahoo.com/screener/unsaved/2d100c4f-9eab-403f-ab91-8de0b917e380?offset={}&count=100'.format(n_page * 100)
	url = 'https://finance.yahoo.com/screener/unsaved/7e03632f-b9e5-469d-afb7-25d18aa8e444?count=100&offset={}'.format(n_page*100)
	page = requests.get(url)
	soup = BeautifulSoup(page.content, 'html.parser')

	sym_tags = soup.find_all('a', {'class':'Fw(600)'})

	for sym in sym_tags:
		symbols_set.add(sym.text)

for sym in symbols_set:
	if len(sym)<1:
		symbols_set.remove("")
	symbols_text += sym + " "

print('Number of symbols is: {}.'.format(len(symbols_set)))
with open(os.path.join('original_dfs', 'symbols.txt'), 'w') as f:
	f.write(symbols_text)

	# --- Download and Save ---
symbols_list = list(symbols_set)
start_date = "2000-01-01"
end_date = "2019-06-01"
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
full_df.to_hdf(os.path.join('original_dfs', 'to-2019-06-large.h5'), 'df', mode='w', format='fixed')

df = pd.read_hdf(os.path.join('original_dfs', 'to-2019-06-large.h5'), 'df')
#df.to_csv(os.path.join('original_dfs', 'to-2019-06-large.csv'))
sys.exit(0)
'''


	# --- Create HDF5 Dataset ---
def create_hdf5(output_filename, raw_dataset, hist_time_steps=30, pred_time_steps=7): #Create parallel dataset
	start_time = time.time()

	print('Reading data file...')
	df = pd.read_hdf(raw_dataset, 'df')
	df.columns = df.columns.swaplevel(0, 1)
	print('Done.')

	# Data Variables - key order for df is SYMBOL, VARIABLE, DATE 
	variables = df.columns.get_level_values(1).unique()
	symbols = df.columns.get_level_values(0).unique()

	# Normalize
	print('Normalizing data...')
	n_df = df.copy()
	n_df = n_df.fillna(method='ffill')
	n_df = n_df.fillna(0)
	for sym in symbols:	#iterate through stocks and normalize data for that stock
		v_df = n_df[sym]
		nv_df=(v_df-v_df.min())/(v_df.max()-v_df.min())
		n_df[sym] = nv_df
	print('Done.')


	max_length = int(len(n_df) / (hist_time_steps + pred_time_steps) * len(symbols) * 1.02)	# Total amount of data periods 

	X_train = np.zeros((max_length, hist_time_steps, 6))
	X_val = np.zeros((max_length, hist_time_steps, 6))
	X_test = np.zeros((max_length, hist_time_steps, 6))

	Y_train = np.zeros((max_length, pred_time_steps))	
	Y_val = np.zeros((max_length, pred_time_steps))
	Y_test = np.zeros((max_length, pred_time_steps))

	split_pattern = [0, 1, 0, 2, 0]	# Train: 0, val: 1, test: 2
	
	n_periods = int((len(n_df) - (hist_time_steps + pred_time_steps)) / (hist_time_steps + pred_time_steps)) + 1


	split_pattern = [0] * n_periods
	val_idx = int(n_periods * 0.8)
	split_pattern[val_idx:] = [1] * (len(split_pattern) - val_idx)

	test_idx = int(n_periods * 0.9)
	split_pattern[test_idx:] = [2] * (len(split_pattern) - test_idx)

	# DEBUG
	n_x_non_finite = 0
	n_y_non_finite = 0
	n_x_contain_zero = 0 
	n_y_contain_zero = 0
	n_elements = 0

	train_next_idx = 0
	val_next_idx = 0
	test_next_idx = 0

	period_idx = 0
	for t in range(len(n_df) - (hist_time_steps + pred_time_steps), 0, -(hist_time_steps + pred_time_steps)): # Iterate through time sections of the full dataset
		time_period_df = n_df[t:t + (hist_time_steps + pred_time_steps)]	# Get time period from full df (dataframe)
		period_split = split_pattern[period_idx%(len(split_pattern))]
		period_idx+=1

		per_n_x_non_finite = n_x_non_finite
		per_n_y_non_finite = n_y_non_finite
		per_n_x_contain_zero = n_x_contain_zero
		per_n_y_contain_zero = n_y_contain_zero
		per_n_elements = n_elements

		for sym in symbols:
			sym_df = time_period_df[sym]
			x_p = sym_df[:hist_time_steps].values
			y_p = sym_df[hist_time_steps:]["Close"].values
			n_elements+=1

			# Filter data
			if not np.isfinite(x_p).all():	# If not all the values are finite, don't add element
				#print('X element conains non finite value.')
				n_x_non_finite+=1
				continue

			if not np.isfinite(y_p).all():	# If not all the values are finite, don't add element
				#print('Y element conains non finite value.')
				n_y_non_finite+=1
				continue

			if np.count_nonzero(x_p==0) > hist_time_steps:
				#print('X element contained zeros.')
				n_x_contain_zero+=1
				continue

			if np.count_nonzero(y_p==0) > 0:
				#print('Y element contained zeros')
				n_y_contain_zero+=1
				continue

			if np.sum(x_p) / (6 * hist_time_steps + 6 * pred_time_steps) < 0.005:	# Get average value for X
				print('X mean value is abnormal, removing')
				continue

			if np.sum(y_p) / pred_time_steps < 0.005 or np.sum(y_p) / pred_time_steps > 0.99: # Get average value for y
				print('Y mean value is abnormal, removing')
				continue

			if period_split == 0:	# Add to traing set
				X_train[train_next_idx] = x_p
				Y_train[train_next_idx] = y_p
				train_next_idx+=1

			if period_split == 1:	# Add to validation set
				X_val[val_next_idx] = x_p
				Y_val[val_next_idx] = y_p
				val_next_idx+=1

			if period_split == 2:	# Add to validation set
				X_test[test_next_idx] = x_p
				Y_test[test_next_idx] = y_p
				test_next_idx+=1

		print('Running average. Non finite X, Y: {}, {}. Elements that contained too many zeros: {}, {}'.format(
			(n_x_non_finite - per_n_x_non_finite) / (n_elements - per_n_elements), (n_y_non_finite - per_n_y_non_finite) / (n_elements - per_n_elements), 
			(n_x_contain_zero - per_n_x_contain_zero) / (n_elements - per_n_elements), (n_y_contain_zero - per_n_y_contain_zero) / (n_elements - per_n_elements)))
		print('{:0.4f}% of periods completed'.format(period_idx/n_periods * 100))

	print('Filtered. Non finite X, Y: {}, {}. Elements that contained too many zeros: {}, {}'.format(n_x_non_finite / n_elements, n_y_non_finite / n_elements, n_x_contain_zero / n_elements, n_y_contain_zero / n_elements))
	
	X_train, Y_train = trim_zeros(X_train, Y_train)
	X_val, Y_val = trim_zeros(X_val, Y_val)
	X_test, Y_test = trim_zeros(X_test, Y_test)

	X_train, Y_train = shuffle(X_train, Y_train)
	X_val, Y_val = shuffle(X_val, Y_val)
	X_test, Y_test = shuffle(X_test, Y_test)

	hf = h5py.File(output_filename, 'w')
	hf.create_dataset('X_train', data=X_train)
	hf.create_dataset('Y_train', data=Y_train)

	hf.create_dataset('X_val', data=X_val)
	hf.create_dataset('Y_val', data=Y_val)

	hf.create_dataset('X_test', data=X_test)
	hf.create_dataset('Y_test', data=Y_test)

	hf.close()

	print('Done. Process took {:.2f} minutes and {} data elements were added.'.format((time.time() - start_time) / 60, len(X_train) + len(X_val) + len(X_test)))


#create_hdf5(os.path.join('datasets', '90Day-to-2019-06-max_zero_hist_t_steps.h5'), os.path.join('original_dfs', 'to-2019-06-large.h5'), hist_time_steps=90)
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
# '90Day-part1-parallel.h5'
# Obsolete datasets (Train, Val, and Test data was overlapping, causing exagurated results too good to be true)
# 'Top-700-20-year-Swe.h5'
# 'Top-700-20-year-Swe-120Day.h5'
# 'Top-100-20-year.h5'
# '700Swe-20Year-30Day.h5'		
dataset = os.path.join('datasets', '90Day-part1-parallel.h5')
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


def scalar_augment(X_elem, Y_elem, min_scalar=1, max_scalar=1):
	scalar = min_scalar + random.random() * (max_scalar - min_scalar)
	#noise = np.random.normal(0, 0.001, X_elem.shape)
	return X_elem * scalar, Y_elem * scalar

data_gen = data_tools.CustomSequence(X_train, Y_train, 128, scalar_augment)


model = keras.models.load_model(os.path.join('checkpoints', 'weights-improvement-18-0.000434.h5'))

#history = model.fit_generator(next(iter(data_gen)), steps_per_epoch=len(data_gen), 
#	validation_data=(X_val, Y_val), epochs=20, callbacks=[reduce_lr, check])

#model.save(os.path.join('models', 'parallel.h5'))

#with open(os.path.join('trainHistoryDict', 'mse_history.txt'), 'wb') as file_pi:
#        pickle.dump(history.history, file_pi)

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

	n_buy_correct = 0
	n_buy_predictions = 0
	n_buy_incorrect = 0
	accum_change_of_correct_buy = 0	# Divide by number 
	accum_change_of_incorrect_buy = 0 # Divide by number of elements
	accum_purchase = 0 # Accum, previous close
	accum_stock_trans_cost = 0 # Total transaction costs
	accum_dev_mean = 0	# Mean of true value (label) accumulated, 7 days in some cases
	mean_all_stock_ROI = 0 # How did the "market" change during this period

	disp_count=0
	for x in range(n_):
		disp_count+=1
		if disp_count % 1000 == 0:
			print('{:.3f} percent evaluated.'.format(x/n_*100))
		i = random.sample(idx_total, 1)[0]
		idx_total.remove(i)

		previous_close = X_[i][-1][1]
		pred = model.predict(np.array([X_[i]]))[0]
		true = Y_[i]

		avg_value_pred = 0	# How will the true and pred stock prices differ from the last close price in X
		avg_value_true = 0
		for j in range(len(pred)):
			avg_value_true+=true[j]/len(true)
			avg_value_pred+=pred[j]/len(pred)

		if avg_value_pred - previous_close >= 0:
			if avg_value_true - previous_close >= 0:
				tp+=1
			elif avg_value_true - previous_close < 0:
				fp+=1
		else:
			if avg_value_true - previous_close >= 0:
				fn+=1
			elif avg_value_true - previous_close < 0:
				tn+=1

		mean_all_stock_ROI += (avg_value_true - previous_close) / previous_close / n_
		
		if pred[np.argmax(pred)] / previous_close > 1.03:	# Decides to purchase stock
			n_buy_predictions += 1
			accum_purchase += previous_close	# Buy stock at opening the next day, which would be similar to close at prediction day.
			accum_stock_trans_cost += previous_close * 0.01	# Counting on 1% transaction fee
			accum_dev_mean += true[np.argmax(pred)]	# Sell at highest predicted day, therefore development is the pred_high day - previous close
			# Could sell 
			if avg_value_true / previous_close > 1.01:	# Stock actually went up 1%, therfore it could be considered a success
				n_buy_correct += 1
				accum_change_of_correct_buy += avg_value_true/previous_close
			else:	# All bought stocks that did not go up 1%
				n_buy_incorrect += 1
				accum_change_of_incorrect_buy += avg_value_true/previous_close


	buy_accuracy = n_buy_correct/n_buy_predictions
	mean_change_correct_buy = accum_change_of_correct_buy / n_buy_correct
	mean_change_incorrect_buy = accum_change_of_incorrect_buy / n_buy_incorrect

	ROI = (accum_dev_mean - accum_purchase - accum_stock_trans_cost) / (accum_purchase + accum_stock_trans_cost)
	msg = 'Market change: {}. Buy Accuracy {}. n_buy {}, n_buy_correct {}, Mean_change_of_stocks_up_1% {}, Mean_not_up_1% {}.\nROI: {}. Result true mean of (pred_days): {}, purchase: {}, trans_cost {}.'.format(mean_all_stock_ROI, buy_accuracy, n_buy_predictions, n_buy_correct, mean_change_correct_buy, mean_change_incorrect_buy, ROI, accum_dev_mean, accum_purchase, accum_stock_trans_cost)
	return (tp, tn, fp, fn), msg




#visualize2(X_val, Y_val, 5)

#visualize3(X_test, Y_test, hist_time_steps=hist_time_steps, pred_time_steps=pred_time_steps)


(tp, tn, fp, fn), msg = evaluate(X_test, Y_test, len(X_test)-1)	# 53.4% Accuracy (TP + TN)/(TP + TN + FP + FN)
									# 49.8% of stock data increased in price
									# 50.1% of stock data decreased in price

Accuracy = (tp + tn)/(tp+tn+fp+fn)
Percent_up = (tp + fn)/(tp+tn+fp+fn)
Percent_down = (tn + fp)/(tp+tn+fp+fn)

print('Evaluation after creating and training on the parallel dataset. Best out of 20 epochs.') # Put description here before evaluating
print(msg)
print('Taking the mean of prediction values and comparing that to the mean of the label values. \n Accuracy, (TP + TN)/(TP + TN + FP + FN), for dataset {} was: {}, Percent of stocks that went up: {}, percent of stocks that went down {}. TP: {}, TN: {}, FP:{}, FN: {}.'.format(dataset, Accuracy, Percent_up, Percent_down, tp, tn, fp, fn))



# //TODO 
# 1.  Change buy and sell method, try selling at the highest predicted day, buying if that day is higher than threshold.
	# Maybe make a probability distribution similar to the pred array. 
	# Make an array of all the change and explore that. For instance the
	# standard deviation of each pred_time_step. This is useful to show the 
	# uncertainty in the data. 
# 2. 
# 3. Reinforcement learning?

# // Notes!
# Since many of the stocks prices are interdependant (if one company performs well that usually benefits similar large companies)
# It might be so that the model learns the shape/pattern of one TRAINING stock before it goes up, for instance after 2008,
# and because they might be interdependant, it might see the same pattern in a VALIDATION stock. So it regonizes this pattern and predicts up > 6%
# and so all of the successful stocks might just be similar patterns from the training data, even the test data might not be "safe". 
# One possible solution could be to divide up the data, not randomly, but instead during different sections of time. For instance training from 2000-2018, val 2019 and test 2020. 


# Future thoughts
# * Add S&P500 (Or other index) to the input data
# * Add news articles to the data somehow. Maybe add sentiment along the input






