import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, h5py, random, data_tools, pickle, os, platform, time, data_tools
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
for n_page in range(3):
	print('Page {}'.format(n_page))
	url = 'https://finance.yahoo.com/screener/unsaved/b643ee85-beab-4e9f-acf7-6c55c6f168ae?count=100&offset={}'.format(n_page*100)
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
start_date = "2019-06-02"
end_date = "2020-02-25"

full_df = data_tools.download_symbols(list(symbols_set), start_date=start_date, end_date=end_date)
full_df.to_hdf(os.path.join('original_dfs', 'from-2019-06-to-2020-02-25-swe290ByVOLUME.h5'), 'df', mode='w', format='fixed')

#df = pd.read_hdf(os.path.join('original_dfs', 'from-2019-06-swe.h5'), 'df')
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


	max_length = int(len(n_df) / (hist_time_steps + pred_time_steps) * len(symbols) * 1.1)	# Total amount of data periods 

	X_train = np.zeros((max_length, hist_time_steps, 6))
	X_val = np.zeros((max_length, hist_time_steps, 6))
	X_test = np.zeros((max_length, hist_time_steps, 6))

	Y_train = np.zeros((max_length, pred_time_steps))	
	Y_val = np.zeros((max_length, pred_time_steps))
	Y_test = np.zeros((max_length, pred_time_steps))

	split_pattern = [2]	# Train: 0, val: 1, test: 2
	
	n_periods = int((len(n_df) - (hist_time_steps + pred_time_steps)) / (hist_time_steps + pred_time_steps)) + 1

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

	train_next_idx = 0
	val_next_idx = 0
	test_next_idx = 0

	period_idx = 0
	for t in range(len(n_df) - (hist_time_steps + pred_time_steps), 0, -(hist_time_steps + pred_time_steps)): # Iterate through time sections of the full dataset
		time_period_df = n_df[t:t + (hist_time_steps + pred_time_steps)]	# Get time period from full df (dataframe)
		period_split = split_pattern[period_idx%(len(split_pattern))]
		print('Period split: {}'.format(split_pattern[period_idx%len(split_pattern)]))
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


#create_hdf5(os.path.join('datasets', '80Day-250Stocks-FROM-2019-06.h5'), os.path.join('original_dfs', 'from-2019-06-to-2020-02-25-swe290ByVOLUME.h5'), hist_time_steps=80)
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
# See datasets folder for more (only in local version)
# Obsolete datasets (Train, Val, and Test data was overlapping, causing exagurated results too good to be true)
# 'Top-700-20-year-Swe.h5'
# 'Top-700-20-year-Swe-120Day.h5'
# 'Top-100-20-year.h5'
# '700Swe-20Year-30Day.h5'		
dataset = os.path.join('datasets', '80Day-250Stocks-FROM-2019-06.h5')
X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(dataset)
hist_time_steps = 80
pred_time_steps = 7

	# --- Model ---
model = keras.models.Sequential()
model.add(keras.layers.GRU(120, return_sequences=True, input_shape=(None,6), reset_after = True, recurrent_activation='sigmoid'))	# None for any number of timesteps
model.add(keras.layers.GRU(120, return_sequences=True, reset_after = True, recurrent_activation='sigmoid'))	# None for any number of timesteps
model.add(keras.layers.GRU(70, return_sequences=False, reset_after = True, recurrent_activation='sigmoid'))
model.add(keras.layers.Dense(70))
model.add(keras.layers.Dropout(0.15))
model.add(keras.layers.Dense(pred_time_steps))


filepath = os.path.join('checkpoints', 'to-2019-06-checkpoints', 'weights-improvement-{epoch:02d}-{val_loss:.6f}.h5')
check = keras.callbacks.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor=0.5, verbose=1, patience=3, min_lr = 0.00001)

opt = keras.optimizers.Adam(lr=0.01)
model.compile(loss='mse', optimizer=opt)
model.summary()


def scalar_augment(X_elem, Y_elem, min_scalar=1, max_scalar=1):
	#scalar = min_scalar + random.random() * (max_scalar - min_scalar)
	#noise = np.random.normal(0, 0.001, X_elem.shape)
	return X_elem, Y_elem

data_gen = data_tools.CustomSequence(X_train, Y_train, 128, scalar_augment)

model_file_name = os.path.join('checkpoints', 'to-2019-06-checkpoints', 'weights-improvement-20-0.000289.h5')
#model_std_error = np.array([0.39474307, 0.36525669, 0.39066132, 0.38776542, 0.3619745, 0.35675924, 0.34576769])	# Inaccuracy based on train data
model_std_error = np.array([0.60684879, 0.57097587, 0.57766695, 0.50046783, 0.50637192, 0.59326133, 0.52038012]) # Inaccuracy based on val data
model = keras.models.load_model(model_file_name)

#history = model.fit_generator(next(iter(data_gen)), steps_per_epoch=len(data_gen), 
#	validation_data=(X_val, Y_val), epochs=20, callbacks=[reduce_lr, check])

#model.save(os.path.join('models', 'parallel.h5'))

#with open(os.path.join('trainHistoryDict', 'to-2019-06_history.txt'), 'wb') as file_pi:
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
		
		if pred[np.argmax(pred)] / previous_close > 1.07:	# Decides to purchase stock
			n_buy_predictions += 1
			accum_purchase += previous_close	# Buy stock at opening the next day, which would be similar to close at prediction day.
			accum_stock_trans_cost += previous_close * 0.01	# Counting on 1% transaction fee
			sell_price = true[-1] # Sell at last random day
			accum_dev_mean += sell_price

			if sell_price / previous_close > 1.01:	# Stock actually went up 1%, therfore it could be considered a success
				n_buy_correct += 1
				accum_change_of_correct_buy += sell_price/previous_close
			else:	# All bought stocks that did not go up 1%
				n_buy_incorrect += 1
				accum_change_of_incorrect_buy += sell_price/previous_close


	buy_accuracy = n_buy_correct/n_buy_predictions
	mean_change_correct_buy = accum_change_of_correct_buy / n_buy_correct
	mean_change_incorrect_buy = accum_change_of_incorrect_buy / n_buy_incorrect

	ROI = (accum_dev_mean - accum_purchase - accum_stock_trans_cost) / (accum_purchase + accum_stock_trans_cost)
	msg = 'Market change: {}. Buy Accuracy {}. n_buy {}, n_buy_correct {}, Mean_change_of_stocks_up_1% {}, Mean_not_up_1% {}.\nROI: {}. Result true mean of (pred_days): {}, purchase: {}, trans_cost {}.'.format(mean_all_stock_ROI, buy_accuracy, n_buy_predictions, n_buy_correct, mean_change_correct_buy, mean_change_incorrect_buy, ROI, accum_dev_mean, accum_purchase, accum_stock_trans_cost)
	return (tp, tn, fp, fn), msg

def visualize4(X_, Y_, hist_time_steps=30, pred_time_steps=7, start=0):
	fig1, f1_axes = plt.subplots(ncols=7, nrows=7, sharex='col', tight_layout=True)
	idx_total = set(range(len(X_)-1))

	for row in range(len(f1_axes)):
		for col in range(len(f1_axes[0])):
			if len(idx_total) < 0:
				print('Data has run out. Total dataset is: {}.'.format(len(X_)))

			index = random.sample(idx_total, 1)[0]
			idx_total.remove(index)

			previous_close = X_[index][-1][1]
			pred = model.predict(np.array([X_[index]]))
			y = np.array([None]*(hist_time_steps + pred_time_steps))		# y values of graph

			for t in range(hist_time_steps + pred_time_steps):	# Add 7 more timesteps to fill graph length (37)
				if t<hist_time_steps:
					y[t] = X_[index][t][1]
				else:
					y[t] = Y_[index][t-hist_time_steps]

			f1_axes[row][col].plot(range(hist_time_steps + pred_time_steps), y)
			f1_axes[row][col].plot(range(hist_time_steps + pred_time_steps), np.append([None] * hist_time_steps, pred[0]))
			f1_axes[row][col].plot(range(hist_time_steps + pred_time_steps), np.append([None] * hist_time_steps, pred[0] + model_std_error * previous_close))
			f1_axes[row][col].plot(range(hist_time_steps + pred_time_steps), np.append([None] * hist_time_steps, pred[0] - model_std_error * previous_close))
			f1_axes[row][col].axvline(x=(hist_time_steps-0.5))


	plt.show()

def evaluate2(X_, Y_, n_):
	print('Evaluating {} random samples.'.format(n_))
	idx_total = set(range(len(X_)-1))

	n_buy_correct = 0
	n_buy_predictions = 0
	n_buy_incorrect = 0
	accum_change_of_correct_buy = 0	# Divide by number 
	accum_change_of_incorrect_buy = 0 # Divide by number of elements
	accum_purchase = 0 # Accum, previous close
	accum_stock_trans_cost = 0 # Total transaction costs
	accum_dev_mean = 0	# Mean of true value (label) accumulated, 7 days in some cases
	mean_all_stock_ROI = 0 # How did the "market" change during this period

	error_arr = np.zeros((n_, len(Y_[0])))
	roi_arr = np.zeros((n_, 1 + 5))	# Columns: ROI buy and sell last day (all stocks), ROI strategy 1, ROI strategy 2 ...
	change_arr = np.zeros((n_, 1 + 5))

	disp_count=0
	for x in range(n_):
		disp_count+=1
		if disp_count % 500 == 0:
			print('{:.1f} percent evaluated.'.format(x/n_*100))
		i = random.sample(idx_total, 1)[0]
		idx_total.remove(i)

		previous_close = X_[i][-1][1]
		if previous_close == 0:
			print('Previous close is 0!')
			continue
		pred = model.predict(np.array([X_[i]]))[0]
		true = Y_[i]

		mean_value_pred = 0	# How will the true and pred stock prices differ from the last close price in X
		mean_value_true = 0
		for j in range(len(pred)):
			mean_value_true+=true[j]/len(true)
			mean_value_pred+=pred[j]/len(pred)

			error_arr[x][j] = (pred[j] - true[j]) / true[j]	# Error in relation to previous close

		roi_arr[x][0] = (true[-1] - previous_close * 1.01) / (previous_close * 1.01)	# How did the stock change over the pred period (taking trans fee into account)
		change_arr[x][0] = (true[-1] - previous_close) / previous_close

		if (true[-1] - previous_close) / previous_close > 0.3 or (true[-1] - previous_close) / previous_close < -0.3:	# Showing only stocks with less than 30% change
			print('Strong change: {}, pred: {}.'.format((true[-1] - previous_close) / previous_close, (pred[-1] - previous_close) / previous_close))
		else:
			# Change less than 30 percent up/down
			roi_arr[x][4] = (true[-1] - previous_close * 1.01) / (previous_close * 1.01)
			change_arr[x][4] = (true[-1] - previous_close) / previous_close

			if pred[4] / previous_close > 1.02:	# Performance for this strategy has been good consistently
				roi_arr[x][5] = (true[-1] - previous_close * 1.01) / (previous_close * 1.01)
				change_arr[x][5] = (true[-1] - previous_close) / previous_close

		# --- PURCHASING STRATEGIES ---
		max_accuracy_day = np.argmin(model_std_error)
		
		if pred[4] / previous_close > 1.02:	# Performance for this strategy has been good consistently
			roi_arr[x][1] = (true[-1] - previous_close * 1.01) / (previous_close * 1.01)
			change_arr[x][1] = (true[-1] - previous_close) / previous_close

		if np.mean(pred[1:-2]) / previous_close > 1.06:	# This strategy performed best last val trail
			roi_arr[x][2] = (true[-1] - previous_close * 1.01) / (previous_close * 1.01)
			change_arr[x][2] = (true[-1] - previous_close) / previous_close

		if pred[4] / previous_close > 1.04:	# This strategy performed best last val trail
			roi_arr[x][3] = (true[-1] - previous_close * 1.01) / (previous_close * 1.01)
			change_arr[x][3] = (true[-1] - previous_close) / previous_close


	mean_roi_arr = np.true_divide(roi_arr.sum(0), (roi_arr!=0).sum(0))
	mean_change_arr = np.true_divide(change_arr.sum(0), (change_arr!=0).sum(0))
	print(change_arr)
	print('  Up   ', np.true_divide((change_arr>0).sum(0), (change_arr!=0).sum(0)))
	print('  Down ', np.true_divide((change_arr<0).sum(0), (change_arr!=0).sum(0)))
	print('Number ', (change_arr!=0).sum(0))
	print('Change ', mean_change_arr)
	print('   ROI ', mean_roi_arr)

	std_error_arr = np.std(error_arr, axis=0)
	mean_error_arr = np.mean(error_arr, axis=0)
	return std_error_arr, mean_error_arr, mean_roi_arr, roi_arr, mean_change_arr




#visualize2(X_val, Y_val, 5)

#visualize4(X_test, Y_test, hist_time_steps=hist_time_steps, pred_time_steps=pred_time_steps)


std_arr, mean_arr, mean_roi_arr, roi_arr, mean_change_arr = evaluate2(X_test, Y_test, len(X_test)-1)	# 53.4% Accuracy (TP + TN)/(TP + TN + FP + FN)
									# 49.8% of stock data increased in price
									# 50.1% of stock data decreased in price

'''
print(std_arr)
print(mean_arr)
print('Mean ROI array: {}'.format(mean_roi_arr))
print('Sum change array: {}'.format(mean_change_arr))
n_per_method = (roi_arr!=0).sum(0)
n_per_method[0] = 0
'''

'''
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(std_arr)
ax.plot(mean_arr)
ax.plot(mean_roi_arr)
ax.plot(mean_change_arr)
ax2 = ax.twinx()
ax2.plot(n_per_method, color='tab:purple')
ax.set_xlabel('Prediction day index and Method index')
ax.set_ylabel('STD (Blue), Mean Error (Orange) and Mean ROI per Method (Green)', fontsize=7)
ax2.set_ylabel('Number of purchases for each method', color='tab:purple')
ax2.tick_params(axis='y', labelcolor='tab:purple')
plt.show()
'''

# //TODO 
# 1.  Try creating a rolling train dataset.
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






