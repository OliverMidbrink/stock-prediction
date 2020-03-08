import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, h5py, random, data_tools, pickle, os, platform, time, data_tools, re
from datetime import datetime, timedelta
from data_tools import trim_zeros
from keras.engine import topology


	# --- Requirements for data ---		###This file gets the latest X market days and makes predictions for the symbols loaded around line 38 ### 
hist_time_steps = 500

#model_file_name = os.path.join('checkpoints', 'to-2019-06-checkpoints', 'weights-improvement-20-0.000289.h5') #30 day model. Change above variable hist_time_steps
model_file_name = os.path.join('checkpoints', 'to-2019-06-sliding-checkpoints', 'weights-improvement-12-0.003575.h5')
model_name = re.sub('\W', '', model_file_name)
#model_file_name = os.path.join('checkpoints', 'weights-improvement-18-0.000434.h5')
#model_std_error = np.array([0.60684879, 0.57097587, 0.57766695, 0.50046783, 0.50637192, 0.59326133, 0.52038012]) # Inaccuracy based on val data
model_loaded = keras.models.load_model(model_file_name)
model_loaded.save_weights(model_file_name[:-3]+'-weights_only.h5')

pred_time_steps = [1, 3, 5, 10, 20, 65]

cpu_compatible_model = keras.models.Sequential()
cpu_compatible_model.add(keras.layers.GRU(120, return_sequences=True, input_shape=(None,6), reset_after = True, recurrent_activation='sigmoid'))	# None for any number of timesteps
cpu_compatible_model.add(keras.layers.GRU(120, return_sequences=True, reset_after = True, recurrent_activation='sigmoid'))	# None for any number of timesteps
cpu_compatible_model.add(keras.layers.GRU(70, return_sequences=False, reset_after = True, recurrent_activation='sigmoid'))
cpu_compatible_model.add(keras.layers.Dense(70))
cpu_compatible_model.add(keras.layers.Dropout(0.15))
cpu_compatible_model.add(keras.layers.Dense(len(pred_time_steps)))
print(cpu_compatible_model.predict(np.zeros((1,500,6))))
cpu_compatible_model.load_weights(os.path.join('checkpoints', 'to-2019-06-sliding-checkpoints', 'weights-improvement-12-0.003575-weights_only.h5'))
print(cpu_compatible_model.predict(np.zeros((1,500,6))))
print(cpu_compatible_model.predict(np.zeros((1,500,6))))
sys.exit(0)


today = datetime.date(datetime.now())
start_date = '2000-01-01'

latest_df_file_name = os.path.join('original_dfs', 'latest', 'latest-{}-steps{}.h5'.format(today, hist_time_steps))


	# --- Download/load data ---
latest_df = None
should_replace = False
if datetime.now().timetuple()[6] < 5 and os.path.isfile(latest_df_file_name):	# Weekday so market is likely open, check if file needs to be refreshed
	file_date = datetime.fromtimestamp(os.path.getmtime(latest_df_file_name))
	market_close_today = datetime.now().replace(hour=17, minute=30, second=0)

	if (market_close_today - file_date).total_seconds() > 0 and (market_close_today - datetime.now()).total_seconds() < 0: # If file was created before market close and time now is after
		should_replace = True
		print('Market should be open today and {} will be refreshed'.format(latest_df_file_name))


if not os.path.isfile(latest_df_file_name) or should_replace:
	print('Creating dataset {}.'.format(latest_df_file_name))
	swe_symbols = ""
	with open(os.path.join('original_dfs', 'swe_500_symbols.txt'), 'r') as f:
		swe_symbols = f.read()

	latest_df = data_tools.download_symbols(swe_symbols.split(), start_date=start_date, end_date=(today + timedelta(days=1)))
	latest_df.to_hdf(latest_df_file_name, 'df', mode='w', format='fixed')	#save file

else:
	latest_df = pd.read_hdf(latest_df_file_name, 'df')


	# --- Process data ---
latest_df.columns = latest_df.columns.swaplevel(0, 1)
variables = latest_df.columns.get_level_values(1).unique()
symbols = latest_df.columns.get_level_values(0).unique()


print('Normalizing data...')

n_df = latest_df.copy()
n_df = n_df.fillna(method='ffill')
n_df = n_df.fillna(0)

for sym in symbols:	#iterate through stocks and normalize data for that stock
	v_df = n_df[sym]
	nv_df=(v_df-v_df.min())/(v_df.max()-v_df.min())
	n_df[sym] = nv_df

print('Done.')


latest_n_df = n_df[-hist_time_steps:]	# Hist time steps long data of 500 swedish stocks
print('Latest row: {}'.format(latest_n_df[-1:]))


# --- Find good stocks to buy ---
def disp_pred(prev_price, pred, title, show_now=True):
	fig, ax = plt.subplots(figsize=(6,4))
	fig.suptitle(title)
	ax.plot(prev_price)
	ax.plot(range(hist_time_steps + len(pred)), np.append([None] * hist_time_steps, pred))
	if show_now:
		plt.show()


def disp_stock(sym):
	start_hist = n_df[-hist_time_steps:-hist_time_steps+1].index.date
	start_year = n_df[-365*1:-365*1+1].index.date
	fig, ax = plt.subplots(nrows=3, figsize=(6,6))
	#fig.tight_layout()
	fig.subplots_adjust(hspace=0.37, top=0.9)

	plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=-20, fontsize=8)
	plt.setp(ax[2].xaxis.get_majorticklabels(), rotation=-30, fontsize=8)

	fig.suptitle('Close price for {}'.format(sym))
	ax[0].plot(n_df[sym]['Close'][-365*5:])
	ax[1].plot(n_df[sym]['Close'][-365*1:])
	ax[2].plot(n_df[sym]['Close'][-hist_time_steps:])
	ax[0].axvline(x=(start_hist), color='red', linestyle='dashed')
	ax[1].axvline(x=(start_hist), color='red', linestyle='dashed')
	ax[2].axvline(x=(start_hist), color='red', linestyle='dashed')

	ax[0].axvline(x=(start_year), color='grey', linestyle='dashed')
	ax[1].axvline(x=(start_year), color='grey', linestyle='dashed')

	plt.show()



# Save as spreadsheet
n_symbols = len(symbols)

input_data = np.zeros((n_symbols, hist_time_steps, 6))	# Shape of input data
for x in range(n_symbols):
	input_data[x] = latest_n_df[symbols[x]].values


predictions = model.predict(input_data)

# Normalize prediction (to percent change from previous close)
for sym_i in range(len(predictions)):
	prev_close = input_data[sym_i][-1][1]

	pred = predictions[sym_i]
	for i in range(len(pred)):
		pred[i] = (pred[i] - prev_close) / prev_close * 100	# Percent change

print('{} symbols, {} input rows, {} predictions.'.format(n_symbols, len(input_data), len(predictions)))


pred_df = pd.DataFrame(predictions, columns=list(range(len(predictions[0]))))	#rows are symbols, columns are pred values

pred_df_rename_dict = {}
for sym_i in range(len(symbols)):
	pred_df_rename_dict[sym_i] = symbols[sym_i]
pred_df.rename(index=pred_df_rename_dict, inplace=True)

pred_df.to_csv(os.path.join('recommendations', 'date{}-time{}-step{}-model-{}.csv'.format(today, datetime.time(datetime.now()), hist_time_steps, model_name)))


# Purchase strategy that seems to work
stocks_to_buy = []
n_sym = 0
n_rec = 0

for sym in symbols:
	n_sym+=1
	if n_sym%10 == 0:
		print('Number {} out of {}. {:.1f}% complete'.format(n_sym, len(symbols), n_sym/len(symbols)*100))
	X = input_data[n_sym-1]
	
	pred = predictions[n_sym-1]	# (-1 because idx start is 1)
	previous_close = X[-1][1]

	if pred[3] / previous_close > 1.07 and np.mean(pred[1:3]) / previous_close > 1.05:	# "Buy" conditions
		stocks_to_buy.append(sym)
		n_rec+=1
		#disp_pred(latest_n_df[sym]['Close'].values, pred, 'Close price for {}'.format(sym))

print('Scanned {} stocks. {} of them were recommended.'.format(n_sym, n_rec))

if input('Preview stocks: ') == 'y':
	for sym in stocks_to_buy:
		disp_stock(sym)


recommendations = " ".join(stocks_to_buy)
rec_txt_name = os.path.join('recommendations', 'date{}-time{}-step{}-model-{}.txt'.format(today, datetime.time(datetime.now()), hist_time_steps, model_name))
with open(rec_txt_name, 'w') as f:
	f.write(recommendations)


