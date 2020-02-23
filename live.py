import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, h5py, random, data_tools, pickle, os, platform, time, data_tools
from datetime import datetime, timedelta
from data_tools import trim_zeros


	# --- Requirements for data ---
hist_time_steps = 90

today = datetime.date(datetime.now())
start_date = '2000-01-01'

latest_df_file_name = os.path.join('original_dfs', 'latest', 'latest-{}-steps{}.h5'.format(today, hist_time_steps))


	# --- Download/load data ---
latest_df = None
if not os.path.isfile(latest_df_file_name):
	swe_symbols = ""
	with open(os.path.join('original_dfs', 'swe_500_symbols.txt'), 'r') as f:
		swe_symbols = f.read()

	latest_df = data_tools.download_symbols(swe_symbols.split(), start_date=start_date, end_date=today)
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

model_file_name = os.path.join('checkpoints', 'to-2019-06-checkpoints', 'weights-improvement-20-0.000289.h5')
#model_file_name = os.path.join('checkpoints', 'weights-improvement-18-0.000434.h5')
model_std_error = np.array([0.60684879, 0.57097587, 0.57766695, 0.50046783, 0.50637192, 0.59326133, 0.52038012]) # Inaccuracy based on val data
model = keras.models.load_model(model_file_name)

stocks_to_buy = []
n_sym = 0
n_rec = 0
for sym in symbols:
	n_sym+=1
	if n_sym%10 == 0:
		print('Number {} out of {}. {:.1f}% complete'.format(n_sym, len(symbols), n_sym/len(symbols)*100))
	X = latest_n_df[sym].values

	pred = model.predict(np.array([X]))[0]
	previous_close = X[-1][1]


	max_accuracy_day = np.argmin(model_std_error)
	if pred[max_accuracy_day] / previous_close > 1.07 and np.mean(pred[1:4]) / previous_close > 1.06:	# Buy conditions
		stocks_to_buy.append(sym)
		n_rec+=1
		#disp_pred(latest_n_df[sym]['Close'].values, pred, 'Close price for {}'.format(sym))

print('Scanned {} stocks. {} of them were recommended.'.format(n_sym, n_rec))

for sym in stocks_to_buy:
	disp_stock(sym)

recommendations = " ".join(stocks_to_buy)
with open(os.path.join('recommendations', '{}-{}step-recommendations.txt'.format(today, hist_time_steps)), 'w') as f:
	f.write(recommendations)
