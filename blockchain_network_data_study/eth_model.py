import pandas as pd 
import os
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
import time

import tensorflow as tf 

from tensorflow import keras
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.keras.models import Sequential

seq_len = 15
future_period_predict = 1
crypto_to_predict = 'ETH'

epochs = 15
batch_size = 64
name = f"{seq_len}_seq_{crypto_to_predict}_pred_{int(time.time())}"



def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0



def preprocessing(df):

	scaler = MinMaxScaler(feature_range=(0,1))
	df = scaler.fit_transform(df.values)

	seq_data = []
	prev_days = deque (maxlen=seq_len)

	for i in df: 
		prev_days.append([n for n in i[:-1]])
		if len (prev_days) == seq_len:
			seq_data.append([np.array(prev_days),i[-1]])

	x = []
	y = []

	for seq, target in seq_data:
		x.append(seq)
		y.append(target)

	return np.array(x), np.array(y)




main_df=pd.DataFrame()

cryptos = ["ETH"]
for crypto in cryptos:
	dataset = f'crypto_datav1/{crypto}.csv'
	df = pd.read_csv(dataset)
		#read in dataset using f strings

	df.rename(columns = {'':f'{crypto}_No.', 'asset': f'{crypto}_asset name', 'time': 'time', 'AdrActCnt': f'{crypto}_active address count', 'HashRate': f'{crypto}_HashRate', 'ReferenceRateUSD': f'{crypto}_USD', 'TxCnt': f'{crypto}_transaction count'}, inplace=True)
		#rename columns in dataframe with incoming file names 


	df['time'] = pd.to_datetime(df['time']).dt.date
	df = df.sort_values(by = 'time')
	df.set_index("time", inplace=True) #arrange by time 
	df = df[[f"{crypto}_active address count", f"{crypto}_transaction count", f"{crypto}_USD"]]
		#picking out the columns we want 

	if len (main_df) == 0:
		main_df=df 
	else: 
		main_df=main_df.join(df)

main_df.fillna(method="ffill", inplace=True)
main_df.dropna(inplace=True)

# scaler = MinMaxScaler(feature_range=(0,1))
# main_df = scaler.fit_transform(main_df.values)

#main_df ['future'] = main_df[f'{crypto_to_predict}_USD'].shift(-future_period_predict)
#main_df ['target'] = list(map(classify, main_df[f'{crypto_to_predict}_USD'], main_df['future']))

# print("orginal dataset")
# print(main_df.shape)
# print(main_df)
# print(main_df.dtypes)
# for c in main_df.columns:
# 	print(c)



times = sorted(main_df.index.values)
last_10pct = sorted(main_df.index.values)[-int(0.10*len(times))]
#print(last_10pct)

validation_main_df = main_df[(main_df.index >= last_10pct)]
main_df = main_df[(main_df.index < last_10pct)]


train_x, train_y = preprocessing(main_df)
validation_x, validation_y = preprocessing(validation_main_df)


# print("\nPreprocessed dataset")

# print(train_x.shape)
# print(train_y.shape)
# print(validation_x.shape)
# print(validation_y.shape)

# print(train_x) 
# print("\n",train_y)

# print("\n",train_x) 
# print("\n",train_y)


DROPOUT = 0.2
WINDOW_SIZE = seq_len

model = keras.Sequential()

model.add(tf.keras.layers.Bidirectional(
	tf.keras.layers.LSTM(WINDOW_SIZE, return_sequences=True),
	input_shape=(WINDOW_SIZE, train_x.shape[-1])))
model.add(Dropout(rate=DROPOUT))

model.add(tf.keras.layers.Bidirectional(
	tf.keras.layers.LSTM((WINDOW_SIZE * 2), return_sequences=True)))
model.add(Dropout(rate=DROPOUT))

model.add(tf.keras.layers.Bidirectional(
	tf.keras.layers.LSTM(WINDOW_SIZE, return_sequences=False)))

model.add(Dense(units=1))

model.add(Activation('sigmoid'))


model.compile(
   loss='mean_squared_error',
   optimizer='adam',
   metrics=['accuracy'])

history = model.fit(
   train_x,
   train_y,
   epochs=epochs,
   batch_size=batch_size,
   shuffle=False,
   validation_split=0.1)

score = model.evaluate(validation_x, validation_y, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()





