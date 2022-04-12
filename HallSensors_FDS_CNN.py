# multivariate multi-headed 1d cnn example
from numpy import array
from numpy import hstack
import math
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from keras.models import load_model

## remember to use the address of the data csv file 
df = read_csv('C:\\Users\\path.csv')

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# # define input sequence
in_seq1 = df['phase'].to_numpy()
in_seq2 = df['hallA'].to_numpy()
in_seq3 = df['hallB'].to_numpy()
in_seq4 = df['hallC'].to_numpy()
out_seq = df['state'].to_numpy()

in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
in_seq3 = in_seq3.reshape((len(in_seq3), 1))
in_seq4 = in_seq4.reshape((len(in_seq4), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

dataset = df.values

# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, in_seq3, in_seq4, out_seq))

# normalize the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaler = StandardScaler()
# dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# choose a number of time steps
n_steps = 50

# convert into input/output
X, y = split_sequences(train, n_steps)
testX, testY = split_sequences(test, n_steps)
allX, allY = split_sequences(dataset, n_steps)

y = to_categorical(y, 9)
testY = to_categorical(testY, 9)
# allY = to_categorical(allY, 5)

# one time series per head
n_features = 1

# separate input data
X1 = X[:, :, 0].reshape(X.shape[0], X.shape[1], n_features)
X2 = X[:, :, 1].reshape(X.shape[0], X.shape[1], n_features)
X3 = X[:, :, 2].reshape(X.shape[0], X.shape[1], n_features)
X4 = X[:, :, 3].reshape(X.shape[0], X.shape[1], n_features)


# Test Inputs
# x_input = array([[80, 85], [90, 95], [100, 105]])
x1 = testX[:, :, 0].reshape(testX.shape[0], testX.shape[1], n_features)
x2 = testX[:, :, 1].reshape(testX.shape[0], testX.shape[1], n_features)
x3 = testX[:, :, 2].reshape(testX.shape[0], testX.shape[1], n_features)
x4 = testX[:, :, 3].reshape(testX.shape[0], testX.shape[1], n_features)


a1 = allX[:, :, 0].reshape(allX.shape[0], allX.shape[1], n_features)
a2 = allX[:, :, 1].reshape(allX.shape[0], allX.shape[1], n_features)
a3 = allX[:, :, 2].reshape(allX.shape[0], allX.shape[1], n_features)
a4 = allX[:, :, 3].reshape(allX.shape[0], allX.shape[1], n_features)


# 1st input model
visible1 = Input(shape=(n_steps, n_features))
cnn1 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible1)
cnn1 = MaxPooling1D(pool_size=2)(cnn1)
cnn1 = Flatten()(cnn1)

# 2nd input model
visible2 = Input(shape=(n_steps, n_features))
cnn2 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible2)
cnn2 = MaxPooling1D(pool_size=2)(cnn2)
cnn2 = Flatten()(cnn2)

# 3rd input model
visible3 = Input(shape=(n_steps, n_features))
cnn3 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible3)
cnn3 = MaxPooling1D(pool_size=2)(cnn3)
cnn3 = Flatten()(cnn3)

# 4th input model
visible4 = Input(shape=(n_steps, n_features))
cnn4 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible4)
cnn4 = MaxPooling1D(pool_size=2)(cnn4)
cnn4 = Flatten()(cnn4)


# merge input models
merge = concatenate([cnn1, cnn2, cnn3, cnn4])
dense = Dense(50, activation='relu')(merge)
output = Dense(9, activation='softmax')(dense)
CNN_model = Model(inputs=[visible1, visible2, visible3, visible4], outputs=output)
# model.compile(loss='mse', optimizer='adam', metrics=['mae'])
CNN_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_accuracy', mode='auto', verbose=1, patience=100, min_delta=0, restore_best_weights=True)

# fit model
CNN_model.fit([X1, X2, X3, X4], y, epochs=1000, batch_size=20,
        validation_data=([x1, x2, x3, x4], testY), verbose=1, callbacks=[es])


# load model (delete the # to use the function) eg: use the path for saving the file
#CNN_model = load_model('C:\\Users\\path.hdf5')

# # make predictions
trainPredict = CNN_model.predict([X1, X2, X3, X4])

testPredict = CNN_model.predict([x1, x2, x3, x4])

score = CNN_model.evaluate([x1, x2, x3, x4], testY, batch_size=1, verbose=0)

# make predictions
allPredict = CNN_model.predict([a1, a2, a3, a4])

# save model (delete the # to use the function) eg: use the path for saving the file
##CNN_model.save('C:\\Users\\path.hdf5', overwrite=True)  
##CNN_model.save('CNN-bldc_HallSensor', overwrite=True)

# # summarize model
CNN_model.summary()

all = pd.DataFrame(allPredict)

# saving the dataframe eg: use the path for saving the file
all.to_csv('C:\\Users\\path.csv')
all.to_csv('CNN-bldc_HallSensor.csv')