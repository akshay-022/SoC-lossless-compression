import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(0,50,501)
y=np.sin(x)
plt.plot(x,y)
plt.show()
df=pd.DataFrame(data=y, index=x, columns=['sine'])
test_percent=0.1
test_point= np.round(len(df)*test_percent)
test_ind= int(len(df)-test_point)
train=df.iloc[:test_ind]#returs numpy array, all those ROWS no
#returns df
test=df.iloc[test_ind:]
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
scaler.fit(train)#acts on arrays and df? yes
scaled_train = scaler.tranform(train)
scaled_test = scaler.tranform(test)
from tensorflow.keras.preprocessing.sequence import TimeSeriesGenerator
length= 50
batch_size= 1

generator= TimeSeriesGenerator(scaled_train, scaled_train, length=length, batch_size=batch_size)
#first two args are x and y
#len(generator)=449, scaledtrain=451
#just breaks entire data into batches nothing else
X,y= generator[0]#x is 2 valued array, y is 1
#taks df as input and returns gen array different only probably

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM
nfeat=1
model=Sequential()
'''model.add(SimpleRNN(length,input_shape=(length, nfeat)))
model.add(Dense(1))
model.compile(optimizer='adam',loss= 'mse')

model.fit_generator(generator,epochs=5 )
first_eval_batch=scaled_train[-length:]#gives numpy array
first_eval_batch= first_eval_batch.reshape(1, length, nfeat)
model.predict(first_eval_batch)#compare it with scaled_tet[0]

test_pred=[]
first_eval_batch=scaled_train[-length:]#also a df#takes from those net indexes
current_batch= first_eval_batch.reshape((1,length,nfeat))

#if predicted_value=[[[99]]]
#then np.append(current_batch[:,1:,:],[[[99]]],axis=1) append along the column 
#more dim, more brackets

for i in range(len(test)):
	current_pred= model.predict(current_batch)[0]
	test_pred.append(current_pred)
	current_batch= np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
	#as prediction always returned as an array

true_predictions= scaler.inverse_transform(test_pred)
#everywhere else we transform scale of xs but in Rnn we tranform scale of ys
test['predictions']=true_predictions
test.plot(figsize=(12,8))
#scaling always returns an array even if input is a df

from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor='val_loss',patience=2)#not 0 as can get kinda noisy at times
validation_generator= TimeSeriesGenerator(scaled_test, scaled_test, length=49, batch_size=1)
#as it has only 49 elements if length =50 as in training data, will overshoot so error
'''
model.add(LSTM(length,input_shape=(length, nfeat)))
model.add(Dense(1))
model.compile(optimizer='adam',loss= 'mse')

from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor='val_loss',patience=2)#not 0 as can get kinda noisy at times
validation_generator= TimeSeriesGenerator(scaled_test, scaled_test, length=49, batch_size=1)
#as it has only 49 elements if length =50 as in training data, will overshoot so error


model.fit_generator(generator,epochs=20,validation_data= validation_generator, callbacks=[early_stop] )#as itll stop anyway
first_eval_batch=scaled_train[-length:]#gives numpy array
first_eval_batch= first_eval_batch.reshape(1, length, nfeat)
model.predict(first_eval_batch)#compare it with scaled_tet[0]

test_pred=[]
first_eval_batch=scaled_train[-length:]#also a df#takes from those net indexes
current_batch= first_eval_batch.reshape((1,length,nfeat))

#if predicted_value=[[[99]]]
#then np.append(current_batch[:,1:,:],[[[99]]],axis=1) append along the column 
#more dim, more brackets

for i in range(len(test)):
	current_pred= model.predict(current_batch)[0]#two dimensional(2 brackets) 
	test_pred.append(current_pred)#to append here we need ony 1 bracket 
	current_batch= np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
	#to append here we need 3 brackets as its [something 1 somethnig]
	#as prediction always returned as an array

true_predictions= scaler.inverse_transform(test_pred)
#everywhere else we transform scale of xs but in Rnn we tranform scale of ys
test['predictions']=true_predictions
test.plot(figsize=(12,8))
#scaling always returns an array even if input is a df
#models always accept 3 dimensional stuff probably
#during.read_csv as an argument say ,parse_dates=True. It'll make the dates the index
#or also infer_datetime_format=True
#forecast_idnex = pd.date_range(start='2019-11-01', periods=12, freq='MS'(meaning months))
#forescast_df= pd.DataFrame(data=forecast, index=forecast_index, columns=['forecast'])
#ax=df.plot()
#forecast_df.plot(ax=ax) to plot om the same graph
#plt.xlim('2018-01-01','2020-12-01') to zoom in on the imortant part of the graph
#df.plot (figsize=(12,8))
#first argument in lstm is number of neurons in the rnn
#model.history.history has 2 loss and val loss maybe thats why create datafrae first them loss.plot();
#to increase eprforamce ca decrease test set and can also increase number of neurons in RNN part
#toooooo many neurons may cause overfitting
#can do df.loc['index name of thatth entry':] to get all entries after that.S