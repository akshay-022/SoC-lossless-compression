import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
data= make_blobs(n_samples=300, n_features=2, centers=2, cluster_std=1.0, random_state=101)
#std is std deviatin or noise
np.random.seed(seed=101)
z_noise= np.random.normal(size=len(x))
z_noise=pd.Series(z_noise)
feat= pd.concat([feat, z_noise],axis=1)
feat.columns= ['X1','X2','X3']
plt.scatter(feat['X1'],feat['X2'],c=y)#colour them based on y

from mpl_toolkits.mplot3d import Axes3D
#%matplotlib notebook
#to get iteractive plot
fig= plt.figure()
ax= fig.add_subplot(1111, projection)
ax.scatter(feat['X1'],feat['X2'],feat['X3'])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
#3-2-3
encoder=Sequential()
encoder.add(Dense(units=2, activation ='relu', input_shape=[3]))
decoder=Sequential()
decoder.add(Dense(units=3, activation ='relu', input_shape=[2]))
autoencoder= Sequential([encoder, decoder])
autoencoder.compile(loss='mse',optimizer=SGD(lr=1.5))
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaled_data= scaler.fit_transform(feat)
autoencoder.fit(scaled_data, scaled_data, epochs=5)

encoded_2dim= encoder.predict(scaled_data)
#encoded_2dim.shape=(300.2)
plt.scatter(encoded_2dim[:,0],encoded_2dim[:,1])