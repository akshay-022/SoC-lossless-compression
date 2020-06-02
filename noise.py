import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import GaussianNoise
sample= GaussianNoise(0.2)#std dev or something i guess
noisey= sample(X_test[:10],training =True)
#adds noise

(X_train,y_train),(X_test,y_test)= mnist.load_data()
plt.imshow(X_train[0])
X_train= X_train/255
X_test= X_test/255



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.optimizers import SGD
tf.random.set_seed(101)
np.random.seed(101)
#3-2-3
encoder=Sequential()
encoder.add(Flatten(input_shape=[28,28]))

encoder.add(GaussianNoise(0.2))

encoder.add(Dense(units=400, activation ='relu'))
encoder.add(Dense(units=200, activation ='relu'))
encoder.add(Dense(units=100, activation ='relu'))
encoder.add(Dense(units=50, activation ='relu'))
encoder.add(Dense(units=25, activation ='relu'))
#go down to around 3%

decoder=Sequential()
decoder.add(Dense(units=50, activation ='relu',input_shape=[25]))
decoder.add(Dense(units=100, activation ='relu'))
#input shape only refers to the shape of the first input into the entire NEURAL NETWORK NOT each layer so input shape even stated later will only refer to the first layer of the NN
decoder.add(Dense(units=200, activation ='relu'))
decoder.add(Dense(units=400, activation ='relu'))
decoder.add(Dense(units=784, activation ='sigmoid'))
decoder.add(Reshape([28,28]))
autoencoder= Sequential([encoder, decoder])
autoencoder.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])#learning rate
autoencoder.fit(X_train, X_train, epochs=8)
#x y both from X_train only
#basically we train it to discosider all the noise terms no matter how much noise is present i guess
t= sample(X_test[:10], training=True)
passed_images = autoencoder.predict(t)
#encoded_2dim.shape=(300.2)
plt.imshow[X_test[0]]
plt.show()
plt.imshow[t[0]]
plt.show()
plt.imshow[passed_images[0]]
plt.show()
#features are always column headings of the dataframe we are doing dot values of so do transpose etc accordingly
