import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
(X_train,y_train),(X_test,y_test)= mnist.load_data()
plt.imshow(X_train[0])
X_train= (X_train/255*2)-1
X_test= (X_test/255*2)-1
only_zeros= X_train[y_train==0]#numpy array
#to get a smaller dataset
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape,LeakyReLU, BatchNormalization, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import SGD
discriminator= Sequential()
discriminator.add(Conv2D(64, kernel_size=5, strides=2,padding='same',activation=LeakyReLU(0.3), input_shape=28,28,1))
discriminator.add(Dropout(0.5))
discriminator.add(Conv2D(128, kernel_size=5, strides=2,padding='same',activation=LeakyReLU(0.3), input_shape=28,28,1))
discriminator.add(Dropout(0.5))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))#0 or 1
discriminator.compile(loss= 'binary_crossentropy',optimizer='adam')
codings_size= 100
generator= Sequential()
generator.add(Dense(7*7*128, activation='relu',input_shape=[codings_size]))
generator.add(Reshape([7,7,128]))
generator.add(Dense(BatchNormalization())#makes mean=0, var=1
generator.add(Conv2DTranspose(64, kernel_size=5, strides=2,padding='same',activation='relu'))
generator.add(Dense(BatchNormalization())
generator.add(Conv2DTranspose(1, kernel_size=5, strides=2,padding='same',activation='tanh'))
GAN= Sequential([generator, discriminator])
discriminator.trainable=False
GAN.compile(loss='binary_crossentropy',optimizer='adam')
batch_size=32
my_data= only_zeros
dataset= tf.data.Dataset.from_tensor_slices(my_data).shuffle(buffer_size=1000)
dataset=dataset.batch(batch_size, drop_remainder=True)
dataset=dataset.prefetch(1)
epochs=20#need gpu
#GAN.layers[0].layers gives generator layers
generator, discriminator= GAN.layers
for epoch in range(epochs):
	print(f"currently on epoch {epoch+1}")
	i=0
	for x_batch in dataset:
		i=i+1
		if i%100==0:
			print(f"\t Currently on batch number {i} of {len(my_data//batch_size)}")
		noise=tf.random.normal(shape=[batch_size, codings_size])
		gen_images=generator(noise)
		x_fake_vs_real= tf.concat([gen_images, tf.dtypes.cast(x_batch, tf.float32)], axis=0)
		y1= tf.constant([[0.0]]*batch_size+[[1.0]]*batch_size)#makes an array of ten zeros and ones
		discriminator.trainable= True
		discriminator.train_on_batch(x_fake_vs_real, y1)  #probably trains on the defaukt number of epochs(1) i guess
		noise= tf.random.normal(shape=[batch_size,codings_size])
		y2= tf.constant([[1.0]]*batch_size)
		discriminator.trainable= False
		GAN.train_on_batch(noise,y2)
noise=tf.random.normal(shape=[10,codings_size])
#noise_shape=[10,100] tensorshape
plt.imshow(noise)
images= generator(noise)
plt.imshow(images[0])#1 2 3 all look same oonly because of model collapse
#in conv2D recaling is /255*2-1(as htan -1 to 1, not 0 to 1)
#with dense etc import LeakyReLU, BatchNormalization, Conv2D, Conv2DTranspose






