import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
path_to_file= "shakespeare.txt"
text= open(path_to_file, 'r').read()
vocab=sorted(set(text))
#len(vocab)=84 incl \n etc
#for pair in enumerate(vocab):
#print(pair)
char_to_ind={char:ind for ind,char in enumerate(vocab)}
char_to_ind['H']
index_to_char= np.array(vocab)
#ind_to_char[33] gives 'h'
encoded_text= np.array([char_to_ind[c] for c in text])
#[reshape(300,)] just adds an extra braxket thast all
#encoded_text.shape is number of letters in the whole thing with their indices as the letters appear, in an array
#by len(line) find length of about 3 lines about 133
seq_len=120
total_num_seq= len(text)//(seq_len+1)#true div give integer answer
char_dataset= tf.data.Dataset.from_tensor_slices(encoded_text)#special slice dataset on which training can be done
#during printing have to convert this to numpy. char_... .take(500) creates dataset of 500 elements
sequences =char_dataset.batch(seq_length+1,drop_remainder=True) #dataset object
def create_seq_targets(seq):
	input_txt= seq[:-1]
	target_txt= seq[1:]
	return(input_txt,target_txt)
#dataset guves each one by one
dataset= sequences.map(create_seq_targets)
#input_text.numpy() convers the dataset into an aarray
#print("".join(ind_to_char[input_txt.numpy()]))
batch_size=128
buffer_size=10000
dataset= dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
#shuffle 10000 at a time and then divide into batches i guess
#data set has 2 of (128,120) each now
vocab_size=len(vocab)
embed_dim=64
rnn_neurons= 1026
from tensorflow.keras.losses import sparse_categorical_crossentropy
def sparse_cat_loss(y_true,y_pred):
	return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU
def create_model(vocab_size, embed_dim, rnn_neurons, batch_size):
	model= Sequential()
	model.add(Embdedding(vocab_size,embed_dim,batch_input_shape=[batch_size,None]))#none to match dims probably
	model.add(GRU(rnn_neurons,return_sequence=True,stateful=True,recurrent_initializer='glorot_uniform'))
	model.add(Dense(vocab_size))
	model.compile('adam',loss=sparse_cat_loss)
	return(model)
model= create_model(vocab_size=vocab_size, embed_dim=embed_dim, rnn_neurons=rnn_neurons,batch_size=batch_size)
for input_example_batch, target_example batch in dataset.take(1)
	example_batch predictions= model(input_example_batch)
#example_batch_predictions.shape=128,120,84
sampled_indices=tf.random.categorical(example_batch_predictions[0],num_samples)
sampled_indices=tf.squeeze(sampled_indices,axis=1).numpy()#squeeze removes all size one dim
epochs=30
#model.fit(dataset, epochs=epochs)
from tensorflow.keras.models import load_model
model=create_model(vocab_size, embed_dim,rnn_neurons,batch_size=1)
model.load_weights('shakespeare_gen.h5')#preprepared dataset as trsining it probably will take a lot of time
model.build(tf.Tensorshape([1,None]))
def generate_text(model,start_seed,gen_size=500,temp=1.0)
	num_generate= gen_size
	input_eval= [char_to_ind[s] for s in start_seed]
	input_eval=tf.expand_dims(input_eval,0)
	text_generated=[]
	temperature=temp
	model.reset_states()
	for i in range(num_generate):
		predictions = model(input_eval)#as models need an extra bracket everytime
		predictions =tf.squeeze(predictions,0)
		predictions=predictions/temperature
		predicted_id= tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
		input_eval= tf.expand_dims([predicted_id,0])
		text_generated.append(ind_to_char[predicted_id])
	return(start_seed+"".join(text_generated))
	print(generate_text(model."JULIET", GEN_SIZE=1000))

