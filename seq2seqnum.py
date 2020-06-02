import pandas as pd
import numpy as np
import string
from string import digits
import matplotlib.pyplot as plt
#%matplotlib inline
import tensorflow as tf
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import re
import os
import io
import time
import random

#enumerate means if [10, 11, 12, 13, 14, 15], then say [x[1] for x in enumerate(a) if x[0] in [1,2,5]]. returs enumerate object. can make it a tuple by list() command. different from a list as tuples are immutable

ar=[]
for i in range(3000):
    arr=np.random.randint(0,51,4)
    ar1=np.sort(arr)
    z=[[[-10]+arr.tolist()+[-12],[-10]+ar1.tolist()+[-12]]]#to convert numpy to a list do this its always more convenient i guess when you're not dealing wih super complex calculations
    ar=ar+z
    arr=np.random.randint(0,51,6)
    ar1=np.sort(arr)
    z=[[[-10]+arr.tolist()+[-12],[-10]+ar1.tolist()+[-12]]]#to convert numpy to a list do this its always more convenient i guess when you're not dealing wih super complex calculations
    ar=ar+z
    arr=np.random.randint(0,51,9)
    ar1=np.sort(arr)
    z=[[[-10]+arr.tolist()+[-12],[-10]+ar1.tolist()+[-12]]]#to convert numpy to a list do this its always more convenient i guess when you're not dealing wih super complex calculations
    ar=ar+z
    arr=np.arange(0,10)
    ar1=arr 
    np.random.shuffle(arr)
    z=[[[-10]+arr.tolist()+[-12],[-10]+ar1.tolist()+[-12]]]#to convert numpy to a list do this its always more convenient i guess when you're not dealing wih super complex calculations
    ar=ar+z

'''for i in range(51):
    arr=np.random.randint(0,51,6)
    ar1=np.sort(arr)
    z=[[[-10]+arr.tolist()+[-12],[-10]+ar1.tolist()+[-12]]]#to convert numpy to a list do this its always more convenient i guess when you're not dealing wih super complex calculations
    ar=ar+z
for i in range(51):
    arr=np.random.randint(0,51,9)
    ar1=np.sort(arr)
    z=[[[-10]+arr.tolist()+[-12],[-10]+ar1.tolist()+[-12]]]#to convert numpy to a list do this its always more convenient i guess when you're not dealing wih super complex calculations
    ar=ar+z
for i in range(101):
    arr=np.arange(0,10)
    ar1=arr 
    np.random.shuffle(arr)
    z=[[[-10]+arr.tolist()+[-12],[-10]+ar1.tolist()+[-12]]]#to convert numpy to a list do this its always more convenient i guess when you're not dealing wih super complex calculations
    ar=ar+z
'''

#print(ar[5])
#print(k for k in range(0,3))
#source,target = zip(*ar) will need to add a couplle of for loops here
#cannot take numpy array anywhere
#source = [[ar[i][0] for i in range[0,50]]] why does this not work
source,target = zip(*[[ar[i][k] for k in range(0,2)] for i in range(12000)])#this only works with tuples
from collections import Counter, OrderedDict
counters=Counter()
countert=Counter()
for i in source:
    for j in range(len(i)):
        counters[i[j]]=i[j]
for i in target: 
    for j in range(len(i)):
        countert[i[j]]=i[j] 

y1 = [list(ele) for ele in counters.most_common()]  
y2 = [list(ele) for ele in countert.most_common()]
#print(len(countert.keys()))
sourcei=list(source)
targeti=list(target)
for i in range(len(sourcei)):
    for j in range(len(sourcei[i])):
        sourcei[i][j]=[x for x,y in enumerate(y1) if y[1]==sourcei[i][j]][0]
for i in range(len(targeti)):
    for j in range(len(targeti[i])):
        targeti[i][j]=[x for x,y in enumerate(y1) if y[1]==targeti[i][j]][0]

#Sequences that are shorter than num_timesteps, padded with 0 at the end.
source_tensor= tf.keras.preprocessing.sequence.pad_sequences(sourcei,padding='post' )#works with lists
#print(source)
# Post pad the shorter sequences with 0
target_tensor= tf.keras.preprocessing.sequence.pad_sequences(target,padding='post' )
source_train_tensor, source_test_tensor, target_train_tensor, target_test_tensor= train_test_split(source_tensor, target_tensor,test_size=0.2)



#setting the BATCH SIZE
BATCH_SIZE = 64
#Create data in memeory 
dataset=tf.data.Dataset.from_tensor_slices((source_train_tensor, target_train_tensor)).shuffle(BATCH_SIZE)
# shuffles the data in the batch
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

#Creates an Iterator for enumerating the elements of this dataset.
#Extract the next element from the dataset
source_batch, target_batch =next(iter(dataset))
print(len(source_train_tensor))

BUFFER_SIZE = len(source_train_tensor)
steps_per_epoch= len(source_train_tensor)//BATCH_SIZE
embedding_dim=512
units=1024
source_vocab_size= len(counters.keys())
target_vocab_size= len(countert.keys())

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, encoder_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size= batch_size
        self.encoder_units=encoder_units
        self.embedding=tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru= tf.keras.layers.GRU(encoder_units, return_sequences=True,return_state=True,   recurrent_initializer='glorot_uniform')
    
    def call(self, x, hidden):
        #pass the input x to the embedding layer
        x= self.embedding(x)#embedding is always of the INDEX in a list
        # pass the embedding and the hidden state to GRU
        output, state = self.gru(x, initial_state=hidden)
        return output, state
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.encoder_units))

encoder = Encoder(source_vocab_size, embedding_dim, units, BATCH_SIZE)
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden= encoder(source_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super( BahdanauAttention, self).__init__()
        self.W1= tf.keras.layers.Dense(units)  # encoder output
        self.W2= tf.keras.layers.Dense(units)  # Decoder hidden
        self.V= tf.keras.layers.Dense(1)
    
    def call(self, query, values):
        #calculate the Attention score
        hidden_with_time_axis= tf.expand_dims(query,1)
        
        score= self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights= tf.nn.softmax(score, axis=1)
        
         #context_vector 
        context_vector= attention_weights * values
       
        #Computes the sum of elements across dimensions of a tensor
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

attention_layer= BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, decoder_units, batch_sz):
        super (Decoder,self).__init__()
        self.batch_sz= batch_sz
        self.decoder_units = decoder_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru= tf.keras.layers.GRU(decoder_units, return_sequences= True,return_state=True,recurrent_initializer='glorot_uniform')
        # Fully connected layer
        self.fc= tf.keras.layers.Dense(vocab_size)
        
        # attention
        self.attention = BahdanauAttention(self.decoder_units)
    
    def call(self, x, hidden, encoder_output):
        
        context_vector, attention_weights = self.attention(hidden,  encoder_output)
        
        # pass output sequnece thru the input layers
        x= self.embedding(x)
        
        # concatenate context vector and embedding for output sequence
        x= tf.concat([tf.expand_dims( context_vector, 1), x],axis=-1)
        
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        
        # output shape == (batch_size * 1, hidden_size)
        output= tf.reshape(output, (-1, output.shape[2]))
        
        # pass the output thru Fc layers
        x= self.fc(output)
        return x, state, attention_weights

decoder= Decoder(target_vocab_size, embedding_dim, units, BATCH_SIZE)
sample_decoder_output, _, _= decoder(tf.random.uniform((BATCH_SIZE,1)), sample_hidden, sample_output)
print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

#Define the optimizer and the loss function
optimizer = tf.keras.optimizers.Adam()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')



def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  return tf.reduce_mean(loss_)

checkpoint_dir = 'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,encoder=encoder,decoder=decoder)

@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        #create encoder
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        #first input to decode is start_
        dec_input = tf.expand_dims([x for x,y in enumerate(y1) if y[1]==-10] * BATCH_SIZE, 1)
        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
          # passing enc_output to the decoder
          predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
          # calculate loss based on predictions  
          loss += loss_function(targ[:, t], predictions)
          # using teacher forcing
          dec_input = tf.expand_dims(targ[:, t], 1)
    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

EPOCHS=25
for epoch in range(EPOCHS):
  start = time.time()
  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0
  # train the model using data in bataches 
  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
    batch_loss = train_step(inp, targ, enc_hidden)
    total_loss += batch_loss
    if batch % 4 == 0:
      print('Epoch {} Batch {} Loss {}'.format(epoch + 1,batch,batch_loss.numpy()))
       # saving (checkpoint) the model every 2 epochs
  if (epoch + 1) % 2 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)
  print('Epoch {} Loss {}'.format(epoch + 1,total_loss / steps_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


#Calculating the max length of the source and target sentences
max_target_length= max(len(t) for t in  target_tensor)
max_source_length= max(len(t) for t in source_tensor)

def evaluate(sentence):
    attention_plot= np.zeros((max_target_length, max_source_length))
    #preprocess the sentnece
    inputs = [-10]+ sentence+ [-12]
    inputsi=inputs

    for i in range(len(inputsi)):
        inputsi[i]=[x for x,y in enumerate(y1) if y[1]==inputsi[i]][0]
    
    # pad the sequence 
    inputsi= tf.keras.preprocessing.sequence.pad_sequences([inputsi], maxlen=max_source_length, padding='post')
    
    #conver to tensors
    inputsi = tf.convert_to_tensor(inputsi)
    
    result= []
    
    # creating encoder
    hidden = [tf.zeros((1, units))]
    encoder_output, encoder_hidden= encoder(inputsi, hidden)
    
    # creating decoder
    decoder_hidden = encoder_hidden
    decoder_input = tf.expand_dims([[x for x,y in enumerate(y1) if y[1]==-10][0]], 0)
    
    for t in range(max_target_length):
        predictions, decoder_hidden, attention_weights= decoder(decoder_input, decoder_hidden, encoder_output)
        
        # storing attention weight for plotting it
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()
        
        prediction_id= tf.argmax(predictions[0]).numpy()
        
        
        if y1[prediction_id][0] == -12:
            return result,sentence, attention_plot
        result += [y1[prediction_id][0]]
        # predicted id is fed back to as input to the decoder
        decoder_input = tf.expand_dims([prediction_id], 0)
        
    return result,sentence, attention_plot

#def plot_attention(attention, sentence, predicted_sentence):
    #fig = plt.figure(figsize=(10,10))

def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)
    
    print('Input : %s' % (sentence))
    print('predicted sentence :{}'.format(result))
    
    #attention_plot= attention_plot[result, sentence]
    #plot_attention(attention_plot, sentence, result)

translate([1,5,2,6,3])
translate([45,41,2,4,9])
translate([33,27,8,3,39])
translate([1,13,18,22,46,38])

