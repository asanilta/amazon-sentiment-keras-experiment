import numpy
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import LSTM,Bidirectional
from keras.layers import SimpleRNN
from keras.layers import GRU
from keras.layers import Convolution1D, MaxPooling1D, Merge
from keras.engine import Input
from keras.optimizers import SGD
from keras.preprocessing import text,sequence
import pandas
import os
from gensim.models.word2vec import Word2Vec

# Input parameters
max_features = 5000
max_len = 200
embedding_size = 300

# Convolution parameters
filter_length = 3
nb_filter = 150
pool_length = 2
cnn_activation = 'relu'
border_mode = 'same'

# RNN parameters
output_size = 50
rnn_activation = 'tanh'
recurrent_activation = 'hard_sigmoid'

# Compile parameters
loss = 'binary_crossentropy'
optimizer = 'rmsprop'

# Training parameters
batch_size = 50
nb_epoch = 3
validation_split = 0.25
shuffle = True

# Read dataset
data = pandas.read_csv("dataset_clothing.csv",delimiter=',',quotechar='"',quoting=0,names=['review','sentiment'],header=None)
x = data['review'].apply(str).values
y = data['sentiment'].values

# Build vocabulary & sequences
tk = text.Tokenizer(nb_words=max_features, filters=text.base_filter(), lower=True, split=" ")
tk.fit_on_texts(x)
x = tk.texts_to_sequences(x)
word_index = tk.word_index
x = sequence.pad_sequences(x,maxlen=max_len)

# Build pre-trained embedding layer
w2v = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

embedding_matrix = numpy.zeros((len(word_index) + 1, embedding_size))
for word,i in word_index.items():
  if word in w2v.vocab:
    embedding_matrix[i] = w2v[word]

embedding_layer = Embedding(len(word_index)+1,
                            embedding_size,
                            weights=[embedding_matrix],
                            input_length=max_len)

#########################################
# Simple RNN

model = Sequential()
model.add(embedding_layer)
model.add(SimpleRNN(output_dim=output_size, activation=rnn_activation))
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

print('Simple RNN')
model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,validation_split=validation_split,shuffle=shuffle)

#########################################
# GRU

model = Sequential()
model.add(embedding_layer)
model.add(GRU(output_dim=output_size,activation=rnn_activation,recurrent_activation=recurrent_activation))
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

print('GRU')
model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,validation_split=validation_split,shuffle=shuffle)

#################################################
# Bidirectional LSTM

model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(LSTM(ouput_dim=output_size,activation=rnn_activation,recurrent_activation=recurrent_activation)))
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

print('Bidirectional LSTM')
model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,validation_split=validation_split,shuffle=shuffle)

##################################################
# LSTM

model = Sequential()
model.add(embedding_layer)
model.add(Dropout(dropout))
model.add(LSTM(output_dim=output_size,activation=rnn_activation,recurrent_activation=recurrent_activation))
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

print('LSTM')
model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,validation_split=validation_split,shuffle=shuffle)

##########################################################
# CNN + LSTM

model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.5))
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode=border_mode,
                        activation=cnn_activation,
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(LSTM(output_dim=output_size,activation=rnn_activation,recurrent_activation=recurrent_activation))
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

print('CNN + LSTM')
model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,validation_split=validation_split,shuffle=shuffle)

###########################################################
# CNN
# Based on "Convolutional Neural Networks for Sentence Classification" by Yoon Kim http://arxiv.org/pdf/1408.5882v2.pdf
# https://github.com/keon/keras-text-classification/blob/master/train.py

filter_sizes = (3,4,5)
num_filters = 100
graph_in = Input(shape=(max_len, embedding_size))
convs = []
for fsz in filter_sizes:
    conv = Convolution1D(nb_filter=num_filters,
                         filter_length=fsz,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1)(graph_in)
    pool = MaxPooling1D(pool_length=2)(conv)
    flatten = Flatten()(pool)
    convs.append(flatten)

if len(filter_sizes) > 1:
    out = Merge(mode='concat')(convs)
else:
    out = convs[0]

graph = Model(input=graph_in, output=out)
model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.25, input_shape=(max_len, embedding_size)))
model.add(graph)
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
opt = SGD(lr=0.01, momentum=0.80, decay=1e-6, nesterov=True)
model.compile(loss=loss, optimizer='rmsprop', metrics=['accuracy'])
model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,validation_split=validation_split,shuffle=shuffle)
model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,validation_split=validation_split,shuffle=shuffle)


###########################################################
# Regular CNN 

# model = Sequential()
# model.add(embedding_layer)
# model.add(Dropout(0.25))
# model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation="relu", border_mode='same'))
# model.add(MaxPooling1D(5))
# model.add(Convolution1D(64, 5, activation=cnn_activation, border_mode=border_mode))
# model.add(MaxPooling1D(pool_length=pool_length))
# model.add(Flatten())
# model.add(Dense(64, activation="relu"))
# model.add(Dense(1, activation="sigmoid"))

# model.compile(loss=loss,optimizer=optimizer, metrics=['accuracy'])
# print('CNN')
# model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,validation_split=validation_split,shuffle=shuffle)