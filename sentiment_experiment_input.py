import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape, Embedding, LSTM
from keras.engine import Input
from keras.preprocessing import text,sequence
import pandas
import os
from gensim.models.word2vec import Word2Vec

# Input parameters
max_features = 2000
max_len = 200
gen_embedding_size = 200
twitter_glove_size = 200
amazon_glove_size = 200
amazon_w2v_size = 200
googlenews_w2v_size = 300

# Model parameters
lstm_output_size = 70
dropout = 0.25

# Compile parameters
loss = 'binary_crossentropy'
optimizer = 'adam'

# Training parameters
batch_size = 300
nb_epoch = 1
validation_split = 0.25
shuffle = True

# Read dataset
data = pandas.read_csv("dataset_clothing.csv",delimiter=',',quotechar='"',quoting=0,names=['review','sentiment'],header=None)
x = data['review'].apply(str).values
y = data['sentiment'].values

# Build vocabulary from dataset
tk = text.Tokenizer(nb_words=max_features, filters=text.base_filter(), lower=True, split=" ")
tk.fit_on_texts(x)
x = tk.texts_to_sequences(x)
word_index = tk.word_index

# Build sequences of word indexes
x = sequence.pad_sequences(x,maxlen=max_len)

# Load pretrained Twitter GloVe model
twitter_glove = {}
f = open('glove.twitter.27B.200d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = numpy.asarray(values[1:], dtype='float32')
    twitter_glove[word] = coefs
f.close()
print('Twitter GloVe loaded')

# Load GloVe model trained on Amazon clothing dataset
amazon_glove = {}
f = open('glovemodel.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = numpy.asarray(values[1:], dtype='float32')
    amazon_glove[word] = coefs
f.close()
print('Amazon GloVe loaded')

# Load pretrained Google News w2v model
googlenews_w2v = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print('Google News w2v loaded')

# Load w2v model trained on Amazon clothing dataset
amazon_w2v = Word2Vec.load('w2vmodel')
print('Amazon w2v loaded')

# Build embedding layers with weights initialized from each model
amazon_glove_matrix = numpy.zeros((len(word_index) + 1, amazon_glove_size))
twitter_glove_matrix = numpy.zeros((len(word_index) + 1, twitter_glove_size))
amazon_w2v_matrix = numpy.zeros((len(word_index) + 1, amazon_w2v_size))
googlenews_w2v_matrix = numpy.zeros((len(word_index) + 1, googlenews_w2v_size))

for word,i in word_index.items():
  if twitter_glove.get(word) is not None:
    twitter_glove_matrix[i] = twitter_glove[word]
  if amazon_glove.get(word) is not None:
    amazon_glove_matrix[i] = amazon_glove[word]
  if word in amazon_w2v.vocab:
    amazon_w2v_matrix[i] = amazon_w2v[word]
  if word in googlenews_w2v.vocab:
    googlenews_w2v_matrix[i] = googlenews_w2v[word]

twitter_glove_emb = Embedding(len(word_index)+1,
                            twitter_glove_size,
                            weights=[twitter_glove_matrix],
                            input_length=max_len)

amazon_glove_emb = Embedding(len(word_index)+1,
                            amazon_glove_size,
                            weights=[amazon_glove_matrix],
                            input_length=max_len)

amazon_w2v_emb = Embedding(len(word_index)+1,
                            amazon_w2v_size,
                            weights=[amazon_w2v_matrix],
                            input_length=max_len)
googlenews_w2v_emb = Embedding(len(word_index)+1,
                            googlenews_w2v_size,
                            weights=[googlenews_w2v_matrix],
                            input_length=max_len)
print('Embedding layers created')

##########################################################
# No embedding (sequence of word indexes)

model = Sequential()
model.add(Reshape((max_len,1),input_shape=(max_len,)))
model.add(LSTM(output_dim=lstm_output_size))
model.add(Dropout(dropout))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

print('No embedding')
model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,validation_split=validation_split,shuffle=shuffle)

##########################################################
# Non-pretrained Embedding (uninitialized weights)

model = Sequential()
model.add(Embedding(len(word_index)+1, gen_embedding_size, input_length=max_len))
model.add(LSTM(output_dim=lstm_output_size))
model.add(Dropout(dropout))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

print('Non-pretrained Embedding')
model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,validation_split=validation_split,shuffle=shuffle)

##########################################################
# Static Amazon GloVe Embedding

model = Sequential()
model.add(amazon_glove_emb)
model.add(LSTM(output_dim=lstm_output_size))
model.add(Dropout(dropout))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.layers[1].trainable=False
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

print('Static Amazon GloVe')
model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,validation_split=validation_split,shuffle=shuffle)

##########################################################
# Dynamic Amazon GloVe Embedding

model = Sequential()
model.add(amazon_glove_emb)
model.add(LSTM(output_dim=lstm_output_size))
model.add(Dropout(dropout))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

print('Dynamic Amazon GloVe')
model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,validation_split=validation_split,shuffle=shuffle)

##########################################################
# Static Twitter GloVe Embedding

model = Sequential()
model.add(twitter_glove_emb)
model.add(LSTM(output_dim=lstm_output_size))
model.add(Dropout(dropout))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.layers[1].trainable=False
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

print('Static Twitter GloVe')
model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,validation_split=validation_split,shuffle=shuffle)

##########################################################
# Dynamic Twitter GloVe Embedding

model = Sequential()
model.add(twitter_glove_emb)
model.add(LSTM(output_dim=lstm_output_size))
model.add(Dropout(dropout))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

print('Dynamic Twitter GloVe')
model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,validation_split=validation_split,shuffle=shuffle)

##########################################################
# Static Amazon GloVe Embedding

model = Sequential()
model.add(googlenews_w2v_emb)
model.add(LSTM(output_dim=lstm_output_size))
model.add(Dropout(dropout))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.layers[1].trainable=False
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

print('Static Google News Word2Vec')
model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,validation_split=validation_split,shuffle=shuffle)

##########################################################
# Dynamic Google News Word2Vec Embedding

model = Sequential()
model.add(googlenews_w2v_emb)
model.add(LSTM(output_dim=lstm_output_size))
model.add(Dropout(dropout))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

print('Dynamic Google News Word2Vec')
model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,validation_split=validation_split,shuffle=shuffle)


##########################################################
# Static Amazon Word2Vec Embedding

model = Sequential()
model.add(amazon_w2v_emb)
model.add(LSTM(output_dim=lstm_output_size))
model.add(Dropout(dropout))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.layers[1].trainable=False
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

print('Static Amazon Word2Vec')
model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,validation_split=validation_split,shuffle=shuffle)

##########################################################
# Dynamic Amazon Word2Vec Embedding

model = Sequential()
model.add(amazon_w2v_emb)
model.add(LSTM(output_dim=lstm_output_size))
model.add(Dropout(dropout))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

print('Dynamic Amazon Word2Vec')
model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,validation_split=validation_split,shuffle=shuffle)

##########################################################
# One-hot

# x_onehot = numpy.zeros((len(x),max_len,max_features))
# for i in range(0,len(x)-1):
#   for j in range(0,max_len-1):
#     word_idx = x[i][j]
#       if word_idx is not 0:
#       x_onehot[i][j][word_idx] = 1

# model = Sequential()
# model.add(LSTM(input_shape=(max_len,max_features),output_dim=lstm_output_size))
# model.add(Dropout(dropout))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
# model.compile(loss=loss,
#               optimizer=optimizer,
#               metrics=['accuracy'])

# print('One-hot')
# model.fit(x_onehot, y, batch_size=batch_size, nb_epoch=nb_epoch,validation_split=validation_split,shuffle=shuffle)