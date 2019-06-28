'''
Steps to proceed
1) Preprocess the data
	- remove the punctuations
	- remove non-letter characters
	-

2) Use embedding to represent the data in dense vectors (Glove / word2vec).
3) Use app. no of CNN layers
4) add a dense layer or softmax layer for 5 classes.

'''

import keras
import numpy as np
import json 
import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import os
import keras.layers
from keras.layers import Embedding , LSTM , Bidirectional,GRU , Dropout
from keras.layers import Dense, Input, GlobalMaxPooling1D , Concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding , Flatten
from keras.models import Model
from keras.initializers import Constant


MAX_WORDS = 20000
GLOVE_DIR = 'glove.6B'
MAX_LENGTH = 1000
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 100

def preprocess_data(training_data):
	print("a")

def simple_cnn():
	sequence_input = Input(shape=(MAX_LENGTH,), dtype='int32')
	embedded_sequences = embedding_layer(sequence_input)
	x = Conv1D(128, 5, activation='relu')(embedded_sequences)
	x = MaxPooling1D(5)(x)
	x = Conv1D(128, 5, activation='relu')(x)
	x = MaxPooling1D(5)(x)
	x = Conv1D(128, 5, activation='relu')(x)
	x = MaxPooling1D(35)(x)  # global max pooling
	x = Flatten()(x)
	x = Dense(128, activation='relu')(x)
	preds = Dense(5, activation='softmax')(x)

	model = Model(sequence_input, preds)
	return model

def deeper_cnn():
	convs = []
	filters = [3,4,5]
	sequence_input = Input(shape=(MAX_LENGTH,), dtype='int32')
	embedded_sequences = embedding_layer(sequence_input)

	for fs in filters:
		l_conv = Conv1D(nb_filter=128,filter_length=fs,activation='relu')(embedded_sequences)
		l_pool = MaxPooling1D(5)(l_conv)
		convs.append(l_pool)
	l_merge = Concatenate(axis=1)(convs)
	x = Conv1D(128, 5, activation='relu')(l_merge)
	x = MaxPooling1D(5)(x)
	x = Conv1D(128, 5, activation='relu')(x)
	x = MaxPooling1D(35)(x)  # global max pooling
	x=Dropout(0.5)(x)
	x = Flatten()(x)
	x = Dense(128, activation='relu')(x)
	preds = Dense(5, activation='softmax')(x)

	model = Model(sequence_input, preds)
	return model

def lstm_cnn():
	


words_to_index={}
index_to_words={}
input_review_json_file = "./reviews_Grocery_and_Gourmet_Food_5.json"
total_review_data = []
with open(input_review_json_file,'r') as f:
	for i,line in enumerate(f):
		data = json.loads(line)
		total_review_data.append((i,data['reviewText'],data['overall']))

print(total_review_data[0])
train_x = [review[1] for review in total_review_data]
train_y = [review[2] for review in total_review_data]
train_y = [float(i)-1. for i in train_y]

tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(train_x)
sequences = tokenizer.texts_to_sequences(train_x)

word_index = tokenizer.word_index
paded = pad_sequences(sequences,MAX_LENGTH)
labels = to_categorical(np.asarray(train_y))

print(len(word_index))
print(labels.shape)
print(paded.shape)

# split the data into a training set and a validation set
indices = np.arange(paded.shape[0])
np.random.shuffle(indices)
data = paded[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * paded.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print(x_train.shape , x_val.shape)

##Importing embeddings from glove
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

##Only using the top most words
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		# words not found in embedding index will be all-zeros.
		embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape)

embedding_layer = Embedding(len(word_index) + 1,
							EMBEDDING_DIM,
							weights=[embedding_matrix],
							input_length=MAX_LENGTH,
							trainable=False)

##Training CNN
###Defining model

model = deeper_cnn()

print(model.summary())

model.compile(loss='categorical_crossentropy',
			  optimizer='adam',
			  metrics=['acc'])

# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),
		  epochs=3, batch_size=128)
test_sent = ["The cream was average","I would recommend anyone to buy this drink."]

test_sequences = tokenizer.texts_to_sequences(test_sent)

paded = pad_sequences(test_sequences,MAX_LENGTH)
a = model.predict(paded)
print(a)







