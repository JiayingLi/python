# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import numpy as np
import pickle
np.random.seed(1337)
from keras.models import Model, model_from_json
from keras.optimizers import Adam, SGD
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.optimizers import *
from keras.models import Sequential
from keras.layers import Merge
from keras.models import load_model
import jieba
import sys
fenlei = [u"航空",u"能源",u"电器",u"交流",u"计算机",u"矿产",u"交通",u"艺术",u"环境",u"农业",u"经济",u"法律",u"医药",u"军事",u"政治",u"体育",u"文学",u"教育",u"哲学",u"历史"]

TEXT_DATA_DIR = './answer/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 200000
#EMBEDDING_DIM = 100
EMBEDDING_DIM = 50
#VALIDATION_SPLIT = 0.2
VALIDATION_SPLIT = 0.4

# first, build index mapping words in the embeddings set
# to their embedding vector
'''
print('Indexing word vectors.')

embeddings_index = {}
#f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
f = open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
'''

print('Indexing word vectors.')
embeddings_index = {}
#embeddings_index = Word2Vec.load('1.model')#('zhwiki_2017_03.word2vec')
#f = open(os.path.join(GLOVE_DIR, '125_vec'))
f = open('zhwiki_2017_03.word2vec')
num = f.readline()
print(num)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
filename = []
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        filename.append(name)
        for fname in sorted(os.listdir(path)):
            #if fname.isdigit():
            fpath = os.path.join(path, fname)
            if sys.version_info < (3,):
                f = open(fpath)
            else:
                f = open(fpath, encoding='latin-1')
            
            words = jieba.cut(f.read(),cut_all=False) 
            data = (' '.join(words)).encode('gb18030').strip()
            texts.append(data)
            f.close()
            labels.append(label_id)

#print(labels_index)
print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
f = open(sys.argv[1])
fenci = jieba.cut(f.read(),cut_all=False)
shuru = (' '.join(fenci)).encode('gb18030').strip()
aray = []
aray.append(shuru)
aray = tokenizer.texts_to_sequences(aray)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
test_data = pad_sequences(aray, maxlen=MAX_SEQUENCE_LENGTH)
print(test_data.shape)
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]

x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]
print('Preparing embedding matrix.')

# prepare embedding matrix
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector # word_index to word_embedding_vector ,<20000(nb_words)

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(nb_words + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,
                            weights=[embedding_matrix],
                            trainable=True)

print('Training model.')

print('Loading CNN model....')
model = model_from_json(open('files/2CNN3Layers_architecture.json').read(),{'embedding_layer': embedding_layer})    
"""
#left model
model_left = Sequential()
#model.add(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))
model_left.add(embedding_layer)
model_left.add(Conv1D(128, 5, activation='tanh'))
model_left.add(MaxPooling1D(5))

model_left.add(Conv1D(128, 5, activation='tanh'))
model_left.add(MaxPooling1D(5))

model_left.add(Conv1D(128, 5, activation='tanh'))
model_left.add(MaxPooling1D(35))
model_left.add(Dropout(0.5))
model_left.add(Flatten())

#right model
model_right = Sequential()
model_right.add(embedding_layer)
model_right.add(Conv1D(128, 4, activation='tanh'))
model_right.add(MaxPooling1D(4))

model_right.add(Conv1D(128, 4, activation='tanh'))
model_right.add(MaxPooling1D(4))

model_right.add(Conv1D(128, 4, activation='tanh'))
model_right.add(MaxPooling1D(28))
model_right.add(Dropout(0.5))
model_right.add(Flatten())

#third model
model_3 = Sequential()
model_3.add(embedding_layer)
model_3.add(Conv1D(128, 6, activation='tanh'))
model_3.add(MaxPooling1D(3))

model_3.add(Conv1D(128, 6, activation='tanh'))
model_3.add(MaxPooling1D(3))

model_3.add(Conv1D(128, 6, activation='tanh'))
model_3.add(MaxPooling1D(30))
model_3.add(Dropout(0.5))
model_3.add(Flatten())


merged = Merge([model_left, model_right,model_3], mode='concat') #merge
model = Sequential()
model.add(merged) # add merge
model.add(Dense(256, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(len(labels_index), activation='softmax'))
"""
#print('Loading CNN model....')
#model = model_from_json(open('files/2CNN3Layers_architecture.json').read())   
print('Loading model weights....')
model.load_weights('files/2CNN3Layers_model_weights.hdf5')
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
result = model.predict(test_data.reshape((1,1000)))
#print(labels[1500])
for i in range(len(result[0])):
    print(fenlei[i].encode('gb18030'),': ',result[0][i])