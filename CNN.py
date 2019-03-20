from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding, SpatialDropout1D
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

# set parameters:
max_features = 5000
maxlen = 4000
batch_size = 10
embedding_dims = 200
nb_filter = 16
filter_length = 5
hidden_dims = 250
nb_epoch = 3



print('Loading data...')
df = pd.read_csv('data/train_set.csv'nrows=1000)                        #Load 1000 records of the dataset
df_test = pd.read_csv('data/test_set.csv',nrows=1000)
test_id = pd.read_csv('data/test_set.csv',nrows=1000)[["id"]].copy()    #Copy id to the final output file for submission

#Extract the words for classification
x_load = df['word_seg']
x_test_load = df_test['word_seg']
labels =(df["class"]-1).astype(int)
y_train = to_categorical(np.asarray(labels))    #One-hot encoding for labels


print("The shape of training set is ",x_load.shape)
print("The shape of test set is ",y_train.shape)

x_load = [i.split(' ') for i in x_load]
x_test_load = [i.split(' ') for i in x_test_load]

#Feature extraction
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_load)
x_train = tokenizer.texts_to_sequences(x_load)
x_test = tokenizer.texts_to_sequences(x_test_load)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
X_train = pad_sequences(x_train, maxlen=maxlen)     #padding and truncating
X_test = pad_sequences(x_test, maxlen=maxlen)

'''

new_x=[]
for x in new_load:
    new_y = []
    for y in x:
        new_y.append( int(y))
    new_x.append(new_y)

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(x_load, maxlen=maxlen)
X_test = sequence.pad_sequences(x_test_load, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1) # transform the data into tfidf vector
x_train = vec.fit_transform(X_train)
x_test = vec.transform(X_test)
print(x_train.shape)
print('-----')
'''
print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(len(word_index)+1,
                    embedding_dims,
                    input_length=maxlen))
model.add(SpatialDropout1D(0.2))
# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(filters=nb_filter,
                        kernel_size=filter_length,
                        padding='valid',
                        activation='relu',
                        strides=1))
# we use max pooling:
model.add(MaxPooling1D(5))
model.add(Flatten())
# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a output layer with 19 unit, which corresponds to 19 output labels:
model.add(Dense(19))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=nb_epoch)

preds = model.predict(X_test)
preds = np.argmax(preds,axis=1)
test_pred = pd.DataFrame(preds)
test_pred.columns = ["class"]
test_pred["class"] = (test_pred["class"]+1).astype(int)
print(test_pred.shape)
print(test_id.shape)
test_pred["id"] = list(test_id["id"])

test_pred[["id","class"]].to_csv('result/CNN.csv',index=None)
