import glob
import os.path as path

from keras.utils import to_categorical
from keras.models import Sequential
from keras import layers
from keras.models import Model
from keras.callbacks import EarlyStopping

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model

import json
from pickle import load
from matplotlib import pyplot as plt
import numpy as np


def load_dataset(filename):
    return load(open(filename, 'rb'))


def max_length(lines):
    return max([len(s.split()) for s in lines])


def fit_tokenizer(text):
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(text)
    return tokenizer


def encode_text(tokenizer, text, length):
    encoded = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(encoded, padding='post', maxlen=length)
    return padded


def define_model(length, vocab_size):
    embedding_size = 32

    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_size, input_length=length))
    model.add(layers.LSTM(200))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(2, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def define_channelized_model(length, vocab_size):

    # channel 1
    inputs1 = layers.Input(shape=(length,))
    embedding1 = layers.Embedding(vocab_size, 100)(inputs1)
    conv1 = layers.convolutional.Conv1D(filters=32, kernel_size=4,
                                        activation='relu')(embedding1)
    drop1 = layers.Dropout(0.5)(conv1)
    pool1 = layers.convolutional.MaxPooling1D(pool_size=2)(drop1)
    flat1 = layers.Flatten()(pool1)
    reshaped1 = layers.Reshape([738, 32])(flat1)
    lstm1 = layers.LSTM(128, input_shape=(738, 32), activation='tanh',
                        recurrent_activation='sigmoid', return_sequences=True)(reshaped1)
    dense01 = layers.Dense(16, activation='relu')(lstm1)
    flat11 = layers.Flatten()(dense01)

    # channel 2
    inputs2 = layers.Input(shape=(length,))
    embedding2 = layers.Embedding(vocab_size, 100)(inputs2)
    conv2 = layers.convolutional.Conv1D(filters=32, kernel_size=6,
                                        activation='relu')(embedding2)
    drop2 = layers.Dropout(0.5)(conv2)
    pool2 = layers.convolutional.MaxPooling1D(pool_size=2)(drop2)
    flat2 = layers.Flatten()(pool2)
    reshaped2 = layers.Reshape([737, 32])(flat2)
    lstm2 = layers.LSTM(256, input_shape=(737, 32), activation='tanh',
                        recurrent_activation='sigmoid', return_sequences=True)(reshaped2)
    dense02 = layers.Dense(32, activation='relu')(lstm2)
    flat22 = layers.Flatten()(dense02)

    # channel 3
    inputs3 = layers.Input(shape=(length,))
    embedding3 = layers.Embedding(vocab_size, 100)(inputs3)
    conv3 = layers.convolutional.Conv1D(filters=32, kernel_size=8,
                                        activation='relu')(embedding3)
    drop3 = layers.Dropout(0.5)(conv3)
    pool3 = layers.convolutional.MaxPooling1D(pool_size=2)(drop3)
    flat3 = layers.Flatten()(pool3)
    reshaped3 = layers.Reshape([736, 32])(flat3)
    lstm3 = layers.LSTM(512, input_shape=(736, 32), activation='tanh',
                        recurrent_activation='sigmoid', return_sequences=True)(reshaped3)
    dense03 = layers.Dense(64, activation='relu')(lstm3)
    flat33 = layers.Flatten()(dense03)

    # merge
    merged = layers.merge.concatenate([flat11, flat22, flat33])
    # interpretation
    dense1 = layers.Dense(10, activation='relu')(merged)
    outputs = layers.Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    # compile
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # summarize
    print(model.summary())
    plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model


# define the max vocabulary limit
vocab_size = 10000

# load the training and testing lines and labels
trainLines, trainLabels = load_dataset('train.pkl')
testLines, testLabels = load_dataset('test.pkl')

# find the max sentence length in the reviews
print('Maximum review length: {}'.format(
    len(max((trainLines + testLines), key=len))))

max_words = 500
tokenizer = fit_tokenizer(trainLines)

word2id = tokenizer.word_index
id2word = {i: word for word, i in word2id.items()}
print(len(word2id))

print('max length = {}  vocab_size = {}'.format(max_words, vocab_size))

trainX = encode_text(tokenizer, trainLines, max_words)
testX = encode_text(tokenizer, testLines, max_words)

callback_array = [EarlyStopping(patience=3, monitor='val_loss')]

model = define_model(max_words, vocab_size)
# model = define_channelized_model(max_words, vocab_size)

history = model.fit(trainX, to_categorical(trainLabels), validation_data=(
    testX, to_categorical(testLabels)), epochs=10, batch_size=64, callbacks=callback_array)
# history = model.fit([trainX, trainX, trainX], np.array(trainLabels), epochs=10,
#                     batch_size=32, validation_data=([testX, testX, testX], np.array(testLabels)), callbacks=callback_array)

model.save('model.h5')

plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

scores = model.evaluate(testX, testLabels, verbose=0)
print(scores[1])
