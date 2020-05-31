import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
import numpy as np
import io


class ImdbClassifier():
    vocab_size = 10000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    oov_tok = "<OOV>"

    def tokenize_data(self, train_sent, test_sent):
        print('tokenizing data')
        tokenizer = Tokenizer(num_words=self.vocab_size,
                              oov_token=self.oov_tok)
        tokenizer.fit_on_texts(train_sent)
        word_index = tokenizer.word_index
        train_seq = tokenizer.texts_to_sequences(train_sent)
        train_pad = pad_sequences(
            train_seq, maxlen=self.max_length, truncating=self.trunc_type)

        test_seq = tokenizer.texts_to_sequences(test_sent)
        test_pad = pad_sequences(test_seq, maxlen=self.max_length)

        return word_index, train_pad, test_pad

    def build_model(self):
        print('building model')
        model = tf.keras.Sequential([layers.Embedding(self.vocab_size, self.embedding_dim,
                                                      input_length=self.max_length),
                                     layers.Flatten(),
                                     layers.Dense(6, activation='leaky_relu'),
                                     layers.Dense(1, activation='sigmoid'),
                                     ])
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model


if __name__ == '__main__':
    imdb, info = tfds.load('imdb_reviews', with_info=True,
                           as_supervised=True, shuffle_files=True)
    train_data, test_data = imdb['train'], imdb['test']

    train_sent = []
    train_label = []

    test_sent = []
    test_label = []

    for s, l in train_data:
        train_sent.append(str(s.tonumpy()))
        train_label.append(l.tonumpy())
    for s, l in test_data:
        test_sent.append(str(s.tonumpy()))
        test_label.append(l.tonumpy())

    train_label_final = np.array(train_label)
    test_label_final = np.array(test_label)

    classifier = ImdbClassifier()
    word_index, train_padded, test_padded = classifier.tokenize_data(
        train_sent, test_sent)

    model = classifier.build_model()
    num_epochs = 20
    history = model.fit(train_padded, train_label_final, epochs=num_epochs,
                        validation_data=(test_padded, test_label_final))
