from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from prepare_data_for_classification import *
from corpus import *
import numpy as np

MY_CORPUS = Corpus('..\\corpora.txt', 'English', seed=123, exclude_stop_words=True)


# Nach Kapitel 15 in Deep Learning for NLP

def load_doc(file):
    with open(file) as f:
        document = f.read()
    return document


# load an already cleaned dataset
def load_clean_dataset(vocab, is_train):
    global MY_CORPUS
    # language ist zB so angegeben "English"

    # load documents
    # UNKNOWN wird hier gerade auch als Label angesehen

    docs = []
    for class_name in MY_CORPUS.gold_classes:
        for story in MY_CORPUS.gold_classes[class_name]:
            if (story in MY_CORPUS.train_stories and is_train) or (story in MY_CORPUS.test_stories and not is_train):
                docs.append(' '.join([word for word in MY_CORPUS.extract_word_sequence(story) if word in vocab]))

    # prepare labels
    ## Labels:
    # animal    --> 0
    # magic     --> 1
    # religious --> 2
    # realistic --> 3
    # ogre      --> 4
    # jokes     --> 5
    # formula   --> 6
    # UNKNOWN   --> 7

    # im Buch gibt es nur zwei Labels
    def _amount_of_stories_in_class_of_train_or_test_subset(class_name, is_train):
        result = 0
        for story in MY_CORPUS.gold_classes[class_name]:
            if (story in MY_CORPUS.train_stories and is_train) or (story in MY_CORPUS.test_stories and not is_train):
                result += 1
        return result

    labels = np.array(
        [0 for _ in range(_amount_of_stories_in_class_of_train_or_test_subset('animal', is_train))]
        + [1 for _ in range(_amount_of_stories_in_class_of_train_or_test_subset('magic', is_train))]
        + [2 for _ in range(_amount_of_stories_in_class_of_train_or_test_subset('religious', is_train))]
        + [3 for _ in range(_amount_of_stories_in_class_of_train_or_test_subset('realistic', is_train))]
        + [4 for _ in range(_amount_of_stories_in_class_of_train_or_test_subset('ogre', is_train))]
        + [5 for _ in range(_amount_of_stories_in_class_of_train_or_test_subset('jokes', is_train))]
        + [6 for _ in range(_amount_of_stories_in_class_of_train_or_test_subset('formula', is_train))]
        + [7 for _ in range(_amount_of_stories_in_class_of_train_or_test_subset('UNKNOWN', is_train))])
    return docs, labels


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# integer encode and pad documents
def encode_docs(tokenizer, max_length, docs):
    # integer encode
    encoded = tokenizer.texts_to_sequences(docs)
    # pad sequences
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded


# define the model
def define_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))

    # Output Layer erhält 8 Nodes, für die 8 Label
    model.add(Dense(8, activation='sigmoid'))

    # compile network
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])

    # plot_model(model, to_file='model.png', show_shapes=True)

    return model


def main():
    # load training data
    vocab = set()
    occurrence_count = {}
    for story in MY_CORPUS:
        for word in MY_CORPUS.extract_word_sequence(story):
            vocab.add(word)
            if word in occurrence_count:
                occurrence_count[word] += 1
            else:
                occurrence_count[word] = 1
    vocab = set(sorted(vocab, reverse=True, key=lambda x: occurrence_count[x])[0:500])
    # print(vocab)

    # load all documents
    train_docs, ytrain = load_clean_dataset(vocab, True)
    test_docs, ytest = load_clean_dataset(vocab, False)

    ytrain = to_categorical(ytrain)
    ytest = to_categorical(ytest)

    # create the tokenizer
    tokenizer = create_tokenizer(train_docs)

    # define vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary size: %d' % vocab_size)

    # calculate the maximum sequence length
    max_length = max([len(s.split()) for s in train_docs])
    print('Maximum length: %d' % max_length)

    # encode data
    Xtrain = encode_docs(tokenizer, max_length, train_docs)
    Xtest = encode_docs(tokenizer, max_length, test_docs)

    # define model
    model = define_model(vocab_size, max_length)

    # fit network
    model.fit(Xtrain, ytrain, epochs=5, verbose=2, validation_split=0.1)

    # save the model
    # model.save('model.h5')

    # evaluate model on training dataset
    _, acc = model.evaluate(Xtrain, ytrain, verbose=0)
    print('Train Accuracy: %.2f' % (acc * 100))

    # evaluate model on test dataset
    _, acc = model.evaluate(Xtest, ytest, verbose=0)
    print('Test Accuracy: %.2f' % (acc * 100))

    return model
