# OLD NAME: corpus

import ast
import numpy as np
import random
import keras
import os
import string
import unicodedata
import pyLDAvis
import pyLDAvis.gensim

from collections import defaultdict
from nltk.corpus import stopwords
from nltk import word_tokenize
from bs4 import BeautifulSoup
from gensim import models
from gensim import corpora
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D


class Corpus:

    def __init__(self, filename, language, test_split=0.2, seed=None, exclude_stop_words=False, binary_mode=False):
        self.binary_mode = binary_mode

        if exclude_stop_words:
            self.stop_words = stopwords.words(language) + (['thee', 'thy', 'thou', 'ye'] if language == 'English'
                                                           else [])
        else:
            self.stop_words = []

        with open(filename, 'r', encoding='utf-8') as f:
            dict_syntaxed_string = f.read()

        if self.binary_mode:
            self.class_names = ('non-magic', 'magic')
        else:
            self.class_names = ('animal', 'magic', 'religious', 'realistic', 'ogre', 'jokes', 'formula')  # , 'UNKNOWN')

        self.test_split = test_split

        def _get_number(atu_string):
            try:
                atu_int = int(''.join(char for char in atu_string if char.isdigit()))
                if atu_int < 1 or atu_int > 2399:
                    return -1
                else:
                    return atu_int
            except ValueError:
                return -1

        # Filtert jetzt UNKNOWN-Stories schon beim Einlesen des Korpus heraus:
        self.stories = [st for st in ast.literal_eval(dict_syntaxed_string)[language] if _get_number(st[2]) != -1]
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.stories)

        def _get_atu_range(class_name):
            if class_name == 'animal':
                return 1, 299
            elif class_name == 'magic':
                return 300, 749
            elif class_name == 'religious':
                return 750, 849
            elif class_name == 'realistic':
                return 850, 999
            elif class_name == 'ogre':
                return 1000, 1199
            elif class_name == 'jokes':
                return 1200, 1999
            elif class_name == 'formula':
                return 2000, 2399
            elif class_name == 'UNKNOWN':
                return -1, -1

        def _is_atu_in_range(atu_string, class_name):
            if self.binary_mode:
                magic_minimum, magic_maximum = _get_atu_range('magic')
                if class_name == 'magic':
                    return magic_minimum <= _get_number(atu_string) <= magic_maximum
                elif class_name == 'non-magic':
                    return 1 <= _get_number(atu_string) < magic_minimum \
                           or magic_maximum < _get_number(atu_string) <= 2399
            else:
                try:
                    minimum, maximum = _get_atu_range(class_name)
                    return minimum <= _get_number(atu_string) <= maximum
                except AssertionError:
                    return True

        def _get_stories_of_class(class_name):
            return [story for story in self.stories if _is_atu_in_range(story[2], class_name)]

        iter_over_class_specific_subsets = (_get_stories_of_class(class_name) for class_name in self.class_names)

        self.gold_classes = {class_name: stories for class_name, stories in
                             zip(self.class_names, iter_over_class_specific_subsets)}

        self.w2i_dict = self.get_word_to_index_dict()

        x_y_test_length = int(self.test_split * len(self.stories))
        self.test_stories = self.stories[:x_y_test_length]
        self.train_stories = self.stories[x_y_test_length:]

        self.simple_reuters_model = None
        self.book_model_data = (None, None, None, None, None)

    def __iter__(self):
        return iter(self.stories)

    #######################################
    #  NAIVE LENGTH-BASED CLASSIFICATION  #
    #######################################

    def get_avg_story_lengths(self):
        # Gibt ein Dictionary mit den Klassennamen
        # und deren durchschnittlicher Märchenlänge zurück
        story_lengths = defaultdict(list)
        for story in self.stories:
            length = len(self.extract_word_sequence(story))
            gold = self.get_gold_class_name(story)
            if gold in story_lengths:
                story_lengths[gold][0] += length
                story_lengths[gold][1] += 1
            else:
                story_lengths[gold] = [length, 1]
        result = {class_name: story_lengths[class_name][0] / story_lengths[class_name][1]
                  for class_name in story_lengths}
        return result

    #######################################
    #    SIMPLE REUTERS CLASSIFICATION    #
    #######################################

    def tokenize(self, text):
        '''
        tokens = []
        text += '\n'

        current = ''
        for size, char in enumerate(text):
            if _is_delimiter(char):
                if len(current) > 0:
                    if current.lower() not in self.stop_words:
                        tokens.append(current.lower())
                current = ''
            else:
                current += char
        '''

        def _is_acceptable_token(possible_token):
            for c in possible_token:
                delimiter_chars = string.punctuation + string.whitespace
                if not (c in delimiter_chars or unicodedata.category(c)[0] in 'ZP'):
                    return True
            return False

        tokens = word_tokenize(text)
        tokens = [tkn.lower() for tkn in tokens]
        tokens = [tkn for tkn in tokens if _is_acceptable_token(tkn)]
        tokens = [tkn for tkn in tokens if all(part_token not in self.stop_words for part_token in tkn.split("'"))]
        return tokens

    def extract_word_sequence(self, story):
        html_text = story[4]
        raw_text = BeautifulSoup(html_text, "html.parser").text
        return self.tokenize(raw_text)

    def get_word_occurrences(self):
        result = []
        for story in self.stories:
            result += self.extract_word_sequence(story)
        return result

    def get_word_frequencies(self):
        word_occurrences = self.get_word_occurrences()
        result = {}
        for word in word_occurrences:
            if word in result:
                result[word] += 1
            else:
                result[word] = 1
        return result

    def get_index_to_word_list(self):
        word_frequencies = self.get_word_frequencies()
        return reversed(sorted(word_frequencies, key=lambda x: word_frequencies[x]))

    def get_word_to_index_dict(self):
        i2w_list = self.get_index_to_word_list()
        return {word: index for index, word in enumerate(i2w_list)}

    def get_index_list_representation(self, story):
        word_sequence = self.extract_word_sequence(story)
        return [self.w2i_dict[word] for word in word_sequence]

    def get_gold_class_name(self, story):
        # TODO: why so complicated?
        this_story_id = story[1]
        for class_name in self.class_names:
            for any_story in self.gold_classes[class_name]:
                other_story_id = any_story[1]
                if this_story_id == other_story_id:
                    return class_name
        return 'UNKNOWN'

    def get_train_and_test_data(self):
        x_test = np.array([self.get_index_list_representation(story) for story in self.test_stories])
        x_train = np.array([self.get_index_list_representation(story) for story in self.train_stories])

        y_test = np.array([self.class_names.index(self.get_gold_class_name(story)) for story in self.test_stories])
        y_train = np.array([self.class_names.index(self.get_gold_class_name(story)) for story in self.train_stories])

        return (x_train, y_train), (x_test, y_test)

    def get_binary_transformed_train_and_test_data(self, max_words=10000):
        (x_train, y_train), (x_test, y_test) = self.get_train_and_test_data()
        tokenizer = Tokenizer(num_words=max_words)
        bin_x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
        bin_x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
        if self.binary_mode:
            bin_y_train = y_train
            bin_y_test = y_test
        else:
            bin_y_train = keras.utils.to_categorical(y_train, len(self.class_names))
            bin_y_test = keras.utils.to_categorical(y_test, len(self.class_names))
        return (bin_x_train, bin_y_train), (bin_x_test, bin_y_test)

    def get_trained_model_for_simple_reuters_classifier(self):
        if self.simple_reuters_model is None:
            max_words = 10000

            model = Sequential()
            model.add(Dense(512, input_shape=(max_words,)))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1 if self.binary_mode else len(self.class_names)))
            model.add(Activation('sigmoid' if self.binary_mode else 'softmax'))

            # TODO: how meaningful is this?
            if self.binary_mode:
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
            else:
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

            binary_transformed_data = self.get_binary_transformed_train_and_test_data(max_words)
            (bin_x_train, bin_y_train), (bin_x_test, bin_y_test) = binary_transformed_data
            model.fit(bin_x_train, bin_y_train, batch_size=(None if self.binary_mode else 32), epochs=2, verbose=1,
                      validation_split=0.1)
            self.simple_reuters_model = model

        return self.simple_reuters_model

    #######################################
    #    BOOK-INSPIRED CLASSIFICATION     #
    #######################################

    @staticmethod
    def load_doc(file):
        with open(file) as f:
            document = f.read()
        return document

    # load an already cleaned dataset
    def load_clean_dataset(self, vocab, is_train):
        # language ist zB so angegeben "English"
        # load documents

        docs = []
        for class_name in self.gold_classes:
            for story in self.gold_classes[class_name]:
                if (story in self.train_stories and is_train) or (story in self.test_stories and not is_train):
                    docs.append(' '.join([word for word in self.extract_word_sequence(story) if word in vocab]))

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
            for story in self.gold_classes[class_name]:
                if (story in self.train_stories and is_train) or (story in self.test_stories and not is_train):
                    result += 1
            return result

        if self.binary_mode:
            labels = np.array(
                [0.0 for _ in range(_amount_of_stories_in_class_of_train_or_test_subset('non-magic', is_train))]
                + [1.0 for _ in range(_amount_of_stories_in_class_of_train_or_test_subset('magic', is_train))])
        else:
            labels = np.array(
                [0 for _ in range(_amount_of_stories_in_class_of_train_or_test_subset('animal', is_train))]
                + [1 for _ in range(_amount_of_stories_in_class_of_train_or_test_subset('magic', is_train))]
                + [2 for _ in range(_amount_of_stories_in_class_of_train_or_test_subset('religious', is_train))]
                + [3 for _ in range(_amount_of_stories_in_class_of_train_or_test_subset('realistic', is_train))]
                + [4 for _ in range(_amount_of_stories_in_class_of_train_or_test_subset('ogre', is_train))]
                + [5 for _ in range(_amount_of_stories_in_class_of_train_or_test_subset('jokes', is_train))]
                + [6 for _ in range(_amount_of_stories_in_class_of_train_or_test_subset('formula', is_train))])
            # + [7 for _ in range(_amount_of_stories_in_class_of_train_or_test_subset('UNKNOWN', is_train))])
            # print(labels)

        return docs, labels

    # fit a tokenizer
    @staticmethod
    def create_tokenizer(lines):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer

    # integer encode and pad documents
    @staticmethod
    def encode_docs(tokenizer, max_length, docs):
        # integer encode
        encoded = tokenizer.texts_to_sequences(docs)
        # pad sequences
        padded = pad_sequences(encoded, maxlen=max_length, padding='post')
        return padded

    # define the model
    def define_model(self, vocab_size, max_length):
        model = Sequential()
        model.add(Embedding(vocab_size, 100, input_length=max_length))
        model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(10, activation='relu'))

        if self.binary_mode:
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

        else:
            # Output Layer erhält 7 Nodes, für die 7 Label
            model.add(Dense(7, activation='softmax'))

            # compile network
            # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['categorical_accuracy'])

        # plot_model(model, to_file='model.png', show_shapes=True)
        return model

    def create_trained_model_data(self):
        # load training data
        vocab = set()
        occurrence_count = {}
        for story in self.stories:
            for word in self.extract_word_sequence(story):
                vocab.add(word)
                if word in occurrence_count:
                    occurrence_count[word] += 1
                else:
                    occurrence_count[word] = 1
        vocab = set(sorted(vocab, reverse=True, key=lambda x: occurrence_count[x])[0:500])
        # print(vocab)

        # load all documents
        train_docs, ytrain = self.load_clean_dataset(vocab, True)
        # test_docs, ytest = self.load_clean_dataset(vocab, False)

        if not self.binary_mode:
            ytrain = to_categorical(ytrain)
            # ytest = to_categorical(ytest)

        # create the tokenizer
        tokenizer = Corpus.create_tokenizer(train_docs)

        # define vocabulary size
        vocab_size = len(tokenizer.word_index) + 1
        print('Vocabulary size: %d' % vocab_size)

        # calculate the maximum sequence length
        max_length = max([len(s.split()) for s in train_docs])
        print('Maximum length: %d' % max_length)

        # encode data
        Xtrain = Corpus.encode_docs(tokenizer, max_length, train_docs)
        # Xtest = Corpus.encode_docs(tokenizer, max_length, test_docs)

        # define model
        model = self.define_model(vocab_size, max_length)

        # fit network
        model.fit(Xtrain, ytrain, epochs=5, verbose=2, validation_split=0.1)

        # save the model
        # model.save('model.h5')

        # evaluate model on training dataset
        # _, acc = model.evaluate(Xtrain, ytrain, verbose=0)
        # print('Train Accuracy: %.2f' % (acc * 100))

        # evaluate model on test dataset
        # _, acc = model.evaluate(Xtest, ytest, verbose=0)
        # print('Test Accuracy: %.2f' % (acc * 100))

        return model, vocab, tokenizer, max_length, Corpus.encode_docs

    def get_trained_model_data_for_book_classifier(self):
        if self.book_model_data == (None, None, None, None, None):
            self.book_model_data = self.create_trained_model_data()
        return self.book_model_data

    #######################################
    #         LDA TOPIC MODELLING         #
    #######################################

    def run_lda(self):
        while True:
            user_input = input('-> Choose number of topics (e.g. 7): ')
            if user_input.isdigit():
                num_topics = int(user_input)
                if num_topics < 2:
                    print('-> Number of topics must be at least 2.')
                    continue
                break
            else:
                print('-> Please enter a valid number.')

        list_of_list_of_tokens = []
        for story in self.stories:
            raw_text = BeautifulSoup(story[4], "html.parser").text
            word_sequence = self.tokenize(raw_text)
            list_of_list_of_tokens.append(word_sequence)

        dictionary_LDA = corpora.Dictionary(list_of_list_of_tokens)
        dictionary_LDA.filter_extremes(no_below=3)

        sample = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in list_of_list_of_tokens]

        lda_model = models.LdaModel(sample, num_topics=num_topics, id2word=dictionary_LDA, random_state=1,
                                    passes=4, alpha='auto', eta='auto')

        vis = pyLDAvis.gensim.prepare(topic_model=lda_model, corpus=sample, dictionary=dictionary_LDA,
                                      sort_topics=False)
        if not os.path.isdir('./temporary'):
            os.mkdir('./temporary')
        pyLDAvis.save_html(vis, './temporary/lda.html')

        predominant_topics_dict = {}
        for i in range(len(self.stories)):
            predominant_topic = max(lda_model[sample[i]], key=lambda x: x[1])[0] + 1
            try:
                predominant_topics_dict[predominant_topic].append(i)
            except KeyError:
                predominant_topics_dict[predominant_topic] = [i]

        print()
        for i, topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=5):
            print(str(i + 1) + ": " + topic)
            print('This is the predominant topic of', len(predominant_topics_dict[i + 1]),
                  'documents.' if len(predominant_topics_dict[i + 1]) != 1 else 'document.')
            store_frequencies = defaultdict(int)
            for story_corpus_index in predominant_topics_dict[i + 1]:
                story = self.stories[story_corpus_index]
                gold = self.get_gold_class_name(story)
                store_frequencies[gold] += 1
            max_freq = -1
            max_arg = None
            for class_name in store_frequencies:
                if store_frequencies[class_name] > max_freq:
                    max_freq = store_frequencies[class_name]
                    max_arg = class_name
            print(max_arg.upper(), 'is the most common tale category in this cluster.\n')

    #######################################
    #        N-GRAM CLASSIFICATION        #
    #######################################

    def load_train_stories(self):
        # return list of stories, list of class numbers
        train_lines = []
        train_labels =[]
        test_lines = []
        test_labels =[]
        for class_name in self.gold_classes:
                for story in self.gold_classes[class_name]:
                    if story in self.train_stories:
                        train_lines.append(story)

                        if self.binary_mode:
                            if class_name == 'magic':
                                train_labels.append(1)
                            else:
                                train_labels.append(0)

                        else:
                            if class_name == 'animal':
                                train_labels.append(0)
                            elif class_name == 'magic':
                                train_labels.append(1)
                            elif class_name == 'religious':
                                train_labels.append(2)
                            elif class_name == 'realistic':
                                train_labels.append(3)
                            elif class_name == 'ogre':
                                train_labels.append(4)
                            elif class_name == 'jokes':
                                train_labels.append(5)
                            elif class_name == 'formula':
                                train_labels.append(6)
                            # elif class_name == 'UNKNOWN':
                            #     train_labels.append(7)
                    
                    else:
                        test_lines.append(story)

                        if self.binary_mode:
                            if class_name == 'magic':
                                test_labels.append(1)
                            else:
                                test_labels.append(0)

                        else:
                            if class_name == 'animal':
                                test_labels.append(0)
                            elif class_name == 'magic':
                                test_labels.append(1)
                            elif class_name == 'religious':
                                test_labels.append(2)
                            elif class_name == 'realistic':
                                test_labels.append(3)
                            elif class_name == 'ogre':
                                test_labels.append(4)
                            elif class_name == 'jokes':
                                test_labels.append(5)
                            elif class_name == 'formula':
                                test_labels.append(6)
                            # elif class_name == 'UNKNOWN':
                            #     test_labels.append(7)

        return train_lines, train_labels, test_lines, test_labels

    # fit a tokenizer
    # def self.create_tokenizer(self, lines):
    #     tokenizer = Tokenizer()
    #     tokenizer.fit_on_texts(lines)
    #     return tokenizer

    # calculate the maximum document length
    def max_length(self, lines):
        return max([len(s.split()) for s in lines])

    # encode a list of lines
    def encode_text(self, tokenizer, lines, length):
        # integer encode
        encoded = tokenizer.texts_to_sequences(lines)
        # pad encoded sequences
        padded = pad_sequences(encoded, maxlen=length, padding='post')
        return padded

    # define the model
    def define_model_n_gram(self, length, vocab_size):
        # channel 1
        inputs1 = Input(shape=(length,))
        embedding1 = Embedding(vocab_size, 100)(inputs1)
        conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
        drop1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling1D(pool_size=2)(drop1)
        flat1 = Flatten()(pool1)
        # channel 2
        inputs2 = Input(shape=(length,))
        embedding2 = Embedding(vocab_size, 100)(inputs2)
        conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
        drop2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling1D(pool_size=2)(drop2)
        flat2 = Flatten()(pool2)
        # channel 3
        inputs3 = Input(shape=(length,))
        embedding3 = Embedding(vocab_size, 100)(inputs3)
        conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling1D(pool_size=2)(drop3)
        flat3 = Flatten()(pool3)
        # merge
        merged = concatenate([flat1, flat2, flat3])
        # interpretation
        dense1 = Dense(10, activation='relu')(merged)
        outputs = Dense(1, activation='sigmoid')(dense1)
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
        # compile
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # summarize
        model.summary()
        # plot_model(model, show_shapes=True, to_file='model.png')
        return model

    def ngram_train(self):
        # load training dataset
        trainLines, trainLabels, testLines, testLabels = self.load_train_stories()
        # create tokenizer
        tokenizer = self.create_tokenizer(trainLines)
        # calculate max document length
        length = self.max_length(trainLines)
        # print('Max document length: %d' % length)
        # calculate vocabulary size
        vocab_size = len(tokenizer.word_index) + 1
        # print('Vocabulary size: %d' % vocab_size)
        # encode data
        trainX = self.encode_text(tokenizer, trainLines, length)
        # define model
        model = self.define_model_n_gram(length, vocab_size)
        # fit model
        trainX = np.asarray(trainX)
        trainLabels = np.asarray(trainLabels)
        model.fit([trainX,trainX,trainX], trainLabels, epochs=7, batch_size=16)
        # save the model
        # model.save('model.h5')
        return model, trainLines, trainLabels, testLines, testLabels, tokenizer, length, vocab_size, trainX
    
    def ngram_test(self, tokenizer, testLines, length):
        return self.encode_text(tokenizer, testLines, length)
