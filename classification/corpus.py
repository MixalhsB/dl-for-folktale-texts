# OLD NAME: corpus

import ast
import string
import unicodedata

import numpy as np
import random
import keras
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


class Corpus:

    def __init__(self, filename, language, test_split=0.2, seed=123, exclude_stop_words=False):
        if exclude_stop_words:
            self.stop_words = stopwords.words(language)
        else:
            self.stop_words = []

        with open(filename, 'r', encoding='utf-8') as f:
            dict_syntaxed_string = f.read()

        self.stories = ast.literal_eval(dict_syntaxed_string)[language]

        if seed is not None:
            random.seed(seed)
            random.shuffle(self.stories)

        self.class_names = ('animal', 'magic', 'religious', 'realistic', 'ogre', 'jokes', 'formula', 'UNKNOWN')

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

        # print(_get_number('UNKNOWN'))
        # assert False

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

    def __iter__(self):
        return iter(self.stories)

    def tokenize(self, text):
        tokens = []
        text += '\n'

        def _is_delimiter(c):
            return not c.isalpha() and not c.isdigit()

        current = ''
        for size, char in enumerate(text):
            if _is_delimiter(char):
                if len(current) > 0:
                    if current.lower() not in self.stop_words:
                        tokens.append(current.lower())
                current = ''
            else:
                current += char
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
            model.add(Dense(len(self.class_names)))
            model.add(Activation('softmax'))

            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            binary_transformed_data = self.get_binary_transformed_train_and_test_data(max_words)
            (bin_x_train, bin_y_train), (bin_x_test, bin_y_test) = binary_transformed_data
            model.fit(bin_x_train, bin_y_train, batch_size=32, epochs=2, verbose=1, validation_split=0.1)
            self.simple_reuters_model = model

        return self.simple_reuters_model
