# OLD NAME: classifier

from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from bs4 import BeautifulSoup
from corpus import Corpus
import numpy as np
import random


class Classifier:

    def __init__(self, corpus):
        self.corpus = corpus

    def dumb_classify(self, html_text):
        random.seed(hash(html_text))
        return random.choice(self.corpus.class_names)

    def simple_reuters_classify(self, html_text):
        model = self.corpus.get_trained_model_for_simple_reuters_classifier()
        raw_text = BeautifulSoup(html_text, "html.parser").text
        word_sequence = [word for word in text_to_word_sequence(raw_text) if word not in self.corpus.stop_words]
        max_index = max(self.corpus.w2i_dict.values())
        i2list_representation = [self.corpus.w2i_dict[word] if word in self.corpus.w2i_dict else max_index + 1
                                 for word in word_sequence]
        x_test = np.empty(1, dtype=np.object)
        x_test[0] = i2list_representation
        tokenizer = Tokenizer(num_words=10000)
        bin_x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
        return self.corpus.class_names[model.predict_classes(bin_x_test)[0]]
