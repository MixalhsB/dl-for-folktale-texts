# OLD NAME: classifier

from keras.preprocessing.text import Tokenizer
from bs4 import BeautifulSoup
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
        word_sequence = self.corpus.tokenize(raw_text)
        max_index = max(self.corpus.w2i_dict.values())
        i2list_representation = [self.corpus.w2i_dict[word] if word in self.corpus.w2i_dict else max_index + 1
                                 for word in word_sequence]
        x_test = np.empty(1, dtype=np.object)
        x_test[0] = i2list_representation
        tokenizer = Tokenizer(num_words=10000)
        bin_x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
        if self.corpus.binary_mode:
            return self.corpus.class_names[int(round(model.predict_proba(bin_x_test)[0][0]))]
        else:
            return self.corpus.class_names[model.predict_classes(bin_x_test)[0]]

    def book_inspired_classify(self, html_text):
        model, vocab, tokenizer, max_length, encode_docs = self.corpus.get_trained_model_data_for_book_classifier()
        raw_text = BeautifulSoup(html_text, "html.parser").text
        word_sequence = self.corpus.tokenize(raw_text)
        x_test = encode_docs(tokenizer, max_length, [' '.join([word for word in word_sequence if word in vocab])])
        if self.corpus.binary_mode:
            return self.corpus.class_names[int(round(model.predict_proba(x_test)[0][0]))]
        else:
            return self.corpus.class_names[model.predict_classes(x_test)[0]]
        
    def length_classify(self, html_text):
        lengths = self.corpus.get_avg_story_lengths()
        raw_text = BeautifulSoup(html_text, "html.parser").text
        word_sequence = self.corpus.tokenize(raw_text)
        this_length = len(word_sequence)
        comparison = [(abs(lengths[cn] - this_length), cn) for cn in lengths]
        comparison.sort()
        return comparison[0][1]
