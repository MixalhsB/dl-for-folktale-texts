# OLD NAME: classifier

from keras.preprocessing.text import Tokenizer
from bs4 import BeautifulSoup
from gensim.models.doc2vec import TaggedDocument
import numpy as np
import random


class Classifier:

    def __init__(self, corpus):
        self.corpus = corpus

    def dumb_classify(self, list_of_html_texts):
        result = []
        for html_text in list_of_html_texts:
            random.seed(hash(html_text))
            result.append(random.choice(self.corpus.class_names))
        return result

    def simple_reuters_classify(self, list_of_html_texts):
        model = self.corpus.get_trained_model_for_simple_reuters_classifier()
        list_of_i2list_representations = []
        for html_text in list_of_html_texts:
            raw_text = BeautifulSoup(html_text, "html.parser").text
            word_sequence = self.corpus.tokenize(raw_text)
            max_index = max(self.corpus.w2i_dict.values())
            i2list_representation = [self.corpus.w2i_dict[word] if word in self.corpus.w2i_dict else max_index + 1
                                     for word in word_sequence]
            list_of_i2list_representations.append(i2list_representation)
        x_test = list_of_i2list_representations
        tokenizer = Tokenizer(num_words=10000)
        bin_x_test = np.array([tokenizer.sequences_to_matrix(i2list_rep, mode='binary') for i2list_rep in x_test])
        if self.corpus.binary_mode:
            predictions = model.predict_proba(bin_x_test)
            return [self.corpus.class_names[int(round(predictions[i][0]))] for i in range(len(bin_x_test))]
        else:
            predictions = model.predict_classes(bin_x_test)
            return [self.corpus.class_names[predictions[i]] for i in range(len(bin_x_test))]

    def book_inspired_classify(self, list_of_html_texts):
        model, vocab, tokenizer, max_length, encode_docs = self.corpus.get_trained_model_data_for_book_classifier()
        docs_to_be_encoded = []
        for html_text in list_of_html_texts:
            raw_text = BeautifulSoup(html_text, "html.parser").text
            word_sequence = self.corpus.tokenize(raw_text)
            docs_to_be_encoded.append(' '.join([word for word in word_sequence if word in vocab]))
        x_test = encode_docs(tokenizer, max_length, docs_to_be_encoded)
        if self.corpus.binary_mode:
            predictions = model.predict_proba(x_test)
            return [self.corpus.class_names[int(round(predictions[i][0]))] for i in range(len(x_test))]
        else:
            predictions = model.predict_classes(x_test)
            return [self.corpus.class_names[predictions[i]] for i in range(len(x_test))]
        
    def length_classify(self, list_of_html_texts):
        result = []
        lengths = self.corpus.get_avg_story_lengths()
        for html_text in list_of_html_texts:
            raw_text = BeautifulSoup(html_text, "html.parser").text
            word_sequence = self.corpus.tokenize(raw_text)
            this_length = len(word_sequence)
            comparison = [(abs(lengths[cn] - this_length), cn) for cn in lengths]
            comparison.sort()
            result.append(comparison[0][1])
        return result

    def doc2vec_classify(self, list_of_html_texts):
        model_dbow, logreg, vector_for_learning = self.corpus.get_trained_model_data_for_doc2vec_classifier()
        dummy_tagged_docs = []
        for html_text in list_of_html_texts:
            raw_text = BeautifulSoup(html_text, "html.parser").text
            word_sequence = self.corpus.tokenize(raw_text)
            dummy_tagged_docs.append(TaggedDocument(word_sequence, tags=[-1]))
        vector_representation = vector_for_learning(model_dbow, dummy_tagged_docs)
        predictions = logreg.predict(vector_representation[1])
        return [self.corpus.class_names[predictions[i]] for i in range(len(predictions))]

    def ngram_classify(self, list_of_html_texts):
        model, tokenizer, length = self.corpus.get_ngram_model()
        docs_to_be_encoded = []
        for html_text in list_of_html_texts:
            raw_text = BeautifulSoup(html_text, "html.parser").text
            word_sequence = self.corpus.tokenize(raw_text)
            docs_to_be_encoded.append(' '.join([word for word in word_sequence]))
        x_test = self.corpus.encode_text(tokenizer, docs_to_be_encoded, length)
        # return [self.corpus.class_names[model.predict(x_test)[i]] for i in range(len(x_test))])
        predictions = model.predict([x_test, x_test, x_test])
        if self.corpus.binary_mode:
            return [self.corpus.class_names[int(round(predictions[i][0]))] for i in range(len(x_test))]
        else:
            return [self.corpus.class_names[np.argmax(predictions[i])] for i in range(len(x_test))]

