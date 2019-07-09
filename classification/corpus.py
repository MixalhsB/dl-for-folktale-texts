# OLD NAME: corpus

import ast
import numpy as np
import random
from bs4 import BeautifulSoup
from keras.preprocessing.text import text_to_word_sequence

class Corpus:
    def __init__(self, filename, language):
        with open(filename, 'r', encoding='utf-8') as f:
            dict_syntaxed_string = f.read()
        
        self.stories = ast.literal_eval(dict_syntaxed_string)[language]
        random.seed(123)
        random.shuffle(self.stories)
        
        self.class_names = ('animal', 'magic', 'religious', 'realistic', 'orge', 'jokes', 'formula', 'UNKNOWN')
        
        def give_number(atu_string):
            try:
                return int(''.join(char for char in atu_string if char.isdigit()))
            except ValueError:
                return -1
        
        def give_atu_range(class_name):
            if class_name == 'animal':
                return (1, 299)
            elif class_name == 'magic':
                return (300, 749)
            elif class_name == 'religious':
                return (750, 849)
            elif class_name == 'realistic':
                return (850, 999)
            elif class_name == 'orge':
                return (1000, 1199)
            elif class_name == 'jokes':
                return (1200, 1999)
            elif class_name == 'formula':
                return (2000, 2399)
            else:
                assert False
        
        def is_atu_in_range(atu_string, class_name):
            try:
                minimum, maximum = give_atu_range(class_name)
                return minimum <= give_number(atu_string) <= maximum
            except AssertionError:
                return True
        
        def give_stories_of_class(class_name):
            return [story for story in self.stories if is_atu_in_range(story[2], class_name)]
        
        iter_over_class_specific_subsets = (give_stories_of_class(class_name) for class_name in self.class_names)
        
        self.gold_classes = {class_name: stories for class_name, stories in zip(self.class_names, iter_over_class_specific_subsets)}
        
        self.w2i_dict = self.get_word_to_index_dict()
        
    def __iter__(self):
        return iter(self.stories)

    @staticmethod
    def extract_word_sequence(story):
        html_text = story[4]
        raw_text = BeautifulSoup(html_text, "html.parser").text
        result = text_to_word_sequence(raw_text)
        return result

    def get_word_occurrences(self):
        result = []
        for story in self.stories:
            result += self.extract_word_sequence(story)
        return result
    
    def get_word_frequencies(self):
        word_occurrences = self.get_word_occurrences()
        all_words = set(word_occurrences)
        result = {}
        for word in word_occurrences:
            if word in result:
                result[word] += 1
            else:
                result[word] = 1
        return result
    
    def get_index_to_word_list(self):
        word_frequencies = self.get_word_frequencies()
        result = reversed(sorted(word_frequencies, key=lambda x: word_frequencies[x]))
        return result
    
    def get_word_to_index_dict(self):
        i2w_list = self.get_index_to_word_list()
        result = {word: index for index, word in enumerate(i2w_list)}
        return result
    
    def yield_index_list_representation(self, story):
        word_sequence = self.extract_word_sequence(story)
        result = [self.w2i_dict[word] for word in word_sequence]
        return result
    
    def yield_gold_class(self, story):
        this_story_id = story[1]
        for class_name in self.class_names:
            for any_story in self.gold_classes[class_name]:
                other_story_id = any_story[1]
                if this_story_id == other_story_id:
                    return class_name
        return 'UNKNOWN'

    def yield_train_and_test_data(self, test_split=0.2):
        assert 0.0 < test_split < 1.0
        x_y_test_length = int(test_split * len(self.stories))
        assert 0 < x_y_test_length < len(self.stories)
        test_stories = self.stories[:x_y_test_length]
        train_stories = self.stories[x_y_test_length:]
        
        x_test = [self.yield_index_list_representation(story) for story in test_stories]
        x_train = [self.yield_index_list_representation(story) for story in train_stories]
        
        y_test = [self.class_names.index(self.yield_gold_class(story)) for story in test_stories]
        y_train = [self.class_names.index(self.yield_gold_class(story)) for story in train_stories]
        
        result = ((x_train, y_train), (x_test, y_test))
        return result
