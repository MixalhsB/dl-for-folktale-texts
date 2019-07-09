# OLD NAME: classifier

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import random

class Classifier: 
    def __init__(self, class_names, train_and_test_data): 
        self.class_names = class_names
        
        (x_train, y_train), (x_test, y_test) = train_and_test_data
        tokenizer = Tokenizer(num_words=10000)
        self.x_train_matrix = tokenizer.sequences_to_matrix(x_train, mode='binary')
        self.x_test_matrix = tokenizer.sequences_to_matrix(x_test, mode='binary')
        self.y_train_matrix = to_categorical(y_train, len(self.class_names))
        self.y_test_matrix = to_categorical(y_test, len(self.class_names))
    
    def dumb_classify(self, text): 
        return random.choice(self.class_names)
    
    def okay_classify(self, text):
        #TODO
        pass
