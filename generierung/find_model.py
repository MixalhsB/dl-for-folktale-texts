from keras.preprocessing.text import Tokenizer
from numpy import array
from keras.utils import to_categorical
from time_limit import *
from pickle import dump
from random import randint
import numpy as np
from selector import Selector

def load_doc(filename):
    # open the file as read only
    with open(filename, "r") as file:
        # read all text
        text = file.read()
        # close the file
        return text

in_filename = "sequence/german_animaltales_sequences.txt"
doc = load_doc(in_filename)
lines = doc.split("\n")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
vocab_size = len(tokenizer.word_index) + 1
sequences = array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical(y, num_classes=vocab_size)
dump(tokenizer, open("tokenizer.pkl", "wb"))

seed_text = lines[randint(0, len(lines))]
seq_length = len(lines[0].split()) - 1

parameters = {"epochs": np.arange(10, 100), "LSTM_neurons1": np.arange(1, 200), "LSTM_neurons2": np.arange(1, 200),
              "Dense_neurons": np.arange(1, 200), "activation1": ["relu", "tanh", "elu"],
              "activation2": ["softmax", "hard_sigmoid"]}

test_models = Selector(vocab_size, seed_text, seq_length, parameters)

print("Start model search...")
model_counter = 0
for _ in time_limit(hours=2):
    #sucht in angegebenen Zeitraum nach models
    model_counter += 1
    loss, acc, Model = test_models.search(X, y)
    #test_models.as_df().head(10).to_csv('test_models.csv')
    #speichert models mit ihrer acc und loss in einer Tabelle ab
    print("model No."+str(model_counter)+":\nloss:", loss, "accuracy:", acc, "\nParameters: ", Model)

model = test_models.best_model()
#print(seed_text + generate_seq(model, tokenizer, seq_length, seed_text, 150))
model.save("found_model.h5")
