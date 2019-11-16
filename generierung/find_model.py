from keras.preprocessing.text import Tokenizer
from numpy import array
from keras.utils import to_categorical
from time_limit import *
from pickle import dump
from random import randint
import numpy as np
from model_selector import Selector

def load_doc(filename):
    # open the file as read only
    with open(filename, "r") as file:
        # read all text
        text = file.read()
        # close the file
        return text

input_data = "sequence/german_realistictales_sequences.txt"

with open(input_data) as f:
    doc = f.read()
    lines = []
    stories = doc.split("\n\n")
    # print(stories)
    # letztes Element der Sequenzen ist '', deswegen :-1
    for item in stories[:-1]:
        story = item.split("\n")
        lines.append(story[1])

    # integer encode sequences of words
    tokenizer = Tokenizer()  # create Tokenizer for encoding
    tokenizer.fit_on_texts(lines)
    # train it on the data -> it finds all unique words and assigns each an integer
    sequences = tokenizer.texts_to_sequences(lines)
    # make a list of integer out of each list of words
    vocab_size = len(tokenizer.word_index) + 1
    sequences = array(sequences)
    X, y = sequences[:, :-1], sequences[:, -1]
    y = to_categorical(y, num_classes=vocab_size)
    seq_length = X.shape[1]
dump(tokenizer, open("tokenizer/german_realistictales_tokenizer.pkl", "wb"))

seed_text = lines[randint(0, len(lines))]

parameters = {"epochs": np.arange(10, 100), "LSTM_neurons1": np.arange(1, 200), "LSTM_neurons2": np.arange(1, 200),
              "Dense_neurons": np.arange(1, 200), "activation1": ["relu", "tanh", "elu"],
              "activation2": ["softmax", "hard_sigmoid"]}

test_models = Selector(vocab_size, seed_text, seq_length, parameters)

print("Start model search...")
model_counter = 0
for _ in time_limit(hours=1):
    #sucht in angegebenen Zeitraum nach models
    model_counter += 1
    loss, acc, Model = test_models.search(X, y)
    #test_models.as_df().head(10).to_csv('test_models.csv')
    #speichert models mit ihrer acc und loss in einer Tabelle ab
    with open("german_realistictales_models.list", "a") as f:
        f.write("model No."+str(model_counter)+":loss: "+str(loss)+";accuracy: "+str(acc)+";Parameters: "+str(Model)+"\n")
        print("model No." + str(model_counter) + ":\nloss:", loss, "accuracy:", acc, "\nParameters: ", Model)

model = test_models.best_model()
#print(seed_text + generate_seq(model, tokenizer, seq_length, seed_text, 150))
model.save("models/german_realistictales_model.h5")
