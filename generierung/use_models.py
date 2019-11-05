#!/usr/bin/python
#-*- coding:utf-8 -*-

import random
from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict


# load doc into memory (training data sequences)
def load_doc(filename):
    # open the file as read only
    with open(filename, "r") as file:
        # read all text
        text = file.read()
        # close the file
        return text


# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        #returns index of word with the highest probability
        
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)


# load cleaned text sequences
while True:
    l = input("Enter a language: [E]nglish   [G]erman")
    if l == "E":
        language = "english"
    elif l == "G":
        language = "german"
    else:
        print("Wrong parameter specification!\nPlease try again.")
        continue
    while True:
        k = input("Please choose one of the following: [A]nimaltales [M]agictales    [R]eligioustales    "
                     "[r]ealistictales  [S]tupidogre    [J]okes [F]ormulatales  [Z]urück")
        if k == "A":
            kind = "animaltales"
            break
        elif k == "M":
            kind = "magictales"
            break
        elif k == "R":
            kind = "religioustales"
            break
        elif k == "r":
            kind = "realistictales"
            break
        elif k == "S":
            kind = "stupidogre"
            break
        elif k == "J":
            kind = "jokes"
            break
        elif k == "F":
            kind = "formulatales"
            break
        elif k == "Z":
            break
        else:
            print("Wrong parameter specification!\nPlease try again.")
    if k == "Z":
        continue
    else:
        break


in_filename = "sequence/" + language.lower() + "_" + kind.lower() + "_sequences.txt"
doc = load_doc(in_filename)
lines = doc.split('\n')

#minus output word
#input of the model has to be as long as seq_length
seq_length = len(lines[0].split()) - 1

# load the model
model_name = "german_animaltales_model.h5"
model = load_model("models/"+ model_name)

# load the tokenizer
tokenizer_name = "german_animaltales_tokenizer.pkl"
tokenizer = load(open("tokenizer/"+ tokenizer_name, 'rb'))

# select a seed text: random line of text from the input text
# maybe the first line?
seed_text = lines[randint(0, len(lines))]
print(seed_text + '\n')

# generate new text
# how long should it be? -> average length of a tale?

#TODO zusaetzlich zur durchschnittlicher satzlaenge, min und max speichern und dann eine random zahl dazwischen nehmen
def avg_tale_length(language, type, range_around_avg):
    """
    computes a random number out of the intervall average-tale-length-range_around_avg and +range_around_avg
    :param language: string
    :param type: string
    :param range_around_avg: int
    :return: random number between average talelength +/- range given
    """
    with open("average_tale_length.txt", encoding = "utf8") as file:
        s = file.readline()
        dictionary = eval(s.replace("<class 'int'>", 'int'))

    avg = dictionary[language+"_"+kind]
    return random.randrange(avg-range_around_avg, avg+range_around_avg)

generated = generate_seq(model, tokenizer, seq_length, seed_text, avg_tale_length(language, kind,100))
print(generated)
