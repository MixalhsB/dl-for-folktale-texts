#!/usr/bin/python
#-*- coding:utf-8 -*-

import random
from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict
import sys


# load doc into memory (training data sequences)
def load_doc(filename):
    # open the file as read only
    with open(filename, "r") as file:
        # read all text
        text = file.read()
        tales = text.split("\n\n")
        texts = ""
        for item in tales:
            texts += item.split("\n")[-1] + "\n"
        # close the file
        return texts


def average_sentence_length(language, type):
    with open("../average_sentence_length.txt", encoding="utf8") as file:
        dictionary = eval(file.read())
    return dictionary[type][language]


class Generate:

    def __init__(self, model, tokenizer, seq_length, seed_text, n_words):
        self.model = model
        self.tokenizer = tokenizer
        self.len = seq_length
        self.seed = seed_text
        self.words = n_words

    # generate a sequence from a language model
    def generate_seq(self):
        result = list()
        in_text = seed_text
        # generate a fixed number of words
        for _ in range(self.words):
            # encode the text as integer
            encoded = self.tokenizer.texts_to_sequences([in_text])[0]
            # truncate sequences to a fixed length
            encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
            # predict probabilities for each word
            yhat = self.model.predict_classes(encoded, verbose=0)
            # returns index of word with the highest probability

            # map predicted word index to word
            out_word = ''
            for word, index in self.tokenizer.word_index.items():
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
seq_length = average_sentence_length("German", kind)

# load the model
model = load_model("models/"+language+"_"+kind+'_model.h5')

# load the tokenizer
tokenizer = load(open("tokenizer/"+language+"_"+kind+"_tokenizer.pkl", 'rb'))

# select a seed text: random line of text from the input text
# maybe the first line?
seed_text = lines[randint(0, len(lines))]
print(seed_text + '\n')

# generate new text
# how long should it be? -> average length of a tale?

def avg_tale_length(language, type, range_around_avg):
    """
    computes a random number out of the intervall average-tale-length-range_around_avg and +range_around_avg
    :param language: string
    :param type: string
    :param range_around_avg: int
    :return: random number between average talelength +/- range given
    """
    with open("../average_tale_length.txt", encoding = "utf8") as file:
        s = file.readline()
        dictionary = eval(s.replace("<class 'int'>", 'int'))

    avg = dictionary[language+"_"+type]
    return random.randrange(avg-range_around_avg, avg+range_around_avg)

def min_max_random(language, type):
    with open("../min_max_tale_length.txt", encoding = "utf8") as file:
        s = file.read()
        dictionary = eval(s.replace("inf", "0"))
        min = dictionary[type][language][0]
        max = dictionary[type][language][1]
        return random.randrange(min, max)

# generated = generate_seq(model, tokenizer, seq_length, seed_text, avg_tale_length(language, kind,100))
generated = Generate(model, tokenizer, seq_length, seed_text, avg_tale_length("German", kind, min_max_random("German", kind)))
# generated = generate_seq(model, tokenizer, seq_length, seed_text, min_max_random(language, kind))
print(generated.generate_seq())

