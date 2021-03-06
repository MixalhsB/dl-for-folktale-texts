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
    file = open(filename, 'r', encoding="UTF8")
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


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
        #in_text = seed_text
        in_text = self.seed
        # generate a fixed number of words
        for _ in range(self.words):
            # encode the text as integer
            encoded = self.tokenizer.texts_to_sequences([in_text])[0]
            # truncate sequences to a fixed length
            # encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
            encoded = pad_sequences([encoded], maxlen=self.len, truncating='pre')
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


# for interaction with the user in the console
while True:
    l = input("Enter a language: [E]nglish   [G]erman   [F]rench    [S]panish")
    if l == "E":
        language = "English"
    elif l == "G":
        language = "German"
    elif l == "F":
        language = "French"
    elif l == "S":
        language = "Spanish"
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

# load sequences
in_filename = "sequence/" + language.lower() + "_" + kind.lower() + "_sequences.txt"
in_filename_title = "sequence/" + language.lower() + "_" + kind.lower() + "_sequences_title.txt"
doc = load_doc(in_filename)
doc_title = load_doc(in_filename_title)

lines = doc.split('\n')
lines_title = doc_title.split('\n')


# determine sequence length
seq_length = len(lines[0].split()) - 1 # wieso wurde diese zeile gelöscht und zum spezielleren Fall abgeändert? (Zeile 131)
seq_length_title = 5 #len(lines_title[0].split()) - 1 model trained on set sequence length 5


# load the models
model = load_model("models/"+language+"_"+kind+'_model.h5')
title_model = load_model("models/"+language + "_title_model.h5")

# load the tokenizers
tokenizer = load(open("tokenizer/"+language+"_"+kind+"_tokenizer.pkl", 'rb'))
title_tokenizer = load(open("tokenizer/"+language+"_title_tokenizer.pkl", 'rb'))

# select a seed text: random line of text from the input text
seed_text_title = lines_title[randint(0, len(lines_title))]
seed_text = lines[randint(0, len(lines))]


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

# generate new text
generated_title = Generate(title_model, title_tokenizer, seq_length_title , seed_text_title, 5)
print("Title of the generated tale: ", generated_title.generate_seq())
generated_text = Generate(model, tokenizer, seq_length, seed_text, avg_tale_length(language, kind, min_max_random(language, kind)))
print("Text of the generated tale: ", generated_text.generate_seq())

