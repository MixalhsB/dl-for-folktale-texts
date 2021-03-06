from random import randint
from pickle import load
import random
from keras.models import load_model
from model_generator import Generator
from classifier import Classifier
from corpus import Corpus
from keras.preprocessing.sequence import pad_sequences

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

def load_doc(filename):
    # open the file as read only
    with open(filename, "r") as file:
        # read all text
        text = file.read()
        # close the file
        return text

def average_sentence_length(language, type):
    with open("../average_sentence_length.txt", encoding="utf8") as file:
        dictionary = eval(file.read())
    return dictionary[type][language.lower()]

def min_max_random(language, type):
    with open("../min_max_tale_length.txt", encoding = "utf8") as file:
        s = file.read()
        dictionary = eval(s.replace("inf", "0"))
        min = dictionary[type][language.capitalize()][0]
        max = dictionary[type][language.capitalize()][1]
        return random.randrange(min, max)

class Eval:

    def __init__(self, model, tokenizer, sequence_file, classifier, language, kind):
        self.model = load_model(model)
        self.language = language
        self.kind = kind
        self.tokenizer = load(open(tokenizer, "rb"))
        self.sequence = sequence_file
        self.classifier = classifier
        self.right_dumb = 0
        self.right_reu = 0

    def evaluate(self):
        #wieviele tales generieren und klassifizieren?
        with open("eval.txt", "w+", encoding="utf-8") as f:
            for i in range(0, 100):
                tale = ""
                doc = load_doc(self.sequence)
                lines = doc.split('\n')
                seq_length = average_sentence_length(self.language, self.kind)
                seed_text = lines[randint(0, len(lines))]
                tale += seed_text
                generated = Generate(self.model, self.tokenizer, seq_length, seed_text, min_max_random(self.language, self.kind))
                tale += generated.generate_seq()
                #klassifizieren
                if self.classifier.dumb_classify(tale) == "non-magic":
                    self.right_dumb += 1
                if self.classifier.simple_reuters_classify(tale) == "non-magic":
                    self.right_reu += 1
            p = (self.right_dumb/10)*100
            f.write(str(p)+"% der "+str(self.language)+" "+str(self.kind)+ "wurden vom dumb classifier korrekt als "
                                                                           "non-magic tales klassifiziert.")
            q = 100-p
            f.write(str(q)+"% der "+str(self.language)+" "+str(self.kind)+" wurden vom dumb classifier falsch als "
                                                                          "magic tales klassifiziert.")
            p = (self.right_reu/10)*100
            f.write(str(p)+"% der "+str(self.language)+" "+str(self.kind)+
                  " wurden vom reuters classifier korrekt als non-magic tales klassifiziert.")
            q = 100 - p
            f.write(str(q)+"% der "+str(self.language)+" "+str(self.kind)+
                  " wurden vom reuters classifier falsch als magic tales klassifiziert.")


languages = ["german"]
kinds = ["animaltales", "religioustales", "realistictales", "stupidogre", "formulatales", "jokes"]
corpus = Corpus('../corpora.dict', 'German', seed=123, binary_mode=True)
classifier = Classifier(corpus)
for language in languages:
    for kind in kinds:
        model = "models/"+language+"_"+kind+"_model.h5"
        tokenizer = "tokenizer/"+language+"_"+kind+"_tokenizer.pkl"
        sequence_file = "sequence/"+language+"_"+kind+"_sequences.txt"
        print("Evaluate "+language+" "+kind)
        evalu = Eval(model, tokenizer, sequence_file, classifier, language, kind)
        evalu.evaluate()

