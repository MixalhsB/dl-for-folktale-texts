# OLD NAME: usemodels_version1.1

from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


# load doc into memory (training data sequences)
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
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
language = input("Enter a language: ")
kind = input("Please choose one of the following: \n animaltales \n magictales \n religioustales \n realistictales \n stupidogre \n jokes \n formulatales")
in_filename = "sequence/" + language.lower() + "_" + kind.lower() + "_sequences.txt"
doc = load_doc(in_filename)
lines = doc.split('\n')

#minus output word
#input of the model has to be as long as seq_length
seq_length = len(lines[0].split()) - 1

# load the model
model = load_model('models/model' + in_filename +'.h5')

# load the tokenizer
tokenizer = load(open('tokenizer/tokenizer' + in_filename + '.pkl', 'rb'))

# select a seed text: random line of text from the input text
# maybe the first line?
seed_text = lines[randint(0,len(lines))]
print(seed_text + '\n')

# generate new text
# how long should it be? -> average length of a tale?
generated = generate_seq(model, tokenizer, seq_length, seed_text, 50)
print(generated)
