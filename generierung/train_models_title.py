# OLD NAME: trainmodels_version1.1

from numpy import array
from pickle import dump
from collections import defaultdict
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# define the model
def define_model(vocab_size, seq_length):
    model = Sequential()
    #size of the embedding vector space = 50 (100,300)
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    #1 LSTM hidden layers with 100 memory cells each
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    #extract features from the sequence
    #dense layer interpret those features
    model.add(Dense(100, activation='relu'))
    #outputlayer predicts next word as a single vector
    #softmax function -> probability distribution
    model.add(Dense(vocab_size, activation='softmax'))
    # compile network
    #crossentropy for multiclass classification problem
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize defined model
    model.summary()
    #plot_model(model, to_file='model.png', show_shapes=True)
    return model

def trainmodel(doc, model_name):
    lines = doc.split("\n")

    # removes empty lines
    lines = list(filter(None, lines))

    # integer encode sequences of words
    tokenizer = Tokenizer() #create Tokenizer for encoding
    tokenizer.fit_on_texts(lines)
    #train it on the data -> it finds all unique words and assigns each an integer
    sequences = tokenizer.texts_to_sequences(lines)
    # pad sequences to the same length
    sequences = pad_sequences(sequences, padding="post")
    #make a list of integer out of each list of words

    # vocabulary size
    vocab_size = len(tokenizer.word_index) + 1

    #mapping of words -> integers is a dictionary attribute called word_index
    #values from 1 to total number of words, so we need to +1

    # separate into input and output
    sequences = array(sequences)
    print(sequences.shape)
    # print("sequences: ", sequences)
    # print(sequences[:, :-1])
    # print(sequences[:, -1])
    X, y = sequences[:,:-1], sequences[:,-1]
    #one hot encode output word -> vector with lots of 0 and a 1 for the word itself
    y = to_categorical(y, num_classes=vocab_size)
    seq_length = X.shape[1] #for the Embedding Layer
    #2nd dimension (columns)

    # define model
    model = define_model(vocab_size, seq_length)
    # fit model
    model.fit(X, y, batch_size=128, epochs=100)
    # save the model to file
    # model_name = in_filename.split("/")[-1]
    # model_name = model_name[:-4] # txt Endung
    # model_name = model_name.replace("sequences", "model")
    # print(model_name)
    model.save('models/' + model_name + '.h5')
    # save the tokenizer
    #we need the mapping from words to integers when we load the model
    #we can save it with Pickle
    model_name = model_name.replace("model", "tokenizer")
    dump(tokenizer, open('tokenizer/' + model_name + '.pkl', 'wb'))

language = 'Spanish'

categories = ["animaltales", "magictales", "religioustales", "realistictales", "stupidogre", "jokes", "formulatales"]
titles = defaultdict(str)
for item in categories:
    titles[language] += load_doc("sequence/" + language + "_" + item + "_sequences_title.txt") + '\n'
    # print(item)
    # print()
    # print(titles[language])

# print(titles[language])
model_name = language + '_title_model'
trainmodel(titles[language], model_name)

# trainmodel("sequence/" + language + '_' +'animaltales_sequences.txt')
# trainmodel("sequence/" + language + '_'+'magictales_sequences.txt') # MemoryError
##
# trainmodel("sequence/" + language + '_'+'religioustales_sequences_title.txt')
##
# trainmodel("sequence/" + language + '_'+'realistictales_sequences_title.txt')
##trainmodel("sequence/" + language + '_'+'stupidogre_sequences.txt')
##
##trainmodel("sequence/" + language + '_'+'jokes_sequences.txt')
##
##trainmodel("sequence/" + language + '_'+'formulatales_sequencDanishes.txt')
