from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from prepare_data_for_classification import *


#Nach Kapitel 15 in Deep Learning for NLP

def load_doc(file):
    with open(file) as f:
        document = f.read()
    return document

# load an already cleaned dataset
def load_clean_dataset(language):
    # language ist zB so angegeben "English"
    
    # load documents
    # UNKNOWN wird hier gerade auch als Label angesehen
    animal = load_doc(language + "_animaltales_cleaned.txt")
    magic = load_doc(language + "_magictales_cleaned.txt")
    religious = load_doc(language + "_religioustales_cleaned.txt")
    realistic = load_doc(language + "_realistictales_cleaned.txt")
    ogre = load_doc(language + "_stupidogre_cleaned.txt")
    jokes = load_doc(language + "_jokes_cleaned.txt")
    formula = load_doc(language + "_formulatales_cleaned.txt")
    UNKNOWN = load_doc(language + "_unknowntexts_cleaned.txt")
    
    docs = animal + magic + religious + realistic + ogre + jokes + formula + unknown
    # prepare labels
    ## Labels:
    # animal    --> 0
    # magic     --> 1
    # religious --> 2
    # realistic --> 3
    # ogre      --> 4
    # jokes     --> 5
    # formula   --> 6
    # UNKNOWN   --> 7

    # im Buch gibt es nur zwei Labels
    labels = array([0 for _ in range(len(animal))] + [1 for _ in range(len(magic))]+ [2 for _ in range(len(religious))]
                   + [3 for _ in range(len(realistic))] + [4 for _ in range(len(ogre))] +  [5 for _ in range(len(jokes))]
                   + [6 for _ in range(len(formula))] + [7 for _ in range(len(UNKNOWN))])
    return docs, labels


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# integer encode and pad documents
def encode_docs(tokenizer, max_length, docs):
    # integer encode
    encoded = tokenizer.texts_to_sequences(docs)
    # pad sequences
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded

# define the model
def define_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    
    # Output Layer erhält 8 Nodes, für die 8 Label
    model.add(Dense(8, activation='sigmoid'))
    
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize defined model
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

# load the vocabulary
vocab_filename = "German" + '_vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())

# load training data
train_docs, ytrain = load_clean_dataset("German")

# create the tokenizer
tokenizer = create_tokenizer(train_docs)

# define vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size)

# calculate the maximum sequence length
max_length = max([len(s.split()) for s in train_docs])
print('Maximum length: %d' % max_length)

# encode data
Xtrain = encode_docs(tokenizer, max_length, train_docs)

# define model
model = define_model(vocab_size, max_length)

# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)

# save the model
model.save('model.h5')
