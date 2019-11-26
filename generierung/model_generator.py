from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.utils import to_categorical
from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer

class Generator:
    """
    Generiert ein Modell mit den angegebenen Parametern:
    *vocab_size* Größe des Vokabulars
    *seq* Sequenz (fürs Embedding)
    *seq_length* Sequenzlänge
    *epochs* Epochen fürs Training
    *LSTM_neurons1/LSTM_neurons2* Anzahl Knoten in der 1. bzw. 2. LSTM-Layer
    *Dense_neurons* Anzahl Knoten in Dense-Layer
    *activation1/activation2* activation functions für die Dense-Layer
    """
    def __init__(self, vocab_size, seq, seq_length, epochs, LSTM_neurons1, LSTM_neurons2, Dense_neurons,
                 activation1, activation2):
        self.parameters = {"epochs": epochs, "LSTM_neurons1" : LSTM_neurons1, "LSTM_neurons2" : LSTM_neurons2,
                           "Dense_neurons" : Dense_neurons, "activation1" : activation1,
                           "activation2" : activation2}
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, seq, input_length=seq_length))
        self.model.add(LSTM(self.parameters["LSTM_neurons1"], return_sequences=True))
        self.model.add(LSTM(self.parameters["LSTM_neurons2"]))
        self. model.add(Dense(self.parameters["Dense_neurons"], activation=self.parameters["activation1"]))
        self. model.add(Dense(vocab_size, activation=self.parameters["activation2"]))
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        #self.model.summary()

    def __str__(self):
        return str(self.parameters.items())

    def create_model(self):
        return self.model


def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


"""
input_data = "sequence/German_animaltales_sequences.txt"

with open(input_data) as f:
    doc = load_doc(input_data)
    lines = doc.split('\n')
    # print(stories)
    # letztes Element der Sequenzen ist '', deswegen :-1
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

neurons1 = 64
neurons2 = 147
neurons3 = 91
activation1 = "elu"
activation2 = "softmax"
model1 = Generator(vocab_size, 50, seq_length, 85, neurons1, neurons2, neurons3, activation1, activation2)
model = model1.create_model()
model.summary()
model_name = "models/german_animaltales_model"
print("model created.")
model.fit(X, y, batch_size=128, epochs=85)
model.save(model_name+".h5")
dump(tokenizer, open("tokenizer/german_animaltales_tokenizer.pkl", "wb"))
"""
