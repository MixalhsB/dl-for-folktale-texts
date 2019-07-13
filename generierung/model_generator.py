from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

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


"""
#EXMPL

from keras.utils import to_categorical
from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer

with open("sequence/german_animaltales_sequences.txt") as f:
    doc = f.read()
    lines = doc.split("\n")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)
    vocab_size = len(tokenizer.word_index) + 1
    sequences = array(sequences)
    X, y = sequences[:, :-1], sequences[:, -1]
    y = to_categorical(y, num_classes=vocab_size)
    seq_length = X.shape[1]

neurons1 = 100
neurons2 = 100
neurons3 = 100
activation1 = "relu"
activation2 = "softmax"
model1 = Generator(vocab_size, 50, seq_length, 10, neurons1, neurons2, neurons3, activation1, activation2)
model = model1.create_model()
model.summary()
model_name = "test_model"
print("model created.")
model.fit(X, y, batch_size=50, epochs=10)
model.save(model_name+".h5")
dump(tokenizer, open('test_tokenizer.pkl', 'wb'))
"""







