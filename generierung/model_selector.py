import numpy as np
from model_generator import Generator
import pandas as pd

class Selector:
    """
    Erstellt mit der Generator-Klasse Modelle mit zufälligen Parametern und rankt diese nach accuracy
    """
    def __init__(self, vocab_size, seq, seq_length, parameters):
        self.params = parameters
        self.vocab_size = vocab_size
        self.seq = seq
        self.seq_length = seq_length
        self.acc = []
        self.loss = []
        self.models = []

    def search(self, x, y):
        selected = dict()
        for name, options in self.params.items():
            selected[name] = np.random.choice(options)
        Model = Generator(self.vocab_size, 50, self.seq_length, **selected)
        model = Model.create_model()
        model.fit(x, y, batch_size=128, epochs=selected["epochs"], verbose=0)
        self.models.append(model)
        loss, acc = model.evaluate(x, y, verbose=0)
        self.acc.append(acc)
        self.loss.append(loss)
        return loss, acc, Model

    def as_df(self):
        return pd.DataFrame({'model': self.models, 'accuracy': self.acc,
                             'loss': self.loss}).sort_values(by=['accuracy'], ascending=False)

    def best_models(self, n):
        return self.as_df().head(n)['model'].values[0]

    def best_model(self):
        return self.best_models(1)