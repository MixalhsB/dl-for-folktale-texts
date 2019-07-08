# OLD NAME: classifier

import random

class Classifier: 
    def __init__(self, classes): 
        self.classes = classes #Liste! 
    def classify(self, text): 
        return random.choice(self.classes) #hier muessen wir spaeter das neuronale Netz einsetzen 
    def getClasses(self): 
        return self.classes 

