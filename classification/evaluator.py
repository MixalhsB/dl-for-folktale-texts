# OLD NAME: evaluator

from collections import defaultdict
from corpus import Corpus
from classifier import Classifier

class Evaluator:
    def __init__(self, corpus, classifier):
        self.corpus = corpus
        self.classifier = classifier

    def evaluate(self):
        numberOfStories = 0
        correctlyClassified = 0
        assignedClasses = defaultdict(list)  # Dictionary mit Klassen als key und Liste von Geschichten als Value
        for story in self.corpus:  # benutzt __iter__ in Klasse Corpus
            numberOfStories += 1
            category = self.classifier.simple_reuters_classify(story[4])  # waehle Kategorie
            assignedClasses[category].append(story)
        for (className, stories) in assignedClasses.items():
            for story in stories:
                if story in self.corpus.gold_classes[className]:
                    correctlyClassified += 1
        return correctlyClassified / numberOfStories
