# OLD NAME: evaluator

from sklearn import metrics


class Evaluator:
    def __init__(self, corpus, classify):
        self.corpus = corpus
        self.classify = classify

    def evaluate(self):
        predicted = []
        true = []
        for story in self.corpus.test_stories:
            predicted_class_name = self.classify(story[4])
            true_class_name = self.corpus.get_gold_class_name(story)
            predicted.append(predicted_class_name)
            true.append(true_class_name)
        result = str(metrics.classification_report(true, predicted, digits=3))
        result += str(metrics.confusion_matrix(true, predicted))
        return result
