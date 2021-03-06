from sklearn import metrics


class Evaluator:
    def __init__(self, list_of_corpora, list_of_classify_instances):
        self.list_of_corpora = list_of_corpora
        self.list_of_classify_instances = list_of_classify_instances
        assert len(self.list_of_corpora) == len(self.list_of_classify_instances)

    def evaluate(self):
        predicted_class_names, true_class_names = [], []
        for i, (current_corpus, current_classify_instance) in enumerate(zip(self.list_of_corpora, self.list_of_classify_instances)):
            print('\nRUN NUMBER ' + str(i + 1) + ' ...\n')
            predicted_class_names += current_classify_instance([story[4] for story in current_corpus.test_stories])
            true_class_names += [current_corpus.get_gold_class_name(story) for story in current_corpus.test_stories]
            ### possibly prevent memory leaks:
            current_corpus.avg_story_lengths = None
            current_corpus.simple_reuters_model = None
            current_corpus.book_model_data = (None, None, None, None, None)
            current_corpus.doc2vec_model_data = (None, None, None)
            current_corpus.ngram_model_data = (None, None, None)
            ###
        print('\nTrue class names:\n', true_class_names, '\n')
        print('\nPredicted class names:\n', predicted_class_names, '\n')

        result = str(metrics.classification_report(true_class_names, predicted_class_names, digits=3))
        result += '\n' + str(metrics.confusion_matrix(true_class_names, predicted_class_names))
        return result
