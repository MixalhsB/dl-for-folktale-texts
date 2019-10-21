import corpus
import classifier
import evaluator


if __name__ == '__main__':
    corpus = corpus.Corpus('..\\..\\corpora.txt', 'English')
    clsf = classifier.Classifier(corpus)
    eval_dumb = evaluator.Evaluator(corpus, clsf.dumb_classify)
    eval_reuters = evaluator.Evaluator(corpus, clsf.simple_reuters_classify)

    print('DUMB:')
    eval_dumb.evaluate()

    print('REUTERS:')
    eval_reuters.evaluate()
