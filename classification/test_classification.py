import corpus
import classifier
import evaluator
import train_classification_model

if __name__ == '__main__':
    corpus = corpus.Corpus('..\\corpora.txt', 'English', seed=123)
    clsf = classifier.Classifier(corpus)
    eval_dumb = evaluator.Evaluator(corpus, clsf.dumb_classify)
    eval_reuters = evaluator.Evaluator(corpus, clsf.simple_reuters_classify)
    # eval_book = evaluator.Evaluator(corpus, clsf.book_inspired_classify)

    resulting_string = '\nDUMB:\n'
    resulting_string += eval_dumb.evaluate() + '\n'

    resulting_string += ('\nREUTERS:\n')
    resulting_string += eval_reuters.evaluate() + '\n'

    # resulting_string += ('\nBOOK-RELATED AFFAIRS:')
    # resulting_string += eval_book.evaluate() + '\n'

    print(resulting_string)

    print('\nBOOK:')
    train_classification_model.main()

