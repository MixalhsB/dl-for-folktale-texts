import corpus
import classifier
import evaluator


if __name__ == '__main__':
    corpus = corpus.Corpus('../corpora.dict', 'German', seed=123)
    clsf = classifier.Classifier(corpus)
    eval_dumb = evaluator.Evaluator(corpus, clsf.dumb_classify)
    eval_reuters = evaluator.Evaluator(corpus, clsf.simple_reuters_classify)
    eval_book = evaluator.Evaluator(corpus, clsf.book_inspired_classify)
    eval_length = evaluator.Evaluator(corpus, clsf.length_classify)

    resulting_string = '\nDUMB:\n'
    resulting_string += eval_dumb.evaluate() + '\n'

    resulting_string += ('\nREUTERS:\n')
    resulting_string += eval_reuters.evaluate() + '\n'

    resulting_string += ('\nBOOK:\n')
    resulting_string += eval_book.evaluate() + '\n'
    
    resulting_string += ('\nLENGTH:\n')
    resulting_string += eval_length.evaluate()

    print(resulting_string)


