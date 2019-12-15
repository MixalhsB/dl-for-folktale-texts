import corpus
import classifier
import evaluator

if __name__ == '__main__':
    corpus = corpus.Corpus('../corpora.dict', 'English', seed=None, exclude_stop_words=True, binary_mode=True)
    clsf = classifier.Classifier(corpus)
    eval_dumb = evaluator.Evaluator(corpus, clsf.dumb_classify)
    eval_length = evaluator.Evaluator(corpus, clsf.length_classify)
    eval_reuters = evaluator.Evaluator(corpus, clsf.simple_reuters_classify)
    eval_book = evaluator.Evaluator(corpus, clsf.book_inspired_classify)
    eval_doc2vec = evaluator.Evaluator(corpus, clsf.doc2vec_classify)
    eval_ngram = evaluator.Evaluator(corpus, clsf.ngram_classify)

    resulting_string = '\nDUMB:\n'
    resulting_string += eval_dumb.evaluate() + '\n'
    print('\nFinished computing DUMB classification.\n')

    resulting_string += '\nLENGTH:\n'
    resulting_string += eval_length.evaluate() + '\n'
    print('\nFinished computing LENGTH classification.\n')

    resulting_string += '\nREUTERS:\n'
    resulting_string += eval_reuters.evaluate() + '\n'
    print('\nFinished computing REUTERS classification.\n')

    resulting_string += '\nBOOK:\n'
    resulting_string += eval_book.evaluate() + '\n'
    print('\nFinished computing BOOK classification.\n')

    resulting_string += '\nDOC2VEC:\n'
    resulting_string += eval_doc2vec.evaluate() + '\n'
    print('\nFinished computing DOC2VEC classification.\n')

    resulting_string += '\nN-GRAM:\n'
    resulting_string += eval_ngram.evaluate()
    print('\nFinished computing N-GRAM classification.\n')

    print(resulting_string)
