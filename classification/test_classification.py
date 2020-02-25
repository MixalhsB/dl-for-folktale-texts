import corpus
import classifier
import evaluator
import translate

if __name__ == '__main__':
    languages_lower = ('english', 'german', 'spanish', 'danish', 'italian', 'french', 'polish', 'dutch', 'russian',
                       'hungarian', 'czech')
    
    language = add_translated = exclude_sw_string = binary_mode_string = number_of_runs = ''
    
    print()
    while language.lower() not in languages_lower:
        language = input('Enter language: ')
    # while add_translated.lower() not in ('y', 'n'):
    #         add_translated = input('Add missing tales Google-translated from other languages to the corpus (y/n)? ')
    while exclude_sw_string.lower() not in ('y', 'n'):
        exclude_sw_string = input('Exclude stopwords (y/n)? ')
    while binary_mode_string.lower() not in ('y', 'n'):
        binary_mode_string = input('Binary classification (y) or multi-class (n)? ')
    while not number_of_runs.isdigit() or int(number_of_runs) < 1:
        number_of_runs = input('Enter number of runs: ')
    print()
    
    exclude_stop_words = True if exclude_sw_string == 'y' else False
    binary_mode = True if binary_mode_string == 'y' else False
    
    list_of_corpora = []
    list_of_classifiers = []
    
    for i in range(int(number_of_runs)):
        list_of_corpora.append(corpus.Corpus('../corpora.dict', language.lower().capitalize(), seed=None,
                                             exclude_stop_words=exclude_stop_words, binary_mode=binary_mode))
        list_of_classifiers.append(classifier.Classifier(list_of_corpora[i]))
        
    eval_dumb = evaluator.Evaluator(list_of_corpora, [clsf.dumb_classify for clsf in list_of_classifiers])
    eval_length = evaluator.Evaluator(list_of_corpora, [clsf.length_classify for clsf in list_of_classifiers])
    eval_reuters = evaluator.Evaluator(list_of_corpora, [clsf.simple_reuters_classify for clsf in list_of_classifiers])
    eval_book = evaluator.Evaluator(list_of_corpora, [clsf.book_inspired_classify for clsf in list_of_classifiers])
    eval_doc2vec = evaluator.Evaluator(list_of_corpora, [clsf.doc2vec_classify for clsf in list_of_classifiers])
    eval_ngram = evaluator.Evaluator(list_of_corpora, [clsf.ngram_classify for clsf in list_of_classifiers])
    
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
