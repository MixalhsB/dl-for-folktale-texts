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
    # while not number_of_runs.isdigit() or int(number_of_runs) < 1:
    #     number_of_runs = input('Enter number of runs: ')
    print()
    
    exclude_stop_words = True if exclude_sw_string == 'y' else False
    binary_mode = True if binary_mode_string == 'y' else False
    
    ground_corpus = corpus.Corpus('../corpora.dict', language.lower().capitalize(), seed=None,
                                  exclude_stop_words=exclude_stop_words, binary_mode=binary_mode)
    
    '''
    if add_translated == 'y':
        for other_lang in (lang for lang in languages_lower if lang != language):
            other_corpus = corpus.Corpus('../corpora.dict', other_lang.capitalize())
            translate.extract_translate_unique_tales(ground_corpus, other_corpus)
        ground_corpus.shuffle_stories_and_split_them()
        ground_corpus.w2i_dict = ground_corpus.get_word_to_index_dict()
    '''
    
    clsf = classifier.Classifier(ground_corpus)
    eval_dumb = evaluator.Evaluator(ground_corpus, clsf.dumb_classify)
    eval_length = evaluator.Evaluator(ground_corpus, clsf.length_classify)
    eval_reuters = evaluator.Evaluator(ground_corpus, clsf.simple_reuters_classify)
    eval_book = evaluator.Evaluator(ground_corpus, clsf.book_inspired_classify)
    eval_doc2vec = evaluator.Evaluator(ground_corpus, clsf.doc2vec_classify)
    eval_ngram = evaluator.Evaluator(ground_corpus, clsf.ngram_classify)

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
