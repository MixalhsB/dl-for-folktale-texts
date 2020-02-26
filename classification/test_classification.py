import corpus
import classifier
import evaluator
import translate
import random


if __name__ == '__main__':
    languages = ('English', 'German', 'Spanish', 'Danish', 'Italian', 'French')
                      # we are *not* considering these langs -> ('Polish', 'Dutch', 'Russian', 'Hungarian', 'Czech')
    
    language_any_cap = add_translated_string = exclude_sw_string = binary_mode_string = number_of_runs = ''
    
    print()
    while language_any_cap.lower().capitalize() not in languages:
        language_any_cap = input('Enter language: ')
    while exclude_sw_string.lower() not in ('y', 'n'):
        exclude_sw_string = input('Exclude stopwords (y/n)? ')
    while binary_mode_string.lower() not in ('y', 'n'):
        binary_mode_string = input('Binary classification (y) or multi-class (n)? ')
    while add_translated_string.lower() not in ('y', 'n'):
        add_translated_string = input('Add missing tales Google-translated from other languages to the corpus (y/n)? ')
    while not number_of_runs.isdigit() or int(number_of_runs) < 1:
        number_of_runs = input('Enter number of runs: ')
    print()
    
    exclude_stop_words = True if exclude_sw_string == 'y' else False
    binary_mode = True if binary_mode_string == 'y' else False
    add_translated = True if add_translated_string == 'y' else False
    
    list_of_corpora = []
    list_of_classifiers = []
    
    for i in range(int(number_of_runs)):
        list_of_corpora.append(corpus.Corpus('../corpora.dict', language_any_cap.lower().capitalize(), seed=None,
                                             exclude_stop_words=exclude_stop_words, binary_mode=binary_mode,
                                             to_be_extended_later=add_translated))
        list_of_classifiers.append(classifier.Classifier(list_of_corpora[i]))
    
    if add_translated:
        for ground_corpus in list_of_corpora:
            empty_dummy_corpus = corpus.Corpus('../corpora.dict', None, dummy_mode=True)
            empty_dummy_corpus.language = ground_corpus.language
            empty_dummy_corpus.train_stories = []
            for other_language in languages:
                if other_language == ground_corpus.language:
                    continue
                other_language_dummy_corpus = corpus.Corpus('../corpora.dict', other_language, dummy_mode=True)
                translate.translate_save_tales(empty_dummy_corpus, other_language_dummy_corpus)
            
            assert len(empty_dummy_corpus.train_stories) >= len(ground_corpus.train_stories)
            random.seed(ground_corpus.seed)
            sample_to_add = random.sample(empty_dummy_corpus.train_stories, len(ground_corpus.train_stories))
            ground_corpus.stories += sample_to_add
            ground_corpus.train_stories += sample_to_add
            random.seed(ground_corpus.seed)
            random.shuffle(ground_corpus.train_stories)
            ground_corpus.w2i_dict = ground_corpus.get_word_to_index_dict()
            ground_corpus.gold_classes = {class_name: stories for class_name, stories in
                                          zip(ground_corpus.class_names, ground_corpus.iter_over_class_specific_subsets)}
            print('\nOKOK:', len(ground_corpus.train_stories), '\n')
        
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
