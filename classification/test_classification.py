from itertools import tee
import corpus
import classifier
import evaluator
import translate
import random
import sys


if __name__ == '__main__':
    languages = ('English', 'German', 'Spanish', 'Danish', 'Italian', 'French')
                      # we are *not* considering these langs -> ('Polish', 'Dutch', 'Russian', 'Hungarian', 'Czech')
    
    language_any_cap = add_translated_string = exclude_sw_string = binary_mode_string = number_of_runs_string = ''
    
    if len(sys.argv) == 1:
        print()
        while language_any_cap.lower().capitalize() not in languages:
            language_any_cap = input('Enter language: ')
        while exclude_sw_string.lower() not in ('y', 'n'):
            exclude_sw_string = input('Exclude stopwords (y/n)? ')
        while binary_mode_string.lower() not in ('y', 'n'):
            binary_mode_string = input('Binary classification (y) or multi-class (n)? ')
        while add_translated_string.lower() not in ('y', 'n'):
            add_translated_string = input('Extend training data by tales Google-translated from other languages (y/n)? ')
        while not number_of_runs_string.isdigit() or int(number_of_runs_string) < 1:
            number_of_runs_string = input('Enter number of runs: ')
        print()
    elif len(sys.argv) == 2:
        try:
            language_must_be_lower, exclude_sw_string, binary_mode_string, add_translated_string, \
                                    number_of_runs_string = sys.argv[1].split('_')
        except ValueError:
            assert False
        assert language_must_be_lower.islower() and language_must_be_lower.capitalize() in languages
        language_any_cap = language_must_be_lower
        assert exclude_sw_string in ('y', 'n')
        assert binary_mode_string in ('y', 'n')
        assert add_translated_string in ('y', 'n')
        assert number_of_runs_string.isdigit() and int(number_of_runs_string) >= 1
    else:
        assert False
    
    exclude_stop_words = True if exclude_sw_string == 'y' else False
    binary_mode = True if binary_mode_string == 'y' else False
    add_translated = True if add_translated_string == 'y' else False
    number_of_runs = int(number_of_runs_string)
    
    list_of_corpora = []
    list_of_classifiers = []
    
    print('\nInitializing training and test corpora ...')
    
    for i in range(number_of_runs):
        list_of_corpora.append(corpus.Corpus('../corpora.dict', language_any_cap.lower().capitalize(), seed=i,
                                             exclude_stop_words=exclude_stop_words, binary_mode=binary_mode,
                                             to_be_extended_later=add_translated))
        ground_corpus = list_of_corpora[i]
        
        def _create_temporary_gold_classes():
            iter_copy, ground_corpus.iter_over_class_specific_subsets = tee(ground_corpus.iter_over_class_specific_subsets)
            ground_corpus.gold_classes = {class_name: stories for class_name, stories in
                                          zip(ground_corpus.class_names, iter_copy)}
        
        def _delete_temporary_gold_classes():
            ground_corpus.gold_classes = None
        
        if ground_corpus.to_be_extended_later:
            _create_temporary_gold_classes()
        while {ground_corpus.get_gold_class_name(st) for st in ground_corpus.stories} \
              != {ground_corpus.get_gold_class_name(st) for st in ground_corpus.train_stories}:
            ground_corpus.seed *= 1.5
            ground_corpus.shuffle_stories_and_split_them()
            print('\n' + str(i) + ': Just had to re-shuffle train-test split due to sparse training data ...\n')
            _create_temporary_gold_classes()
        if ground_corpus.to_be_extended_later:
            _delete_temporary_gold_classes()
        
        list_of_classifiers.append(classifier.Classifier(ground_corpus))
    
    if add_translated:
        for i, ground_corpus in enumerate(list_of_corpora):
            print('Extending by translated tales in progress:', i + 1, '/', len(list_of_corpora))
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
    
    output_filename = language_any_cap.lower() + '_' + exclude_sw_string + '_' + binary_mode_string + '_'
    output_filename += add_translated_string + '_' + number_of_runs_string + '.txt'
    resulting_string = ''
    
    eval = evaluator.Evaluator(list_of_corpora, [clsf.dumb_classify for clsf in list_of_classifiers])
    to_be_added = '\nDUMB:\n' + eval.evaluate() + '\n'
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(to_be_added)  
    resulting_string += to_be_added
    print('\nFinished computing DUMB classification.\n')
    
    eval = evaluator.Evaluator(list_of_corpora, [clsf.length_classify for clsf in list_of_classifiers])
    to_be_added = '\nLENGTH:\n' + eval.evaluate() + '\n'
    with open(output_filename, 'a', encoding='utf-8') as f:
        f.write(to_be_added)  
    resulting_string += to_be_added
    # for ground_corpus in list_of_corpora:
    #     ground_corpus.avg_story_lengths = None
    print('\nFinished computing LENGTH classification.\n')
    
    eval = evaluator.Evaluator(list_of_corpora, [clsf.simple_reuters_classify for clsf in list_of_classifiers])
    to_be_added = '\nREUTERS:\n' + eval.evaluate() + '\n'
    with open(output_filename, 'a', encoding='utf-8') as f:
        f.write(to_be_added)  
    resulting_string += to_be_added
    # for ground_corpus in list_of_corpora:
    #     ground_corpus.simple_reuters_model = None
    print('\nFinished computing REUTERS classification.\n')
    
    eval = evaluator.Evaluator(list_of_corpora, [clsf.book_inspired_classify for clsf in list_of_classifiers])
    to_be_added = '\nBOOK:\n' + eval.evaluate() + '\n'
    with open(output_filename, 'a', encoding='utf-8') as f:
        f.write(to_be_added)  
    resulting_string += to_be_added
    # for ground_corpus in list_of_corpora:
    #     ground_corpus.book_model_data = (None, None, None, None, None)
    print('\nFinished computing BOOK classification.\n')
    
    eval = evaluator.Evaluator(list_of_corpora, [clsf.doc2vec_classify for clsf in list_of_classifiers])
    to_be_added = '\nDOC2VEC:\n' + eval.evaluate() + '\n'
    with open(output_filename, 'a', encoding='utf-8') as f:
        f.write(to_be_added)  
    resulting_string += to_be_added
    # for ground_corpus in list_of_corpora:
    #    ground_corpus.doc2vec_model_data = (None, None, None)
    print('\nFinished computing DOC2VEC classification.\n')
    
    eval = evaluator.Evaluator(list_of_corpora, [clsf.ngram_classify for clsf in list_of_classifiers])
    to_be_added = '\nN-GRAM:\n' + eval.evaluate() + '\n'
    with open(output_filename, 'a', encoding='utf-8') as f:
        f.write(to_be_added)  
    resulting_string += to_be_added
    # for ground_corpus in list_of_corpora:
    #    ground_corpus.ngram_model_data = (None, None, None)
    print('\nFinished computing N-GRAM classification.\n')
    
    print(resulting_string)
    print('\nSuccessfully exported results to "' + output_filename + '".')
