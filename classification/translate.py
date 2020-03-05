from time import sleep
from bs4 import BeautifulSoup
from json.decoder import JSONDecodeError
import googletrans
import os.path
from corpus import Corpus
from classifier import *


# How many tales do two corpora in different languages have in common?
def story_intersection(c1, c2):
    ''' Takes two Corpus-objects as input, and returns a tuple containing
    - the set of atu numbers appearing in both corpuses
    - the ratio between common atus vs. atus in corpus1
    - the ratio between common atus vs. atus in corpus1
    '''
    c1_atus = {story[2] for story in c1.stories}
    c2_atus = {story[2] for story in c2.stories}
    i = c1_atus & c2_atus
    return i, len(i) / len(c1_atus), len(i) / len(c2_atus)

def unique_tales(ground_corpus, other_corpus):
    '''  Takes two Corpus-objects as input, first one is the main corpus,
    the second one will be used to expand the first one. All tales only
    in the other corpus  will be added to the stories of the ground corpus
    '''
    previous_len = len(ground_corpus.stories)
    ground_atus = {story[2] for story in ground_corpus.stories}
    other_atus = {story[2] for story in other_corpus.stories}
    unique_atus = other_atus - ground_atus
    #return(unique_atus)
    n = 0
    for atu in unique_atus:
        for story in other_corpus.get_stories_of_atu(atu):
            ground_corpus.stories.append(story)
        n += 1
    print("The ground corpus was extended from a total number of "
          + str(previous_len) + " tales to now " + str(previous_len + n)
          + " tales in total.") 

def extract_translate_unique_tales(ground_corpus, other_corpus):
    ''' Takes two Corpus-objects and a filename as input, first one is the main corpus,
    the second one will be used to expand the first one. All tales only
    in the other corpus will be added to the stories of the ground corpus.
    '''
    previous_len = len(ground_corpus.stories)

    ground_language = ground_corpus.language
    other_language = other_corpus.language
    
    ground_atus = {story[2] for story in ground_corpus.stories}
    other_atus = {story[2] for story in other_corpus.stories}
    
    unique_atus = other_atus - ground_atus
    
    print('ui', len(unique_atus), len(other_atus))
    
    n = 0
        
    for atu in unique_atus:
        stories = other_corpus.get_stories_of_atu(atu)
        assert stories != []
        for story in stories:
            tale = BeautifulSoup(story[4], "html.parser").text
            title = BeautifulSoup(story[0], "html.parser").text
        
        try:
            translator = googletrans.Translator()
            print(title)
            trans_tale = translator.translate(tale, src=other_language, dest=ground_language).text
            trans_title = translator.translate(title, src=other_language, dest=ground_language).text
            print(title + ' -> ' + trans_title)
            new_story = trans_title, story[1], story[2], story[3], trans_tale
            ground_corpus.stories.append(new_story)
            n += 1
        except JSONDecodeError:
            print('error while trying to connect to Google Translate')
            pass

    if len(unique_atus) > 0:
        print("\nThe ground corpus was extended from a total number of "
              + str(previous_len) + " tales to now " + str(previous_len + n)
              + " tales in total.\n")
        
def translate_tales_for_training_data(ground_corpus, other_corpus):
    ''' Takes two Corpus-objects and a filename as input, first one is the main corpus,
    the second one will be used to expand the first one. ALL tales
    in the other corpus will be added to the TRAINING stories of the ground corpus.
    '''
    previous_len = len(ground_corpus.train_stories)

    ground_language = ground_corpus.language
    other_language = other_corpus.language
    

    other_stories = {story for story in other_corpus.stories}
    assert other_stories != []
    
    n = 0
        
    for story in other_stories:
        tale = BeautifulSoup(story[4], "html.parser").text
        title = BeautifulSoup(story[0], "html.parser").text
        
        try:
            translator = googletrans.Translator()
            print(title)
            trans_tale = translator.translate(tale, src=other_language, dest=ground_language).text
            trans_title = translator.translate(title, src=other_language, dest=ground_language).text
            print(title + ' -> ' + trans_title)
            new_story = trans_title, story[1], story[2], story[3], trans_tale
            ground_corpus.train_stories.append(new_story)
            n += 1
        except JSONDecodeError:
            print('error while trying to connect to Google Translate')
            pass

    if len(other_stories) > 0:
        print("\nThe ground corpus' training data was extended from a total number of "
              + str(previous_len) + " tales to now " + str(previous_len + n)
              + " tales in total.\n")

        
def translate_save_tales(ground_corpus, other_corpus):
    ''' Takes two Corpus-objects and a filename as input, first one is the main corpus,
    the second one will be used to expand the first one. ALL tales
    in the other corpus will be added to the TRAINING stories of the ground corpus.
    After method has been called on language pair once, result is saved in a file. For every other
    call, the data will be read from the file. 
    '''
    
    previous_len = len(ground_corpus.train_stories)

    ground_language = ground_corpus.language
    other_language = other_corpus.language

    other_stories = {story for story in other_corpus.stories}
    assert other_stories != []
    
    save_file = str(ground_language) + "_" + str(other_language) + ".txt"
    
    if os.path.isfile('pretranslated/' + save_file):
        with open('pretranslated/' + save_file) as s_f:
            n=0
            for line in s_f:
                story = line.split("\t")
                assert story[1].isdigit()
                story[1] = int(story[1])
                story = tuple(story)
                ground_corpus.train_stories.append(story)
                n+=1

    else:
        with open('pretranslated/' + save_file, 'w', encoding= 'utf-8') as s_f:
            
            n = 0
                
            for story in other_stories:
                tale = BeautifulSoup(story[4], "html.parser").text
                title = BeautifulSoup(story[0], "html.parser").text

                try:
                    translator = googletrans.Translator()
                    print(title)
                    trans_tale = translator.translate(tale, src=other_language, dest=ground_language).text
                    trans_title = translator.translate(title, src=other_language, dest=ground_language).text
                    print(title + ' -> ' + trans_title)
                    new_story = trans_title, story[1], story[2], story[3], trans_tale
                    ground_corpus.train_stories.append(new_story)
                    n += 1
                except JSONDecodeError:
                    print('error while trying to connect to Google Translate')
                    continue

                s_f.write(str(trans_title) + "\t" +
                          str(story[1]) + "\t" +
                          str(story[2]) + "\t" +
                          str(story[3]) + "\t" +
                          str(trans_tale) + "\n")

        if len(other_stories) > 0:
            print("\nThe ground corpus' training data was extended from a total number of "
                    + str(previous_len) + " tales to now " + str(previous_len + n)
                     + " tales in total " + "\n")




# German = Corpus('../corpora.dict', 'German', seed=123, binary_mode=True)
# English = Corpus('../corpora.dict', 'English', seed=123, binary_mode=True)

#translate_tales_for_training_data(English,German)
#print("hi")
#translate_save_tales(English,German)
#print("hello")

##French = Corpus('../corpora.dict', 'French', seed=123, binary_mode=True)
##Dutch = Corpus('../corpora.dict', 'Dutch', seed=123, binary_mode=True)
##Spanish = Corpus('../corpora.dict', 'Spanish', seed=123, binary_mode=True)
##Italian = Corpus('../corpora.dict', 'Italian', seed=123, binary_mode=True)
##Danish = Corpus('../corpora.dict', 'Danish', seed=123, binary_mode=True)
##Czech = Corpus('../corpora.dict', 'Czech', seed=123, binary_mode=True)
##
##
##for lang in [German, French, Dutch, Spanish, Italian, Danish, Czech]:
##    print(str(lang) + " contains " + str(len(unique_tales(English, lang))) + " tales that are not in the English corpus.")
