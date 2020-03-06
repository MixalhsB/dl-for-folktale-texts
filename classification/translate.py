from time import sleep
from bs4 import BeautifulSoup
from json.decoder import JSONDecodeError
import googletrans
import os
###'''
import time
import random
import itertools
import subprocess
###'''
from corpus import Corpus
from classifier import Classifier


###'''
vpn_countries = ['South Africa', 'Egypt' , 'Australia', 'New Zealand',  'South Korea', 'Singapore', 'Taiwan', 'Vietnam',
                 'Hong Kong', 'Indonesia', 'Thailand', 'Japan', 'Malaysia', 'United Kingdom', 'Netherlands', 'Germany',
                 'France', 'Belgium', 'Switzerland', 'Sweden', 'Spain', 'Denmark', 'Italy', 'Norway', 'Austria', 'Romania',
                 'Czech Republic', 'Luxembourg', 'Poland', 'Finland', 'Hungary', 'Latvia', 'Russia', 'Iceland', 'Bulgaria',
                 'Croatia', 'Moldova', 'Portugal', 'Albania', 'Ireland', 'Slovakia', 'Ukraine', 'Cyprus', 'Estonia', 'Georgia',
                 'Greece', 'Serbia', 'Slovenia', 'Azerbaijan', 'Bosnia and Herzegovina', 'Macedonia', 'India', 'Turkey',
                 'Israel', 'United Arab Emirates', 'United States', 'Canada', 'Mexico', 'Brazil', 'Costa Rica', 'Argentina',
                 'Chile']

def select_server(l):
    return random.choice(l)
###'''

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

    other_stories = other_corpus.stories
    
    if not os.path.isdir('pretranslated'):
        os.mkdir('pretranslated')
    
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
    
        def split_prosa_text_into_rough_halfs(text):
            middle_cut_point = text.count('.') // 2
            assert middle_cut_point > 0
            text_left_half = ''
            text_right_half = ''
            period_count = 0
            reached_right = False
            for char in text:
                if char == '.':
                    period_count += 1
                if not reached_right:
                    text_left_half += char
                else:
                    text_right_half += char
                if period_count == middle_cut_point:
                    reached_right = True
            print('--> needed to split text in halfs of lengths', len(text_left_half), 'and', len(text_right_half))
            assert len(text_left_half) > 0
            assert len(text_right_half) > 0
            assert len(text_left_half) + len(text_right_half) == len(text)
            return text_left_half, text_right_half
        
        def cut_text_into_small_enough_parts(text, max_size=13000):
            if len(text) > max_size:
                text_left_half, text_right_half = split_prosa_text_into_rough_halfs(text)
                return [cut_text_into_small_enough_parts(text_left_half), cut_text_into_small_enough_parts(text_right_half)]
            else:
                return [text]
        
        def flatten(container):
            for i in container:
                if isinstance(i, (list, tuple)):
                    for j in flatten(i):
                        yield j
                else:
                    yield i
        
        with open('pretranslated/' + save_file, 'w', encoding= 'utf-8') as s_f:
            
            n = 0
            
            translator = googletrans.Translator()
            
            for story in other_stories:
                tale = BeautifulSoup(story[4], "html.parser").text
                title = BeautifulSoup(story[0], "html.parser").text
                
                print(title, '---', len(tale))
                
                whole_or_parts = list(flatten(cut_text_into_small_enough_parts(tale)))
                
                # Translate the tale:
                trans_tale_parts = []
                for text in whole_or_parts:
                    for attempt_count in range(5):
                        if attempt_count > 0:
                            print('Tale translation attempt', attempt_count + 1, '/', 5, '...')
                        try:
                            trans_tale_parts.append(translator.translate(text, src=other_language, dest=ground_language).text)
                            break
                        except JSONDecodeError:
                            print('Disconnecting from possible previous VPN server ...')
                            process = subprocess.Popen(['nordvpn', '-d'], shell=True,
                                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            process.wait()
                            input('Press ENTER when ready to continue (1a) ...')
                            print('Connecting to new VPN server ...')
                            process = subprocess.Popen(['nordvpn', '-c', '-g', select_server(vpn_countries)],
                                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            process.wait()
                            input('Press ENTER when ready to continue (2a) ...')
                    else:
                        print('Could not translate tale "' + title + '".')
                        assert False
                trans_tale = ' '.join(trans_tale_parts)
                
                # Translate the title:
                for attempt_count in range(5):
                    if attempt_count > 0:
                        print('Title translation attempt', attempt_count + 1, '/', 5, '...')
                    try:
                        trans_title = translator.translate(title, src=other_language, dest=ground_language).text
                        break
                    except JSONDecodeError:
                        print('Disconnecting from possible previous VPN server ...')
                        process = subprocess.Popen(['nordvpn', '-d'], shell=True,
                                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        process.wait()
                        input('Press ENTER when ready to continue (1b) ...')
                        print('Connecting to new VPN server ...')
                        process = subprocess.Popen(['nordvpn', '-c', '-g', select_server(vpn_countries)],
                                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        process.wait()
                        input('Press ENTER when ready to continue (2b) ...')
                else:
                    print('Could not translate title "' + title + '".')
                    assert False
                
                print(title + ' -> ' + trans_title)
                new_story = trans_title, story[1], story[2], story[3], trans_tale
                ground_corpus.train_stories.append(new_story)
                n += 1

                s_f.write(str(trans_title) + "\t" +
                          str(story[1]) + "\t" +
                          str(story[2]) + "\t" +
                          str(story[3]) + "\t" +
                          str(trans_tale) + "\n")

        if len(other_stories) > 0:
            print("\nThe ground corpus' training data was extended from a total number of "
                    + str(previous_len) + " tales to now " + str(previous_len + n)
                     + " tales in total " + "\n")


def story_intersection(c1, c2):
    ''' Takes two Corpus-objects as input, and returns a tuple containing
    - the set of atu numbers appearing in both corpuses
    - the ratio between common atus vs. atus in corpus1
    - the ratio between common atus vs. atus in corpus1
    '''
    c1_atus = [story[2] for story in c1.stories]
    c2_atus = [story[2] for story in c2.stories]
    i = c1_atus & c2_atus
    return i, len(i) / len(c1_atus), len(i) / len(c2_atus)


def unique_tales(ground_corpus, other_corpus):
    '''  Takes two Corpus-objects as input, first one is the main corpus,
    the second one will be used to expand the first one. All tales only
    in the other corpus  will be added to the stories of the ground corpus
    '''
    previous_len = len(ground_corpus.stories)
    ground_atus = [story[2] for story in ground_corpus.stories]
    other_atus = [story[2] for story in other_corpus.stories]
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
    
    ground_atus = [story[2] for story in ground_corpus.stories]
    other_atus = [story[2] for story in other_corpus.stories]
    
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
