# How many tales do two corpora in different languages have in common?
def story_intersection(c1, c2):
    ''' Takes two Corpus-objects as input, and returns a tuple containing
    - the set of atu numbers appearing in both corpuses
    - the ratio between common atus vs. atus in corpus1
    - the ratio between common atus vs. atus in corpus1
    '''
    c1_atus= {story[2] for story in c1.stories}
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
        ground_corpus.stories.append(other_corpus.get_tale_of_atu(atu))
        n += 1
    print("The ground corpus was extended from a total number of "
          + str(previous_len)+ " tales to now " + str(previous_len + n)
          + " tales in total.") 

def extract_translate_unique_tales(ground_corpus, other_corpus, filename):
    ''' Takes two Corpus-objects and a filename as input, first one is the main corpus,
    the second one will be used to expand the first one. All tales only
    in the other corpus will be added to the stories of the ground corpus. The translated
    tales will be stored in filename, so that they can be classified as evaluation.
    The filename will contain lines, where the first element is the story and the second its atu.
    '''
    previous_len = len(ground_corpus.stories)

    ground_language = ground_corpus.language
    other_language = other_corpus.language
    
    ground_atus = {story[2] for story in ground_corpus.stories}
    other_atus = {story[2] for story in other_corpus.stories}
    
    unique_atus = other_atus - ground_atus
    
    translator = googletrans.Translator()
    n = 0
    
    with open(filename) as outfile:
        
        for atu in unique_atus:
            tale = other_corpus.get_tale_of_atu(atu)
            story = BeautifulSoup(tale[4], "html.parser").text
            title = BeautifulSoup(tale[0], "html.parser").text
            try:
            
                trans_tale = translator.translate(story, src=other_language, dest=ground_language).text
                trans_title = translator.translate(title, src=other_language, dest=ground_language).text
            except:
                print("Error with tale: \n")
                print("Title: " + title + "\n")
                print("Story: " + story + "\n")
            new_story = trans_title, tale[1], tale[2], tale[3], trans_tale
    
            ground_corpus.stories.append(new_story)
            n += 1

            outfile.write(trans_tale + "\t" + tale[2] + "\n")
            
        print("The ground corpus was extended from a total number of "
              + str(previous_len) + " tales to now " + str(previous_len + n)
              + " tales in total.")

##German = Corpus('../corpora.dict', 'German', seed=123, binary_mode=True)
##English = Corpus('../corpora.dict', 'English', seed=123, binary_mode=True)
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
