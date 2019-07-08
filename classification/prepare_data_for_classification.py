# OLD NAME: prepare-data-for-classification

import string
import re
from os import listdir
import os.path
from collections import Counter
from nltk.corpus import stopwords




# Nach Kapitel 15 aus Deep Learning for NLP
# Einteilung in Traings- und Testdaten ist schon gegebe
   
# load_doc ist in save_tales miteingebaut

# Im Prinzip gleich wie bei prepare-data-for-generating, bloß werden
# Stop-Words herausgefiltert. 
# turn a doc into clean tokens
def clean_tale(tale, language):
    # split into tokens by white space
    tokens = tale.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    
    # remove html-Tags <p> and <br>
    tokens = [word for word in tokens if word != 'p' and word != 'br']
    
    # filter out stop words, depending on the language
    stop_words = set(stopwords.words(language))
    tokens = [w for w in tokens if not w in stop_words]

    # transform all tokens to lower case
    tokens = [w.lower() for w in tokens]
    
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

# Nur die Tokens des festgelegten Vokabulars bleiben erhalten
def filter_tokens(tokens, vocab):
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)

#Entfernt alle Wörter des Vokabulars, die seltener als ein min_count und häufiger als ein max_count vorkommen
def change_vocab(vocab, min_count, max_count):
    vocab = {word: vocab[word] for word in vocab if word[vocab] >= min_count and word[vocab] <= max_count}
    return vocab


# Falls besonders häufige / besonders seltene Tokens rausgefiltert werden sollen, wird ein Vokabular erstellt
# Das vocab ist eine Instanz von Counter()
# load doc and add to vocab
def add_tokens_to_vocab(tokens, vocab):
    vocab.update(tokens)



def save_tales(in_file,language, lng):
    
    print("Working on " + in_file)
    
    with open(in_file, 'r', encoding= 'utf-8') as doc:
        tale_list= []
        voc = Counter()
        
        # Jedes Märchen wird mit clean_tale bearbeitet und das Vokabular wird erweitert
        for tale in doc:
            tokens = clean_tale(tale, language)
            add_tokens_to_vocab(tokens,voc)
            tale_list.append(tokens)

        # in Dateien, die wie bei prepare-data-for-generating, ein sequences, nicht mehr das clean, als Endung haben, wird das Ergebnis reingeschrieben:
        
        # Ich würde die Dateien gern in einen Nebenordner von Raw speichern, kriege das mit dem richtigen
        # Pfad aber irgendwie nicht hin
        
        #print(os.path.abspath(os.curdir))
        #os.chdir(lng)
        #print(os.path.abspath(os.curdir))
        #try:
            #os.mkdir("Clean")
        #except:
         #   pass
        #os.chdir("Clean")
        
        out_file =  in_file[11:-4] + "_" + "cleaned.txt"
       
        
        with open(out_file, 'w+', encoding = 'utf-8') as out:
            for tale in tale_list:
                # Falls das Vokabular angepasst werden soll:
                # change_vocab(voc, min, max)
                t = filter_tokens(tale,voc)
                out.write(t)
                out.write("\n")


def clean_language(lng):
    print ("Loading, cleaning and organizing tales in " + lng)

    d = os.path.join(lng, "Raw")
    
    save_tales(d +  os.sep + lng + '_' + 'animaltales.txt', lng.lower(), lng)
    save_tales(d + os.sep + lng + '_' + 'magictales.txt', lng.lower(), lng)
    save_tales(d + os.sep + lng + '_' + 'religioustales.txt', lng.lower(), lng)
    save_tales(d +  os.sep + lng + '_' + 'realistictales.txt', lng.lower(),lng)
    save_tales(d + os.sep + lng + '_' + 'stupidogre.txt', lng.lower(),lng)
    save_tales(d + os.sep + lng + '_' + 'jokes.txt', lng.lower(),lng)
    save_tales(d + os.sep + lng + '_' + 'formulatales.txt', lng.lower(), lng)
                            

    print("Done with " + lng + " .")

clean_language('German') 
        
            
            
            


