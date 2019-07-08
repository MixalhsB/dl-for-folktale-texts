# OLD NAME: atu_crawler_version4.3

__author__ = 'Simon'
import requests #ein Modul um Websites zu laden und querys durchzuführen

import re
import urllib.parse #eine Modulfunktion um URL strings in seine Komponenten aufzuteilen oder URL componenten zu einem URL string zu verbinden #google
from time import sleep #Programm soll für n Sekunden schlafen
#from corpus_classes import Story
from collections import *


class Story:

    def __init__(self):
        #(Titel,Datenbanknummer,ATU Nummer,Sprache,Text)
        self.story = ()

    def set_title(self,title):
        self.story += (title,) #(...,) --> damit es als tupel addiert wird

    def set_id(self,index):
        self.story += (int(index),)

    def set_atu_type(self,atu_type):
        self.story += (atu_type,)

    def set_language(self,language):
        self.story += (language,)

    def set_text(self,text):
        self.story += (text,)





        
def get_story(link, corpus_language):
    global num_matching_errors
    print('Trying to get a story from: ' + link)
    result = Story() #result ist eine Instanz der Klasse Story

    text_re = re.compile(r'id=story>(?:<p.*?>)?(.*?)</div>')
    id_re = re.compile(r'id=([0-9]+)')
    title_re = re.compile(r'<TITLE>(.*?) - MFtD<')
    atu_re = re.compile(r'\(ATU.*?([0-9]+?[A-Z]?)\)')



###Test ob die Seite überhaupt geladen werden kann###
    
    successful = False
    n = 1
    while(not successful and n <= 61):
        try:
            r = requests.get(link) #lade einen Link, vom Objekt r kann man jetzt Informationen lesen
            successful = True #es kommt keine Fehlermeldung -> successful
        except requests.exceptions.ConnectionError: #Netzwerkproblem
            print('Sleeping for ' + str(n) + ' seconds ...')
            sleep(n) #das Programm macht für n Sekunden eine Pause
            n += 5
    if (not successful or r.status_code != 200): #Status Code 200 bedeutet korrekte Anfrage,keine Fehler (vgl. 404)
        print('Could not access: '+link)
        return None

#########################################################

    try:
        r.encoding = 'utf-8' #Zeichenkodierung der Daten ist einheitlich UTF8
        title = title_re.findall(r.text)[0]
        #r.text : Requests dekodiert automatisch den Inhalt der Antwort vom Server
        #mit dem regulären Ausdruck für Titel wird der Titel als String herausgesucht (als 1. Element einer Liste)


        #result soll ein Tupel der Form (Titel,Datenbanknummer,ATU Nummer,Sprache,Text) werden
        #deswegen habe ich hier die Reihenfolge, in der die Methoden aufgerufen werden,leicht geändert
        
        result.set_title(title) #es gibt eine Klassenmethode set_title

        index = id_re.findall(link)[0] #sucht die ID
        
        result.set_id(index) #es gibt eine Klassenmethode set_id

        

        

        atu_candidate = atu_re.findall(r.text) #sucht nach der ATU Nummer
        
        if len(atu_candidate) > 1:
            #print("Error: Multiple possible ATU story types identified! Story: "+str(id)+", Candidate list: "+str(atu_candidate))
            atu_number = atu_candidate[0]
        elif len(atu_candidate) == 1:
            atu_number = atu_candidate[0]
        else:
            atu_number = 'UNKNOWN'
        result.set_atu_type(atu_number) #es gibt eine Klassenmethode set_atu_type

        result.set_language(corpus_language) #es gibt eine Klassenmethode set_language

        text = text_re.findall(r.text)[0] #sucht den Text
        result.set_text(text) #es gibt eine Klassenmethode set_text
        
    except IndexError: #bei der Anfrage [0] wird ein Index Error geworfen, mind. ein regex hat nicht gematched
        num_matching_errors += 1
        return None
    return result.story #ein 5-stelliges Tupel mit den Inhalten
    #Titel,ID,ATU,Sprache,Text

i = 0

def get_corpus(link, language):
    global num_matching_errors
    global i

    

    corpus = []
    story_link_re = re.compile(r'href=\'(.*action=story.*?)\'>') #Jedes Märchen enthält den action=story Teil
    next_page_link_re = re.compile(r'href=\'(.*?)\'>volgende') #bringt uns zum nächsten Märchen

###Test ob link gelanden werden kann###
    successful = False
    n = 1
    while(not successful and n <= 61):
        try:
            r = requests.get(link) #der Link zu allen Märchen einer Sprache wird geladen
            successful = True
        except requests.exceptions.ConnectionError:
            print('Sleeping happened ...')
            sleep(n)
            n += 5
    if (not successful or r.status_code != 200):
        print('Could not access: '+link)
        return corpus #leere Liste, falls ConnectionError

###

    story_links = story_link_re.findall(r.text) #sucht nach story_link_re
    #Liste von relativen URLS
    
    for rel_link in story_links:
        abs_link = urllib.parse.urljoin('http://www.mftd.org', rel_link)
        #ergibt eine vollständige URL / kombiniert eine base URL mit einer anderen URL
        
        s = get_story(abs_link, language) #obige Funktion wird aufgerufen
        if s != None:
            corpus.append(s) #das result (Objekt von Story()) wird in die corpus Liste aufgenommen
            i += 1
    next_page_link = next_page_link_re.findall(r.text) #sucht nach next_page_link
    
    assert(len(next_page_link) <= 1) #es soll entweder kein oder 1 next_page_link gefunden werden
    #wenn das nicht so ist kommt ein AssertionError

    #für Testzwecke das i -> wie viele Märchen sollen geladen werden
    #while i != 0:

        #i -= 1
    
    if len(next_page_link) == 1:
        rel_np_link = next_page_link[0]
        abs_np_link = urllib.parse.urljoin('http://www.mftd.org', rel_np_link)
        corpus += get_corpus(abs_np_link, language) #get_corpus ruft sich selbst mit einem neuen Link auf
            

    return corpus #Liste von results

if __name__ == '__main__':

    corpora = defaultdict(list) #Dictionary mit Sprachen als keys und Korpora als Werte
    corpora_no_atu = defaultdict(list)
    num_matching_errors = 0

    #Links zu den Daten, nach Sprache sortiert
    
    czech_link = 'http://www.mftd.org/index.php?action=browse&langname=Czech'
    danish_link = 'http://www.mftd.org/index.php?action=browse&langname=Danish'
    dutch_link = 'http://www.mftd.org/index.php?action=browse&langname=Dutch'
    english_link = 'http://www.mftd.org/index.php?action=browse&langname=English'
    french_link = 'http://www.mftd.org/index.php?action=browse&langname=French'
    german_link = 'http://www.mftd.org/index.php?action=browse&langname=German'
    hungarian_link = 'http://www.mftd.org/index.php?action=browse&langname=Hungarian'
    italian_link = 'http://www.mftd.org/index.php?action=browse&langname=Italian'
    polish_link = 'http://www.mftd.org/index.php?action=browse&langname=Polish'
    russian_link = 'http://www.mftd.org/index.php?action=browse&langname=Russian'
    spanish_link = 'http://www.mftd.org/index.php?action=browse&langname=Spanish'

    unknowntexts = defaultdict(list)
    animaltales =  defaultdict(list)
    magictales =  defaultdict(list)
    religioustales =  defaultdict(list)
    realistictales =  defaultdict(list)
    stupidogre =  defaultdict(list)
    jokes =  defaultdict(list)
    formulatales =  defaultdict(list)
    
    for link in [german_link]: #[english_link, french_link, german_link, italian_link, spanish_link]:  # [czech_link, danish_link, dutch_link, english_link, french_link, german_link, hungarian_link, italian_link, polish_link, russian_link, spanish_link]:
        counter = 0
        unknowncounter = 0
        animallen = 0
        magiclen = 0
        religiouslen = 0
        realisticlen = 0
        stupidogrelen = 0
        jokeslen = 0
        formulalen = 0
        
        language = link.split('=')[-1].lower() #am Ende des Links steht immer die Sprache
##        #Erstelle pro Sprache einen Ordner in dem die Märchen gespeichert werden sollen:
##        dirName = language
##        if not os.path.exists(dirName):
##            os.mkdir(dirName)
##            print("Directory " , dirName ,  " created ")
##        else:    
##            print("Directory " , dirName ,  " already exists")
            
        print('Processing Folktales in: '+language+'...')
        corpus = get_corpus(link, language)
        for story in corpus:
            if story[2] != 'UNKNOWN':
                corpora[language].append(story)
                counter +=1
                #(Titel,Datenbanknummer,ATU Nummer,Sprache,Text)

    #1-299 Animal
    #300-749 Magic
    #750-849 Religion
    #850-999 Realistic
    #1000-1199 Stupid Ogre
    #1200-1999 Jokes
    #2000-2399 formula tales

                try:
                    atu = int(story[2]) 

                except ValueError: #manche ATUs enthalten zum Schluss einen Buchstaben wie A/B

                    atu = int(story[2][:-1])

                if 1 <= atu <= 299:

                    animaltales[language].append(story[4])
                    animallen += len(story[4].split())

                if 300 <= atu <= 749:
                    magictales[language].append(story[4])
                    magiclen += len(story[4].split())

                if 750 <= atu <= 849:
                    religioustales[language].append(story[4])
                    religiouslen += len(story[4].split())

                if 850 <= atu <= 999:
                    realistictales[language].append(story[4])
                    realisticlen += len(story[4].split())

                if 1000 <= atu <= 1199:
                    stupidogre[language].append(story[4])
                    stupidogrelen += len(story[4].split())

                if 1200 <= atu <= 1999:
                    jokes[language].append(story[4])
                    jokeslen += len(story[4].split())

                if 2000 <= atu <= 2399:
                    formulatales[language].append(story[4])
                    formulalen += len(story[4].split())

                
            else:
                corpora_no_atu[language].append(story)
                unknowncounter += 1
                unknowntexts[language].append(story[4])
                
        #sprich ein Wert ist immer eine Liste von results
        #in dieser Liste stecken Märchen in der Form von (Titel,Datenbanknummer,ATU Nummer,Sprache,Text)
        
        #print('Matching Errors: '+str(num_matching_errors))

        
        with open("clean/" + language + "_" + 'corpora.txt', 'w+', encoding='utf-8') as outfile:
            outfile.write(str(corpora[language]))

        with open("clean/" + language + "_" + 'corpora_no_atu.txt', 'w+', encoding='utf-8') as outfile:
            outfile.write(str(corpora_no_atu[language]))
            

        with open("clean/" + language + "_" + 'unknowntexts_clean.txt',"w+",encoding = 'utf-8') as f:
            for t in unknowntexts[language]:
                f.write(t)
                f.write('\n')

        with open("clean/" + language + "_" + 'animaltales_clean.txt',"w+",encoding = 'utf-8') as f:
            for t in animaltales[language]:
                f.write(t)
                f.write('\n')

        with open("clean/" + language + "_" + 'magictales_clean.txt',"w+",encoding = 'utf-8') as f:
            for t in magictales[language]:
                f.write(t)
                f.write('\n')

        with open("clean/" + language + "_" + 'religioustales_clean.txt',"w+",encoding = 'utf-8') as f:
            for t in religioustales[language]:
                f.write(t)
                f.write('\n')

        with open("clean/" + language + "_" + 'realistictales_clean.txt',"w+",encoding = 'utf-8') as f:
            for t in realistictales[language]:
                f.write(t)
                f.write('\n')

        with open("clean/" + language + "_" + 'stupidogre_clean.txt',"w+",encoding = 'utf-8') as f:
            for t in stupidogre[language]:
                f.write(t)
                f.write('\n')

        with open("clean/" + language + "_" + 'jokes_clean.txt',"w+",encoding = 'utf-8') as f:
            for t in jokes[language]:
                f.write(t)
                f.write('\n')

        with open("clean/" + language + "_" + 'formulatales_clean.txt',"w+",encoding = 'utf-8') as f:
            for t in formulatales[language]:
                f.write(t)
                f.write('\n')

        print("Number of Folktales with ATU: " + str(counter))
        print("Number of Animal Folktales: " + str(len(animaltales[language])))
        print("Average length of an animal tale: " + str(animallen/len(animaltales[language])))
        
        print("Number of Magic Folktales: " + str(len(magictales[language])))
        print("Average length of a magic tale: " + str(magiclen/len(magictales[language])))
        
        print("Number of Religious Folktales: " + str(len(religioustales[language])))
        print("Average length of a religious tale: " + str(religiouslen/len(religioustales[language])))
        
        print("Number of Realistic Folktales: " + str(len(realistictales[language])))
        print("Average length of a realistic tale: " + str(realisticlen/len(realistictales[language])))
        
        print("Number of Stupid Ogre Folktales: " + str(len(stupidogre[language])))
        print("Average length of a stupid ogre tale: " + str(stupidogrelen/len(stupidogre[language])))
        
        print("Number of Jokes: " + str(len(jokes[language])))
        print("Average length of a joke: " + str(jokeslen/len(jokes[language])))
        
        print("Number of Formula Folktales: " + str(len(formulatales[language])))
        print("Average length of an formula tale: " + str(formulalen/len(formulatales[language])))
        
        print("Number of Folktales without ATU: " + str(unknowncounter))
        print("Number of Folktales in total: " +str(i))
        
    

