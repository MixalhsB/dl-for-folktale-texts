# OLD NAME: prepare-data-for-generating-1.3

import string
import re
import os


# load doc into memory
def load_doc(filename):
    # open the file as read only
    with open(filename, "r", encoding="utf-8") as file:
        # read all text
        text = file.read()
        # close the file
        return text


# turn a doc into clean tokens
def clean_doc(doc):
    # replace '--' with a space ' '
    doc = doc.replace('--', ' ')
    doc = doc.replace("&#039;", "\'")
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # remove html Tags <p> and <br>
    tags = re.compile('<.*?>')
    tokens = [tags.sub("", w) for w in tokens]
    # make lower case
    tokens = [word.lower() for word in tokens]
    return tokens


# save tokens to file, one dialog per line
def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w',encoding = 'utf-8')
    file.write(data)
    file.close()

    
# load
def load_tales(in_filename):


    print("Loading,cleaning and organizing sequences of " + str(in_filename))
    doc = load_doc(in_filename)
    print(doc[:200])

    # clean document
    tokens = clean_doc(doc)
    print(tokens[:200])
    print('Total Tokens: %d' % len(tokens))
    print('Unique Tokens: %d' % len(set(tokens))) #Vocabulary size

    # organize into sequences of tokens
    # as input to our model
    length = 50 + 1
    sequences = list()
    for i in range(length, len(tokens)):
        # select sequence of tokens
        seq = tokens[i-length:i]
        # convert into a line
        line = ' '.join(seq)
        # store
        sequences.append(line)
    print('Total Sequences: %d' % len(sequences))

    return sequences

#load tales and return sequences for generating:
language = 'german'
animalseq = load_tales("clean/" + language + '_'+ 'animaltales_clean.txt')
magicseq = load_tales("clean/" + language + '_'+'magictales_clean.txt')
religiousseq = load_tales("clean/" + language + '_'+'religioustales_clean.txt')
realisticseq = load_tales("clean/" + language + '_'+'realistictales_clean.txt')
stupidogreseq = load_tales("clean/" + language + '_'+'stupidogre_clean.txt')
jokesseq = load_tales("clean/" + language + '_'+'jokes_clean.txt')
formulaseq = load_tales("clean/" + language + '_'+'formulatales_clean.txt')

if not os.path.isdir("sequence"):
    os.makedirs("sequence")

# save sequences to files
out_filename1 = "sequence/" + language + '_'+'animaltales_sequences.txt'
save_doc(animalseq, out_filename1)
out_filename2 = "sequence/" + language + '_'+'magictales_sequences.txt'
save_doc(magicseq, out_filename2)
out_filename3 = "sequence/" + language + '_'+'religioustales_sequences.txt'
save_doc(religiousseq, out_filename3)
out_filename4 = "sequence/" + language + '_'+'realistictales_sequences.txt'
save_doc(realisticseq, out_filename4)
out_filename5 = "sequence/" + language + '_'+'stupidogre_sequences.txt'
save_doc(stupidogreseq, out_filename5)
out_filename6 = "sequence/" + language + '_'+'jokes_sequences.txt'
save_doc(jokesseq, out_filename6)
out_filename7 = "sequence/" + language + '_'+'formulatales_sequences.txt'
save_doc(formulaseq, out_filename7)

print("Files created: " + '\n' + out_filename1 + '\n' + out_filename2)
print(out_filename3 + '\n' + out_filename4 + '\n' +out_filename5)
print(out_filename6 + '\n' +out_filename7 + '\n')
      

