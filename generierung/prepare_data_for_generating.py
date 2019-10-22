import string
import re
import os


# funktioniert noch nicht!!!

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r',encoding = 'utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# turn a doc into clean tokens
def clean_doc(doc):
    # replace '--' with a space ' '
    doc = doc.replace('--', ' ')
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # remove html Tags <p> and <br>
    tokens = [word for word in tokens if word != 'p' and word != 'br']
    # make lower case
    tokens = [word.lower() for word in tokens]
    return tokens


# save tokens to file, one dialog per line
def save_doc(lines, filename):
    title = lines[0]
    text = lines[1]
    length = len(title)
    data = ""
    for i in range(0,length):
        data += title[i] + "\n" + text[i] + "\n\n"
    with open(filename, 'w+', encoding = 'utf-8') as f:
        f.write(data)

    
# load
def load_tales(in_filename):


    print("Loading,cleaning and organizing sequences of " + str(in_filename))
    doc = load_doc(in_filename)
    print(doc[:200])

    # einzelne Märchen sind durch \n\n getrennt
    fairytales = doc.split("\n\n")
    tales = dict()
    for text in fairytales:
        story = text.split("\n")
        # Titel als key, Text als value
        try:
            tales[story[0]] = story[1]
        except IndexError:
            continue

    # clean document
    title_tokens = []
    text_tokens = []
    for title, text in tales.items():
        title_tokens += clean_doc(title)
        text_tokens += clean_doc(text)
    print(title_tokens[:200])
    print(text_tokens[:200])
    print('Total Title Tokens: %d' % len(title_tokens))
    print('Total Text Tokens: %d' % len(text_tokens))
    print('Unique Title Tokens: %d' % len(set(title_tokens))) #Vocabulary size
    print('Unique Text Tokens: %d' % len(set(text_tokens)))


    # organize into sequences of tokens
    # as input to our model
    length = 50 + 1
    title_sequences = list()
    text_sequences = list()
    for i in range(length, len(title_tokens)):
        # select sequence of tokens
        title_seq = title_tokens[i-length:i]
        # convert into a line
        title_line = ' '.join(title_seq)
        # store
        title_sequences.append(title_line)
    for j in range(length, len(text_tokens)):
        text_seq = text_tokens[j-length:j]
        text_line = " ".join(text_seq)
        text_sequences.append(text_line)
    print('Total Title Sequences: %d' % len(title_sequences))
    print("Total Text Sequences: %d" % len(title_sequences))

    return (title_sequences, text_sequences)

if __name__ == "__main__":
    #load tales and return sequences for generating:
    languages = ["Czech", "Danish", "Dutch", "English", "French", "German", "Hungarian", "Italian", "Polish", "Russian", "Spanish"]
    os.chdir("..") # um in den vorderen Ordner zu gelangen (da dort der "clean" Ordner liegt)
    for language in languages:
        animalseq = load_tales("clean/" + language + '_'+ 'animaltales_clean.txt')
        magicseq = load_tales("clean/" + language + '_'+'magictales_clean.txt')
        religiousseq = load_tales("clean/" + language + '_'+'religioustales_clean.txt')
        realisticseq = load_tales("clean/" + language + '_'+'realistictales_clean.txt')
        stupidogreseq = load_tales("clean/" + language + '_'+'stupidogre_clean.txt')
        jokesseq = load_tales("clean/" + language + '_'+'jokes_clean.txt')
        formulaseq = load_tales("clean/" + language + '_'+'formulatales_clean.txt')

        # save sequences to files
        out_filename1 = "generierung/sequence/" + language + '_animaltales_sequences.txt'
        save_doc(animalseq, out_filename1)
        out_filename2 = "generierung/sequence/" + language + '_magictales_sequences.txt'
        save_doc(magicseq, out_filename2)
        out_filename3 = "generierung/sequence/" + language + '_religioustales_sequences.txt'
        save_doc(religiousseq, out_filename3)
        out_filename4 = "generierung/sequence/" + language + '_realistictales_sequences.txt'
        save_doc(realisticseq, out_filename4)
        out_filename5 = "generierung/sequence/" + language + '_stupidogre_sequences.txt'
        save_doc(stupidogreseq, out_filename5)
        out_filename6 = "generierung/sequence/" + language + '_jokes_sequences.txt'
        save_doc(jokesseq, out_filename6)
        out_filename7 = "generierung/sequence/" + language + '_formulatales_sequences.txt'
        save_doc(formulaseq, out_filename7)

        print("Files created: " + '\n' + out_filename1 + '\n' + out_filename2)
        print(out_filename3 + '\n' + out_filename4 + '\n' +out_filename5)
        print(out_filename6 + '\n' +out_filename7 + '\n')
        