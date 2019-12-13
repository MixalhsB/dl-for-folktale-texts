'''
heavily inspired by https://towardsdatascience.com/implementing-multi-class-text-classification-with-doc2vec-df7c3812824d
'''
from random import shuffle
from corpus import Corpus
from bs4 import BeautifulSoup 
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import utils
import csv
from tqdm import tqdm
import multiprocessing
import nltk
from nltk.corpus import stopwords


if __name__ == '__main__': 

    language = input("Enter language: ")
    cor = Corpus('../corpora.dict', language)
    categories = ['animal', 'magic', 'religious', 'realistic', 'ogre', 'jokes', 'formula']



    # Function for tokenizing
    def tokenize_text(text):
        tokens = []
        for sent in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sent):
                if len(word) < 2:
                    continue
                tokens.append(word.lower())
        return tokens

    # Associating the tags(labels) with numbers
    tags_index = {'animal': 1 , 'magic': 2, 'religious': 3, 'realistic': 4, 'ogre': 5, 'jokes': 6, 'formula': 7}


    # Initializing the variables
    train_documents = []
    test_documents = []
    i = 0

    # List with TaggedDocument objects containing a list with the tokenized story and a list containing the category number 
    for category in categories: 
        i=0
        for story in cor.gold_classes[category]:
            raw_text = BeautifulSoup(story[4], "html.parser").text #story[4] is just the text of the story 
            if i<5:  
                train_documents.append(TaggedDocument(words=tokenize_text(raw_text), tags=[tags_index.get(category)]))
                i+=1
            else:
                test_documents.append(TaggedDocument(words=tokenize_text(raw_text), tags=[tags_index.get(category)]))
                i=0
            shuffle(train_documents)
            shuffle(test_documents) 

    repeat = True
    while repeat == True: 
        vector_size_input = int(input("Enter vector size between 100 and 300: "))
        window_size_input = int(input("Enter a window size (maximum number of context words) between 1 and 10: "))

        # Feature Vector
        cores = multiprocessing.cpu_count()

        model_dbow = Doc2Vec(dm=1, window_size = window_size_input, vector_size=vector_size_input, negative=5, hs=0, min_count=2, sample = 0, workers=cores, alpha=0.025, min_alpha=0.001)
        model_dbow.build_vocab([x for x in tqdm(train_documents)])
        train_documents  = utils.shuffle(train_documents)
        model_dbow.train(train_documents,total_examples=len(train_documents), epochs=30)

        def vector_for_learning(model, input_docs):
            sents = input_docs
            targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
            return targets, feature_vectors

        model_dbow.save('./'+language+'.d2v')



        y_train, X_train = vector_for_learning(model_dbow, train_documents)
        y_test, X_test = vector_for_learning(model_dbow, test_documents)

        logreg = LogisticRegression(n_jobs=1, C=1e5)
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        print('Testing accuracy for stories' % accuracy_score(y_test, y_pred))
        print('Testing F1 score for stories: {}'.format(f1_score(y_test, y_pred, average='weighted')))

        repeat = True if input("Again? y/n: ")=='y' else False
