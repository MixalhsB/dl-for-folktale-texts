from collections import defaultdict
import os
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import math
import csv

class Story:

	def __init__(self, title, index, atu_type, language, text):
		#(Titel,Datenbanknummer,ATU Nummer,Sprache,Text)
		self.title = title
		self.index = index
		self.atu_type = atu_type
		self.language = language
		self.text = text

	def __str__(self):
		# for backwardscompatibility
		return str((self.title, self.index, self.atu_type, self.language, self.text))


def read_corpus(path):
	with open(path, "r", encoding = 'utf-8') as f:
		s = f.readline()
		data = eval(s.replace("<class 'list'>", 'list')) # for defaultdict typen
		return data

def write(path, content):
	with open("clean/" + path + '_clean.txt',"w+",encoding = 'utf-8') as f:
		# schreibe Titel und Text in einzelnene Zeilen separiert durch Leerzeile
		for story in content:
			f.write(story.title)
			f.write('\n')
			f.write(story.text)
			f.write('\n\n')

def average_sentence_length(content):
	number_of_sentences = 0
	number_of_words = 0
	for story in content:
		list_of_sentences = sent_tokenize(story.text)
		number_of_sentences += len(list_of_sentences)

		number_of_words += len(word_tokenize(story.text))
	try:
		res = round(number_of_words / number_of_sentences)
	except ZeroDivisionError:
		res = 0
	return res

def average_title_length(content):
	number_of_titles = 0
	number_of_words = 0
	for story in content:
		title = story.title
		number_of_titles += 1
		number_of_words += len(word_tokenize(title))
	try:
		res = round(number_of_words / number_of_titles)
	except ZeroDivisionError:
		res = 0
	return res





if __name__ == "__main__":

	## Initialisierung

	corpora = defaultdict(list) #Dictionary mit Sprachen als keys und Korpora als Werte
	corpora_no_atu = defaultdict(list)
	num_matching_errors = 0

	unknowntexts = defaultdict(list)
	animaltales =  defaultdict(list)
	magictales =  defaultdict(list)
	religioustales =  defaultdict(list)
	realistictales =  defaultdict(list)
	stupidogre =  defaultdict(list)
	jokes =  defaultdict(list)
	formulatales =  defaultdict(list)


	counter = 0
	unknowncounter = 0

	# dictionary that saves average sentence lengths per tale type and language
	sentence_length_average = {}
	min_max = {}
	sentence_length_average["animaltales"] = {}
	sentence_length_average["realistictales"] = {}
	sentence_length_average["magictales"] = {}
	sentence_length_average["religioustales"]= {}
	sentence_length_average["stupidogre"] = {}
	sentence_length_average["jokes"] = {}
	sentence_length_average["formulatales"] = {}
	min_max["animaltales"] = {}
	min_max["realistictales"] = {}
	min_max["magictales"] = {}
	min_max["religioustales"] = {}
	min_max["stupidogre"] = {}
	min_max["jokes"] = {}
	min_max["formulatales"] = {}

	length = defaultdict(int)
	title_length_average = {}
	title_length_average["animaltales"] = {}
	title_length_average["realistictales"] = {}
	title_length_average["magictales"] = {}
	title_length_average["religioustales"]= {}
	title_length_average["stupidogre"] = {}
	title_length_average["jokes"] = {}
	title_length_average["formulatales"] = {}

	writer = csv.writer(open("statistics.csv", "w+", newline=""), delimiter='\t')
	writer.writerow(["language", "tales", "animal", "realistic", "magic", "religious", "ogre", "jokes", "formula"])

	# corpora = read_corpus('corpora.txt')
	# corpora_no_atu = read_corpus('corpora_no_atu')
	corpora_default = read_corpus('corpora.txt')
	for language in corpora_default:
		number_of_tales = len(corpora_default[language])
		animallen = 0
		magiclen = 0
		religiouslen = 0
		realisticlen = 0
		stupidogrelen = 0
		jokeslen = 0
		formulalen = 0
		min_max_animallen = [math.inf, 0]
		min_max_magiclen = [math.inf, 0]
		min_max_religiouslen = [math.inf, 0]
		min_max_realisticlen = [math.inf, 0]
		min_max_stupidogrelen = [math.inf, 0]
		min_max_jokeslen = [math.inf, 0]
		min_max_formulalen = [math.inf, 0]
		for story in corpora_default[language]:
			story = Story(story[0],story[1],story[2],story[3],story[4])
			
			if story.atu_type == 'UNKNOWN':
				corpora_no_atu[language].append(str(story))
				unknowncounter += 1
				unknowntexts[language].append(story)
			else:
				corpora[language].append(str(story))

				try:
					atu = int(story.atu_type)
				except ValueError:
					atu = int(story.atu_type[:-1])
				
				#1-299 Animal
				#300-749 Magic
				#750-849 Religion
				#850-999 Realistic
				#1000-1199 Stupid Ogre
				#1200-1999 Jokes
				#2000-2399 formula tales

				if 1 <= atu <= 299:

					animaltales[language].append(story)
					animallen += len(story.text.split())
					if len(story.text.split()) < min_max_animallen[0]: # neues minimum
						min_max_animallen[0] = len(story.text.split())
					if len(story.text.split()) > min_max_animallen[1]: # neues maximum
						min_max_animallen[1] = len(story.text.split())


				elif 300 <= atu <= 749:
					magictales[language].append(story)
					magiclen += len(story.text.split())
					if len(story.text.split()) < min_max_magiclen[0]: # neues minimum
						min_max_magiclen[0] = len(story.text.split())
					if len(story.text.split()) > min_max_magiclen[1]: # neues maximum
						min_max_magiclen[1] = len(story.text.split())

				elif 750 <= atu <= 849:
					religioustales[language].append(story)
					religiouslen += len(story.text.split())
					if len(story.text.split()) < min_max_religiouslen[0]: # neues minimum
						min_max_religiouslen[0] = len(story.text.split())
					if len(story.text.split()) > min_max_religiouslen[1]: # neues maximum
						min_max_religiouslen[1] = len(story.text.split())

				elif 850 <= atu <= 999:
					realistictales[language].append(story)
					realisticlen += len(story.text.split())
					if len(story.text.split()) < min_max_realisticlen[0]: # neues minimum
						min_max_realisticlen[0] = len(story.text.split())
					if len(story.text.split()) > min_max_realisticlen[1]: # neues maximum
						min_max_realisticlen[1] = len(story.text.split())

				elif 1000 <= atu <= 1199:
					stupidogre[language].append(story)
					stupidogrelen += len(story.text.split())
					if len(story.text.split()) < min_max_stupidogrelen[0]: # neues minimum
						min_max_stupidogrelen[0] = len(story.text.split())
					if len(story.text.split()) > min_max_stupidogrelen[1]: # neues maximum
						min_max_stupidogrelen[1] = len(story.text.split())

				elif 1200 <= atu <= 1999:
					jokes[language].append(story)
					jokeslen += len(story.text.split())
					if len(story.text.split()) < min_max_jokeslen[0]: # neues minimum
						min_max_jokeslen[0] = len(story.text.split())
					if len(story.text.split()) > min_max_jokeslen[1]: # neues maximum
						min_max_jokeslen[1] = len(story.text.split())

				elif 2000 <= atu <= 2399:
					formulatales[language].append(story)
					formulalen += len(story.text.split())
					if len(story.text.split()) < min_max_formulalen[0]: # neues minimum
						min_max_formulalen[0] = len(story.text.split())
					if len(story.text.split()) > min_max_formulalen[1]: # neues maximum
						min_max_formulalen[1] = len(story.text.split())

		dirName = 'clean'
		if not os.path.exists(dirName):
			os.mkdir(dirName)

		write(language + '_animaltales' , animaltales[language])
		write(language + '_magictales' , magictales[language])
		write(language + '_religioustales' , religioustales[language])
		write(language + "_realistictales", realistictales[language])
		write(language + '_stupidogre' , stupidogre[language])
		write(language + '_jokes' , jokes[language])
		write(language + '_formulatales' , formulatales[language])
		write(language + '_unknowntexts' , unknowntexts[language])

		# print("Average sentence length: ", language + '_animaltales', average_sentence_length(animaltales[language]))
		sentence_length_average["animaltales"][language] = average_sentence_length(animaltales[language])
		title_length_average["animaltales"][language] = average_title_length(animaltales[language])
		sentence_length_average["realistictales"][language] = average_sentence_length(animaltales[language])
		title_length_average["realistictales"][language] = average_title_length(animaltales[language])
		sentence_length_average["magictales"][language] = average_sentence_length(magictales[language])
		title_length_average["magictales"][language] = average_title_length(magictales[language])
		sentence_length_average["religioustales"][language] = average_sentence_length(religioustales[language])
		title_length_average["religioustales"][language] = average_title_length(religioustales[language])
		sentence_length_average["stupidogre"][language] = average_sentence_length(stupidogre[language])
		title_length_average["stupidogre"][language] = average_title_length(stupidogre[language])
		sentence_length_average["jokes"][language] = average_sentence_length(jokes[language])
		title_length_average["jokes"][language] = average_title_length(jokes[language])
		sentence_length_average["formulatales"][language] = average_sentence_length(formulatales[language])
		title_length_average["formulatales"][language] = average_title_length(formulatales[language])
		# print("average sentence length: ", sentence_length_average)
		min_max["animaltales"][language] = min_max_animallen
		min_max["realistictales"][language] = min_max_realisticlen
		min_max["magictales"][language] = min_max_magiclen
		min_max["religioustales"][language] = min_max_religiouslen
		min_max["stupidogre"][language] = min_max_stupidogrelen
		min_max["jokes"][language] = min_max_jokeslen
		min_max["formulatales"][language] = min_max_formulalen

		# write into number_of_tales_statistics
		# (["language", "tales", "animal", "realistic", "magic", "religious", "ogre", "jokes", "formula"]
		writer.writerow([language, number_of_tales, len(animaltales[language]), len(realistictales[language]),
						 len(magictales[language]), len(religioustales[language]),
						 len(stupidogre[language]), len(jokes[language]), len(formulatales[language])])
		# write number of tokens
		sum_of_tokens = animallen + realisticlen + magiclen + religiouslen + stupidogrelen + jokeslen + formulalen
		writer.writerow([" ", sum_of_tokens, animallen, realisticlen, magiclen, religiouslen, stupidogrelen, jokeslen,
						 formulalen])


		with open(dirName + "/" + language + '_corpora.txt', 'w+', encoding='utf-8') as f:
			f.write(str(corpora[language]))

		with open(dirName + "/" + language + '_corpora_no_atu.txt', 'w+', encoding='utf-8') as f:
			f.write(str(corpora_no_atu[language]))

		# print("Number of Folktales with ATU: " + str(counter))
		# print("Number of Animal Folktales: " + str(len(animaltales[language])))

		# average length of tales is saved into dictionary
	
		try:
			# print("Average length of an animal tale: " + str(animallen / len(animaltales[language])))
			length[language + "_animaltales"] = round(animallen / len(animaltales[language]))

			# print("Number of Magic Folktales: " + str(len(magictales[language])))
			# print("Average length of a magic tale: " + str(magiclen / len(magictales[language])))
			length[language + "_magictales"] = round(magiclen / len(magictales[language]))

			# print("Number of Religious Folktales: " + str(len(religioustales[language])))
			# print("Average length of a religious tale: " + str(religiouslen / len(religioustales[language])))
			length[language + "_religioustales"] = round(religiouslen / len(religioustales[language]))

			# print("Number of Realistic Folktales: " + str(len(realistictales[language])))
			# print("Average length of a realistic tale: " + str(realisticlen / len(realistictales[language])))
			length[language + "_realistictales"] = round(realisticlen / len(realistictales[language]))

			# print("Number of Stupid Ogre Folktales: " + str(len(stupidogre[language])))
			# print("Average length of a stupid ogre tale: " + str(stupidogrelen / len(stupidogre[language])))
			length[language + "_stupidogre"] = round(stupidogrelen / len(stupidogre[language]))

			# print("Number of Jokes: " + str(len(jokes[language])))
			# print("Average length of a joke: " + str(jokeslen / len(jokes[language])))
			length[language + "_jokes"] = round(jokeslen / len(jokes[language]))

			# print("Number of Formula Folktales: " + str(len(formulatales[language])))
			# print("Average length of an formula tale: " + str(formulalen / len(formulatales[language])))
			length[language + "_formulatales"] = round(formulalen / len(formulatales[language]))
		except ZeroDivisionError:
			pass
		# print(length)
	with open("average_tale_length.txt", "w+", encoding="utf-8") as f:
		f.write(str(length))
	with open("average_sentence_length.txt", "w+", encoding="utf-8") as g:
		g.write(str(sentence_length_average))
	with open("average_title_length.txt", "w+", encoding="utf-8") as h:
		h.write(str(title_length_average))

	with open("min_max_tale_length.txt", "w+", encoding="utf-8") as file:
		file.write(str(min_max))

		# print("Number of Folktales without ATU: " + str(unknowncounter))
		# print("Number of Folktales in total: " + str(i))