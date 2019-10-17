from collections import defaultdict

class Story:

	def __init__(self, title, index, atu_type, language, text):
		#(Titel,Datenbanknummer,ATU Nummer,Sprache,Text)
		self.title = title
		self.index = index
		self.atu_type = atu_type
		self.language = language
		self.text = text

	def __str__(self):
		# für backwardscompatibility
		return str((self.title, self.index, self.atu_type, self.language, self.text))


def read_corpus(path):
	with open(path, "r", encoding = 'utf-8') as f:
		s = f.readline()
		data = eval(s.replace("<class 'list'>", 'list')) # für defaultdict typen
		return data

def write(path, content):
	with open("clean/" + path + '_clean.txt',"w+",encoding = 'utf-8') as f:
		# schreibe Titel und Text in einzelnene Zeilen separiert durch Leerzeile
		for story in content:
			f.write(story.title)
			f.write('\n')
			f.write(story.text)
			f.write('\n\n')




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
	animallen = 0
	magiclen = 0
	religiouslen = 0
	realisticlen = 0
	stupidogrelen = 0
	jokeslen = 0
	formulalen = 0


	# corpora = read_corpus('corpora.txt')
	# corpora_no_atu = read_corpus('corpora_no_atu')
	corpora_default = read_corpus('corpora.txt')
	for language in corpora_default:
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

				elif 300 <= atu <= 749:
					magictales[language].append(story)
					magiclen += len(story.text.split())

				elif 750 <= atu <= 849:
					religioustales[language].append(story)
					religiouslen += len(story.text.split())

				elif 850 <= atu <= 999:
					realistictales[language].append(story)
					realisticlen += len(story.text.split())

				elif 1000 <= atu <= 1199:
					stupidogre[language].append(story)
					stupidogrelen += len(story.text.split())

				elif 1200 <= atu <= 1999:
					jokes[language].append(story)
					jokeslen += len(story.text.split())

				elif 2000 <= atu <= 2399:
					formulatales[language].append(story)
					formulalen += len(story.text.split())


		write(language + '_animaltales' , animaltales[language])
		write(language + '_magictales' , magictales[language])
		write(language + '_religioustales' , religioustales[language])
		write(language + '_stupidogre' , stupidogre[language])
		write(language + '_jokes' , jokes[language])
		write(language + '_formulatales' , formulatales[language])
		write(language + '_unknowntexts' , unknowntexts[language])
		

		with open('clean/' + language + '_corpora.txt', 'w+', encoding='utf-8') as f:
			f.write(str(corpora[language]))

		with open('clean/' + language + '_corpora_no_atu.txt', 'w+', encoding='utf-8') as f:
			f.write(str(corpora_no_atu[language]))