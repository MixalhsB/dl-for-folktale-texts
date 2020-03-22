# Deep Learning for the Processing and Interpretation of Folktale Texts

Our software project 19/20 was about the processing of folktale text while using deep learning methods. Our team worked in two groups: one of them classified folktales into different types like e.g. animal tales or jokes and the second group tried to artificially generate new folktales. Please have a look at the respective subsections for further information.

**1 Classification** 

TODO add description here...

**2 Generation**

**2.1 How it works**

To generate folktales one has to call the use_models.py file from the console with Python 3 and all the requirements that are listed in the file _requirements.txt_. Then, one can enter a language:  either English or German.  Afterwards, the user is asked to select a type of tale that he wishes to be generated.  To select a type, he has to only type in the first letter of the tale that is insquare brackets.  A title and a text will then be generated and displayed in the console.

The general idea is that a neural network architecture gets a sequence of words as input, like for example "The cat chases the", and then has to predict the next word that has the largest probability in this context ("cat"). Our model is based on two Long Short-Term Memory cells and can generate a desired title and text of a certain type of folktale in either English or German.
Of course, this model does not produce syntactically and semantically correct stories, as this task would be too hard for a software project like this. Nevertheless, it is able to produce somewhat senseful content and it helped us a lot to better understand deep learning methods and neural language models.

**2.2 Organization of the files and directories**

main directory dl-for-foltale-texts:

- TODO add descriptions here


directory generierung:

- pretrained models are stored in the _generierung/models_ directory
- pretrained tokenizers are stored in the _generierung/tokenizer_ directory
- generated sequences that are necessary to generate text in the _use_models.py_ file are stored in the _generierung/sequence_ directory
- _prepare_data_for_generating.py_ is used to load and clean folktales to then shape them into sequences
- _train_model_text.py_ was used to train the models that generate folktales
- _train_models_title.py_ was used to train the models that generate titles
- TODO add further file descriptions here...
