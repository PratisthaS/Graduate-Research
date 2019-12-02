
#to log the time taken to run the program
import time
import nltk
# word tokenizer to get individual words in the text
from nltk.tokenize import word_tokenize
# to get the POS of a particular word
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import re

import spacy


#training file



class preProcessor:

    def __init__(self):
        #Data structure to hold words in the training set
        self.data = []

    def preProcess(sent):
        sent = nltk.word_tokenize(sent)
        sent = nltk.pos_tag(sent)
        return sent

    def removeStopwords(self,text):
        STOPWORDS = stopwords.words('english')
        tokenized_text = word_tokenize(text)
        clean_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]

        return clean_text

    def cleanText(self):
        count = 0
        tokenized_data = []
        for i in self.data:
            clean_text = self.removeStopwords(i)
            count = count + len(clean_text)
            tokenized_data.append(clean_text)

        print("After pre-processing: {} words ".format(count) +"\n")
        return tokenized_data


    def spacyNer(self):
        self.nlp = spacy.load("emptyModel")
        self.nlp.max_length = 2400000
        #train_file = open("Dataset/1837FullText.txt", "r")
        #content = train_file.read()
        doc = self.nlp(self.clean_string)
        #displacy.serve(doc,style="ent")
        i =0;

        #labels = [x.label_ for x in doc.ents]
        #Counter(labels)
        for ent in doc.ents:
            if (ent.label_ == "GPE"):
                i = i+1;
                print(i, ent.text, ent.start_char, ent.end_char, ent.label_)



    def createDataset(self):
        #train_file = open("Dataset/1830FullText.txt", "r");
        train_file = open("Dataset/1837FullText.txt", "r");
        #train_file=open("Dataset/sampleTest.txt","r", encoding="utf8")
        content = train_file.read()
        tokenizer = RegexpTokenizer(r'\w+')
        self.data = tokenizer.tokenize(content)
        tokenized_data = self.cleanText()
        self.clean_list = [x for x in tokenized_data if x != []]
        self.clean_string = ' '.join(map(str,self.clean_list))
        self.spacyNer()

        """
        lines = train_file.readlines()
        i =0;
        for line in lines:
            if line.rstrip():
                i=i+1;
                print(i , line)
        """


start_time = time.time()
Obj = preProcessor()
Obj.createDataset()


print("--- %s seconds ---" % (time.time() - start_time))


