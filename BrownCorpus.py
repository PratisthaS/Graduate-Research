import re
import time
import os
from math import log
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
from builtins import sorted
from nltk.corpus import brown
from nltk import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
from gensim import models, corpora
from gensim.test.utils import datapath
# from Build_Corpus import Build_Corpus
from sklearn.cluster import AgglomerativeClustering
from nltk.corpus.reader.plaintext import PlaintextCorpusReader


class LDA_Stability_Check:

	@staticmethod
	def print_welcome_text():
		print("\n# ------------------------------------------------------------------- #")
		print("#        Latent Dirichlet Allocation - Stability Issue                #")
		print("# ------------------------------------------------------------------- #\n")
	# End of print_welcome_text method

	def __init__(self):

		self.print_welcome_text()

		self.NUM_TOPICS = 10     # Number of possible topics
		self.NUM_REP_WORDS = 10  # Number of representative words
		self.data = []           # Data Structure to hold words in the corpus
		self.data = brown.sents()
		# sys.stdout = open('our_corpus_completeLinkage.txt', 'wt')

	# End of constructor


	@staticmethod
	def remove_stopwords(text):
		STOPWORDS = stopwords.words('english')
		tokenized_text = word_tokenize(text.strip().lower())
		cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]

		return cleaned_text
	# End of remove_stopwords method

	def bhattacharya_difference(self, pd_df, colp, colq):
		# df[col_name].mean() gives the mean of a dataframe column
		# df[col_name].std() gives the std of a dataframe column

		if colp != colq:
			meanp = pow(pd_df[colp].mean(), 2.0)
			meanq = pow(pd_df[colq].mean(), 2.0)
			stdp = pow(pd_df[colp].std(), 2.0)
			stdq = pow(pd_df[colq].std(), 2.0)
			Dist_B = 0.25 * log(0.25*((stdp / stdq) + (stdq / stdp) + 2)) + 0.25 * (pow(meanp - meanq, 2.0) / (stdp + stdq))
			return Dist_B
		else:
			return 0

	def clean_text(self):
		tokenized_data = []
		count = 0
		for aList in self.data:
			cleaned_text = self.remove_stopwords(aList)
			# cleaned_text = self.remove_special_characters(cleaned_text, remove_digits=False)
			count = count + len(cleaned_text)
			tokenized_data.append(cleaned_text)

		print("After pre-proccessing: {} words ".format(count))
		return tokenized_data
	# End of clean_text method

	def run_LDA(self):

		self.data = brown.words(categories=['news', 'editorial', 'reviews'])
		tokenized_data = self.clean_text()  # tokenize & then do pre-processing



	# End of run_LDA method

# End of class LDA_Stability_Check

start_time = time.time()
Obj = LDA_Stability_Check()
Obj.run_LDA();
print("--- %s seconds ---" % (time.time() - start_time))


