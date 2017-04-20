import csv,gensim,logging,sys,os.path,multiprocessing, nltk, io, glove, itertools, pprint
import cPickle as pickle
import numpy as np
from nltk.corpus import stopwords,wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from joblib import Parallel, delayed
from inflection import singularize

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("glovex")

def get_wordnet_pos(treebank_tag):
	if treebank_tag.startswith('J'):
		return wordnet.ADJ
	elif treebank_tag.startswith('V'):
		return wordnet.VERB
	elif treebank_tag.startswith('N'):
		return wordnet.NOUN
	elif treebank_tag.startswith('R'):
		return wordnet.ADV
	else:
		return wordnet.NOUN

class ACMDL_DocReader(object):
	def __init__(self,path):
		self.filepath = path
		self.total_words = 0
		self.total_docs = 0
		self.stop = set(stopwords.words("english"))
		self.tokeniser = RegexpTokenizer(r'\w+')
		self.first_pass = True
		self.finalised = False

	def __iter__(self):
		lem = WordNetLemmatizer()
		with io.open(self.filepath+".csv",mode="r",encoding='ascii',errors="ignore") as i_f:
			for row in csv.DictReader(i_f):
				#Default
				#docwords = [w for w in self.tokeniser.tokenize(row["Abstract"].lower()) if w not in self.stop]

				#Singularise
				docwords = [singularize(w) for w in self.tokeniser.tokenize(row["Abstract"].lower()) if w not in self.stop]

				#tag+lemmatize
				#docwords = nltk.pos_tag(self.tokeniser.tokenize(row["Abstract"].lower()))
				#docwords = [lem.lemmatize(w,pos=get_wordnet_pos(t)) for w,t in docwords if w not in self.stop]

				if self.first_pass:
					self.total_words += len(docwords)
					self.total_docs += 1

				yield docwords
		self.first_pass = False

	def preprocess(self,suffix=".preprocessed", no_below=0.001, no_above=0.5):
		preprocessed_path = self.filepath+"_below"+str(no_below)+"_above"+str(no_above)+suffix
		if not os.path.exists(preprocessed_path):
			logger.info(" ** Pre-processing started.")
			self.dictionary = gensim.corpora.Dictionary(self)
			self.word_occurrence = {k:0.0 for k in self.dictionary.token2id.keys()}
			logger.info("   **** Dictionary created.")
			self.dictionary.filter_extremes(no_below=max(2,no_below*self.total_docs),no_above=no_above,keep_n=None)
			logger.info("   **** Dictionary filtered.")
			#logger.info(" **** Co-occurrence pruned.")
			self.documents = [self.dictionary.doc2bow(d) for d in self]
			logger.info("   **** BoW representations constructed.")
			self.calc_cooccurrence()
			logger.info("   **** Co-occurrence matrix constructed.")
			with open(preprocessed_path,"wb") as pro_f:
				pickle.dump((self.documents,self.cooccurrence,self.dictionary),pro_f)
		else:
			logger.info(" ** Pre-existing pre-processed file found.  Remove "+preprocessed_path+
						" and re-run if you did not intend to reuse it.")
			with open(preprocessed_path,"rb") as pro_f:
				self.documents,self.cooccurrence,self.dictionary = pickle.load(pro_f)
				self.first_pass = False
		logger.info(" ** Pre-processing complete.")

	def calc_cooccurrence(self):
		self.cooccurrence = {wk:{} for wk in range(len(self.dictionary))}
		for doc in self.documents:
			self.total_docs += 1.0
			for wk,wc in doc:
				self.total_words += wc
				self.word_occurrence[self.dictionary[wk]] += 1.0
				for wk2,wc2 in doc:
					if wk != wk2:
						try:
							self.cooccurrence[wk][wk2] += 1.0
						except KeyError:
							self.cooccurrence[wk][wk2] = 1.0

def extract_document_cooccurrence_matrix(doc, coocurrence):
	cooc_mat = np.zeros([len(doc),len(doc)])
	for i1,i2 in itertools.combinations(range(len(doc)),2):
		d1 = doc[i1][0]
		d2 = doc[i2][0]
		cooc_mat[i1,i2] = coocurrence[d1][d2]
	return cooc_mat

def eval_document_surprise_matrix(doc, model, cooccurrence):
	surp_mat = np.zeros([len(doc),len(doc)])
	for i1,i2 in itertools.combinations(range(len(doc)),2):
		d1 = doc[i1][0]
		d2 = doc[i2][0]
		# take dot product of vectors
		surp_mat[i1,i2] = np.dot(model.W[d1],model.ContextW[d2]) + model.b[d1] + model.ContextB[d2]
		# correct for the rare feature scaling described in https://nlp.stanford.edu/pubs/glove.pdf
		if cooccurrence[d1][d2] < model.x_max:
			surp_mat[i1,i2] *= 1.0/pow(cooccurrence[d1][d2] / model.x_max,model.alpha)
	return surp_mat

if __name__ == "__main__":
	inputfile = sys.argv[1]
	acm = ACMDL_DocReader(inputfile)
	acm.preprocess(no_below=0.0001, no_above=1)
	model = glove.Glove(acm.cooccurrence, d=500, alpha=0.75, x_max=10.0)
	logger.info(" ** Training GloVe")
	init_step_size = 0.1
	step_size_decay = 10
	doc = acm.documents[0]
	for epoch in range(20):
		err = model.train(workers=multiprocessing.cpu_count() - 2, batch_size=1000, step_size=init_step_size/(1+epoch/step_size_decay))
		logger.info("   **** Training GloVe: epoch %d, error %.5f" % (epoch, err))
		surp_mat = eval_document_surprise_matrix(acm.documents[0],model,acm.cooccurrence)
		cooc_mat = extract_document_cooccurrence_matrix(acm.documents[0],acm.cooccurrence)
		print "Surprise Matrix - log(Cooccurrence Matrix)"
		print np.array_str(np.subtract(surp_mat,np.log(cooc_mat)),max_line_width=200)
	#print model.W[acm.dictionary.token2id["computer"]]

'''
	surp_mat = eval_document_surprise_matrix(acm.documents[0],model,acm.cooccurrence)
	print "Surprise Matrix"
	print np.array_str(surp_mat,max_line_width=200)
	cooc_mat = extract_document_cooccurrence_matrix(acm.documents[0],acm.cooccurrence)
	print "Cooccurrence Matrix"
	print np.array_str(cooc_mat,max_line_width=200)
	print "Surprise Matrix - Cooccurrence Matrix"
	print np.array_str(np.subtract(surp_mat,cooc_mat),max_line_width=200)
	print "Surprise Matrix - log(Cooccurrence Matrix)"
	print np.array_str(np.subtract(surp_mat,np.log(cooc_mat)),max_line_width=200)
	'''