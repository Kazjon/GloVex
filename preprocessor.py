import csv,gensim,logging,sys,os.path,multiprocessing, nltk, io, glove, itertools, pprint, argparse, scipy.spatial, glob
import cPickle as pickle
import numpy as np
from nltk.corpus import stopwords,wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from joblib import Parallel, delayed
from inflection import singularize
from prettytable import PrettyTable

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

	def load(self,preprocessed_path):
		with open(preprocessed_path,"rb") as pro_f:
			self.documents,self.word_occurrence, self.cooccurrence,self.dictionary, self.total_docs = pickle.load(pro_f)
			self.first_pass = False

	def preprocess(self,suffix=".preprocessed", no_below=0.001, no_above=0.5, force_overwrite = False):
		self.argstring = "_below"+str(no_below)+"_above"+str(no_above)
		preprocessed_path = self.filepath+self.argstring+suffix
		if not os.path.exists(preprocessed_path) or force_overwrite:
			logger.info(" ** Pre-processing started.")
			self.dictionary = gensim.corpora.Dictionary(self)
			logger.info("   **** Dictionary created.")
			self.dictionary.filter_extremes(no_below=max(2,no_below*self.total_docs),no_above=no_above,keep_n=None)
			self.word_occurrence = {k:0.0 for k in self.dictionary.token2id.keys()}
			logger.info("   **** Dictionary filtered.")
			self.documents = [self.dictionary.doc2bow(d) for d in self]
			logger.info("   **** BoW representations constructed.")
			self.calc_cooccurrence()
			logger.info("   **** Co-occurrence matrix constructed.")
			with open(preprocessed_path,"wb") as pro_f:
				pickle.dump((self.documents,self.word_occurrence, self.cooccurrence,self.dictionary, self.total_docs),pro_f)
		else:
			logger.info(" ** Existing pre-processed file found.  Rerun with --overwrite_preprocessing"+
						" if you did not intend to reuse it.")
			self.load(preprocessed_path)
		logger.info(" ** Pre-processing complete.")

	def calc_cooccurrence(self, normalise = False):
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
		if normalise:
			for wk,wv in self.dictionary.iteritems():
				self.cooccurrence[wk] = {wk2:float(wv2)/self.word_occurrence[wv] for wk2,wv2 in self.cooccurrence[wk].iteritems()}

def word_pair_surprise(w1_w2_cooccurrence, w1_occurrence, w2_occurrence, n_docs, offset = 1):
	# Offset is Laplacian smoothing
	w1_w2_cooccurrence = min(min(w1_occurrence,w2_occurrence),max(0,w1_w2_cooccurrence)) #Capped due to estimates being off sometimes
	p_w1_given_w2 = (w1_w2_cooccurrence + offset) / (w2_occurrence + offset)
	p_w1 = (w1_occurrence + offset) / (n_docs + offset)
	return p_w1_given_w2 / p_w1

def extract_document_cooccurrence_matrix(doc, coocurrence):
	cooc_mat = np.zeros([len(doc),len(doc)])
	for i1,i2 in itertools.combinations(range(len(doc)),2):
		d1 = doc[i1][0]
		d2 = doc[i2][0]
		cooc_mat[i1,i2] = coocurrence[d1][d2]
	return cooc_mat

def estimate_word_pair_cooccurrence(wk1, wk2, model, cooccurrence):
	# take dot product of vectors
	cooc = np.dot(model.W[wk1],model.ContextW[wk2]) + model.b[wk1] + model.ContextB[wk2]
	# correct for the rare feature scaling described in https://nlp.stanford.edu/pubs/glove.pdf
	if cooccurrence[wk1][wk2] < model.x_max:
		cooc *= 1.0/pow(cooccurrence[wk1][wk2] / model.x_max,model.alpha)
	return cooc[0]

def estimate_document_cooccurrence_matrix(doc, model, cooccurrence):
	cooc_mat = np.zeros([len(doc),len(doc)])
	for i1,i2 in itertools.combinations(range(len(doc)),2):
		d1 = doc[i1][0]
		d2 = doc[i2][0]
		cooc_mat[i1,i2] = estimate_word_pair_cooccurrence(d1, d2, model, cooccurrence)
	return np.triu(np.exp(cooc_mat), k=1)

def top_n_surps_from_doc(doc, model, cooccurrence, word_occurrence, dictionary, n_docs, top_n = 10):
	if len(doc):
		est_cooc_mat = estimate_document_cooccurrence_matrix(doc,model,cooccurrence)
		surps = document_cooccurrence_to_surprise(doc, est_cooc_mat, word_occurrence, dictionary, n_docs)
		surps.sort(key = lambda x: x[2], reverse=False)
		return surps[:min(top_n,len(surps))]
	else:
		return []

#Returns an ordered list of most-to-least surprising word combinations as (w1,w2,surprise) tuples
def document_cooccurrence_to_surprise(doc, cooc_mat, word_occurrence, dictionary, n_docs):
	surp_list = []
	it = np.nditer(cooc_mat+cooc_mat.T, flags=['multi_index'])
	while not it.finished:
		w1 = doc[it.multi_index[0]][0]
		w2 = doc[it.multi_index[1]][0]
		if not w1 == w2:
			s = word_pair_surprise(it[0], word_occurrence[dictionary[w1]], word_occurrence[dictionary[w2]], n_docs)
			surp_list.append((dictionary[w1], dictionary[w2],s))
		it.iternext()
	surp_list.sort(key = lambda x: x[2], reverse=False)
	return surp_list

#Return the surprises from the given list that have the most similar feature word (i.e. w1) to the one in the given surp.
def most_similar_features(surp, surp_list, model, n = 10):
	results = []
	for surp2 in surp_list:
		surp_feat = model.W[surp[0]]
		surp2_feat = model.W[surp2[0]]
		results.append([surp,surp2,scipy.spatial.distance.euclidean(surp_feat, surp2_feat)])
	results.sort(key = lambda x: x[2])
	return results[:n]

#Return the surprises from the given list that have the most similar context word(s) (i.e. w2) to the one in the given surp.
def most_similar_contexts(surp, surp_list, model, n = 10):
	results = []
	for surp2 in surp_list:
		surp_context = model.W[surp[1]]
		surp2_context = model.W[surp2[1]]
		results.append([surp,surp2,scipy.spatial.distance.euclidean(surp_context, surp2_context)])
	results.sort(key = lambda x: x[2])
	return results[:n]

#Return the surprises from the given list that have the most similar vector difference to the one in the given surp.
def most_similar_differences(surp, surp_list, model, n = 10):
	results = []
	for surp2 in surp_list:
		surp_diff = model.W[surp[0]] -  model.W[surp[1]]
		surp2_diff = model.W[surp2[0]] -  model.W[surp2[1]]
		results.append([surp,surp2,surp_diff,surp2_diff,scipy.spatial.distance.euclidean(surp_diff, surp2_diff)])
	results.sort(key = lambda x: x[4])
	return results[:n]

def glovex_model(filepath, argstring, cooccurrence, dims=100, alpha=0.75, x_max=100, force_overwrite = False, suffix = ".glovex"):
	model_path = filepath+argstring
	model_files = glob.glob(model_path+"_epochs*"+suffix)
	if not len(model_files) or force_overwrite:
		model = glove.Glove(cooccurrence, d=dims, alpha=alpha, x_max=x_max)
	else:
		highest_epochs = max([int(f.split("epochs")[1].split(".")[0]) for f in model_files])
		logger.info(" ** Existing model file found.  Re-run with --overwrite_model if you did not intend to reuse it.")
		with open(model_path+"_epochs"+str(highest_epochs)+suffix,"rb") as pro_f:
			model = pickle.load(pro_f)
	return model

def save_model(model,path,args,suffix=".glovex"):
	with open(path+args+suffix,"wb") as f:
		pickle.dump(model,f)

def estimate_document_surprise_pairs(doc, model, acm):
	est_cooc_mat = estimate_document_cooccurrence_matrix(doc,model,acm.cooccurrence)
	return document_cooccurrence_to_surprise(doc, est_cooc_mat, acm.word_occurrence, acm.dictionary, len(acm.documents))

def percentile_doc_surprise(doc, model, acm, percentile = 95):
	surps = estimate_document_surprise_pairs(doc, model, acm)
	return np.percentile([x[2] for x in surps], percentile)

def print_top_n_surps(model, acm):
	#Parallelised version -- takes forever, not sure why, leaving this for now
	#top_surps = Parallel(n_jobs=cores)(delayed(top_n_surps_from_doc)(doc,
	#																 model,
	#																 acm.cooccurrence,
	#																 acm.word_occurrence,
	#																 acm.dictionary,
	# 																 acm.total_docs)
	#								   								 for doc in acm.documents)
	#top_surps = sorted(itertools.chain.from_iterable(top_surps))
	#top_surps = top_surps[:10]

	top_surps = []
	for doc in acm.documents:
		if len(doc):
			top_surps += estimate_document_surprise_pairs(doc, model, acm)[:10]
			top_surps = list(set(top_surps))
			top_surps.sort(key = lambda x: x[2], reverse=False)
			top_surps = top_surps[:10]

	print "Top 10 surprising combos"
	w1s = []
	w2s = []
	w1_occs = []
	w2_occs = []
	est_surps = []
	est_coocs = []
	obs_coocs = []
	obs_surps = []
	for surp in top_surps:
		w1s.append(surp[0])
		w2s.append(surp[1])
		w1_occ = acm.word_occurrence[surp[0]]
		w2_occ = acm.word_occurrence[surp[1]]
		w1_occs.append(w1_occ)
		w2_occs.append(w2_occ)
		est_surps.append(surp[2])
		wk1 = acm.dictionary.token2id[surp[0]]
		wk2 = acm.dictionary.token2id[surp[1]]
		est_coocs.append(estimate_word_pair_cooccurrence(wk1, wk2, model, acm.cooccurrence))
		w1_w2_cooccurrence = acm.cooccurrence[wk1][wk2]
		obs_coocs.append(w1_w2_cooccurrence)
		obs_surp = word_pair_surprise(w1_w2_cooccurrence, w1_occ, w2_occ, len(acm.documents))
		obs_surps.append(obs_surp)

	tab = PrettyTable()
	tab.add_column("Word 1",w1s)
	tab.add_column("Word 2",w2s)
	tab.add_column("W1 occs", w1_occs)
	tab.add_column("W2 occs", w2_occs)
	tab.add_column("Obs. cooc",obs_coocs)
	tab.add_column("Obs. surp",obs_surps)
	tab.add_column("Est. cooc",est_coocs)
	tab.add_column("Est. surp",est_surps)
	tab.float_format = ".4"
	print tab


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run GloVeX on some text.")
	parser.add_argument("inputfile", help='The file path to work with (omit the ".csv")')
	parser.add_argument("--dims", default = 100, type=int, help="The number of dimensions in the GloVe vectors.")
	parser.add_argument("--epochs", default = 100, type=int, help="The number of epochs to train GloVe for.")
	parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate for SGD.")
	parser.add_argument("--glove_x_max", default = 100.0, type=float, help="x_max parameter in GloVe.")
	parser.add_argument("--glove_alpha", default = 0.75, type=float, help="alpha parameter in GloVe.")
	parser.add_argument("--no_below", default = 0.001, type=float,
						help="Min fraction of documents a word must appear in to be included.")
	parser.add_argument("--no_above", default = 0.75, type=float,
						help="Max fraction of documents a word can appear in to be included.")
	parser.add_argument("--overwrite_model", action="store_true",
						help="Ignore (and overwrite) existing .glovex file.")
	parser.add_argument("--overwrite_preprocessing", action="store_true",
						help="Ignore (and overwrite) existing .preprocessed file.")

	args = parser.parse_args()
	acm = ACMDL_DocReader(args.inputfile)
	acm.preprocess(no_below=args.no_below, no_above=args.no_above, force_overwrite=args.overwrite_preprocessing)
	model = glovex_model(args.inputfile, acm.argstring, acm.cooccurrence, args.dims, args.glove_alpha, args.glove_x_max,
						 args.overwrite_model)
	logger.info(" ** Training GloVe")
	init_step_size = args.learning_rate
	step_size_decay = 10.0
	cores = multiprocessing.cpu_count() - 2
	for epoch in range(args.epochs):
		err = model.train(workers=cores, batch_size=1000, step_size=init_step_size/(1.0+epoch/step_size_decay))
		logger.info("   **** Training GloVe: epoch %d, error %.5f" % (epoch, err))
		if epoch and epoch % 50 == 0:
			print_top_n_surps(model, acm)
			save_model(model, args.inputfile, "_below"+str(args.no_below)+"_above"+str(args.no_above)+"_epochs"+str(epoch))
	save_model(model, args.inputfile, "_below"+str(args.no_below)+"_above"+str(args.no_above)+"_epochs"+str(epoch))