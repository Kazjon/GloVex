import csv,gensim,logging,sys,os.path,multiprocessing, nltk, io, argparse, glob
import s_glove
from glove import glove
import cPickle as pickle
import numpy as np
from nltk.corpus import stopwords,wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from joblib import Parallel, delayed
from inflection import singularize
from prettytable import PrettyTable
import evaluate
import random
import fisher

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

def significance(w1_occurrence, w2_occurrence, cooccurrences, n_docs):
	table = [[cooccurrences,w2_occurrence-cooccurrences],[w1_occurrence,(n_docs-w1_occurrence)]]
	oddsratio, pvalue = fisher_exact(table, alternative="less")
	return pvalue

''' Old, slow scipy-based significance test.
def significance_on_tuple(sig_tuple):
	_, _, w1_occurrence, w2_occurrence, cooccurrences, n_docs = sig_tuple
	table = [[cooccurrences,w2_occurrence-cooccurrences],[w1_occurrence,(n_docs-w1_occurrence)]]
	oddsratio, pvalue = fisher_exact(table, alternative="less")
	return pvalue
'''

def significance_on_tuple(sig_tuple):
	_, _, w1_occurrence, w2_occurrence, cooccurrences, n_docs = sig_tuple
	pvalue = fisher.pvalue(cooccurrences,w2_occurrence-cooccurrences,w1_occurrence,(n_docs-w1_occurrence))
	return pvalue.left_tail

class DocReader(object):
	def __init__(self,path,famcat_path, run_name=None, use_sglove=False):
		self.filepath = path
		self.run_name = run_name
		self.famcat_filepath = famcat_path
		self.total_words = 0
		self.total_docs = 0
		self.stop = set(stopwords.words("english"))
		self.tokeniser = RegexpTokenizer(r'\w+')
		self.first_pass = True
		self.finalised = False
		self.doc_ids = []
		self.doc_titles = []
		self.doc_raws = []
		self.doc_famcats = []
		self.per_fc_keys_to_all_keys = {}
		self.all_keys_to_per_fc_keys = {}
		self.lem = WordNetLemmatizer()
		self.famcats = []
		self.docs_per_fc = {}
		self.cooccurrence_p_values = {}
		self.use_sglove = use_sglove

	def __iter__(self):
		raise NotImplementedError

	def load(self,preprocessed_path):
		with open(preprocessed_path,"rb") as pro_f:
			self.documents,self.word_occurrence, self.cooccurrence,self.dictionary, self.total_docs, self.doc_ids, self.doc_titles, self.doc_raws, self.doc_famcats, self.per_fc_keys_to_all_keys, self.all_keys_to_per_fc_keys, self.docs_per_fc, self.cooccurrence_p_values, self.use_sglove = pickle.load(pro_f)
			self.famcats = self.cooccurrence.keys()
			self.first_pass = False

	def preprocess(self,suffix=".preprocessed", no_below=0.001, no_above=0.5, force_overwrite = False):
		if self.run_name is not None:
			self.argstring = "_"+self.run_name+"_below"+str(no_below)+"_above"+str(no_above)
		else:
			self.argstring = "_below"+str(no_below)+"_above"+str(no_above)
		preprocessed_path = self.filepath+self.argstring+suffix
		if not os.path.exists(preprocessed_path) or force_overwrite:
			logger.info(" ** Pre-processing started.")
			self.dictionary = gensim.corpora.Dictionary(self)
			logger.info("   **** Dictionary created.")
			self.dictionary.filter_extremes(no_below=max(2,no_below*self.total_docs),no_above=no_above,keep_n=None)
			logger.info("   **** Dictionary filtered.")
			self.documents = [self.dictionary.doc2bow(d) for d in self]
			logger.info("   **** BoW representations constructed.")
			self.calc_cooccurrence()
			logger.info("   **** Co-occurrence matrix constructed.")
			if self.use_sglove:
				self.calc_cooccurrence_significance_parallel()
				logger.info("   **** Co-occurrence signficance matrix calculated.")
			with open(preprocessed_path,"wb") as pro_f:
				pickle.dump((self.documents,self.word_occurrence, self.cooccurrence,self.dictionary, self.total_docs, self.doc_ids, self.doc_titles, self.doc_raws, self.doc_famcats, self.per_fc_keys_to_all_keys, self.all_keys_to_per_fc_keys, self.docs_per_fc, self.cooccurrence_p_values, self.use_sglove),pro_f)
		else:
			logger.info(" ** Existing pre-processed file found.  Rerun with --overwrite_preprocessing"+
						" if you did not intend to reuse it.")
			self.load(preprocessed_path)
		logger.info(" ** Pre-processing complete.")

	#Note: Normalisation not implemented for personalised version (w/ famcats)
	def calc_cooccurrence(self, normalise = False):
		if self.famcat_filepath is None:
			self.word_occurrence = {k:0.0 for k in self.dictionary.token2id.keys()}
			self.cooccurrence = {wk:{} for wk in range(len(self.dictionary))}
			for doc in self.documents:
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

		else:
			self.cooccurrence = {}
			self.word_occurrence = {}

			for doc,doc_fcs in zip(self.documents,self.doc_famcats):
				for wk,wc in doc:
					self.total_words += wc
					for fc in doc_fcs:
						if fc not in self.cooccurrence.keys():
							self.cooccurrence[fc] = {}
							self.word_occurrence[fc] = {}
							self.per_fc_keys_to_all_keys[fc] = {}
							self.all_keys_to_per_fc_keys[fc] = {}
							self.docs_per_fc[fc] = 0
						self.docs_per_fc[fc] += 1
						try:
							self.word_occurrence[fc][self.dictionary[wk]] += 1.0
						except KeyError:
							self.word_occurrence[fc][self.dictionary[wk]] = 1.0
						if wk not in self.all_keys_to_per_fc_keys[fc].keys():
							new_id = len(self.all_keys_to_per_fc_keys[fc])
							self.per_fc_keys_to_all_keys[fc][new_id] = wk
							self.all_keys_to_per_fc_keys[fc][wk] = new_id
						if self.all_keys_to_per_fc_keys[fc][wk] not in self.cooccurrence[fc].keys():
							self.cooccurrence[fc][self.all_keys_to_per_fc_keys[fc][wk]] = {}
						already_rekeyed_words = set(self.all_keys_to_per_fc_keys[fc].keys())
						for wk2,wc2 in doc:
							if wk != wk2:
								if wk2 not in already_rekeyed_words:
									new_id = len(self.all_keys_to_per_fc_keys[fc])
									self.all_keys_to_per_fc_keys[fc][wk2] = new_id
									self.per_fc_keys_to_all_keys[fc][new_id] = wk2
								try:
									self.cooccurrence[fc][self.all_keys_to_per_fc_keys[fc][wk]][self.all_keys_to_per_fc_keys[fc][wk2]] += 1.0
								except KeyError:
									self.cooccurrence[fc][self.all_keys_to_per_fc_keys[fc][wk]][self.all_keys_to_per_fc_keys[fc][wk2]] = 1.0

			self.famcats = self.cooccurrence.keys()

	def calc_cooccurrence_significance(self):
		if len(self.famcats):
			for fc in self.famcats:
				self.cooccurrence_p_values[fc] = {}
				for w1 in self.cooccurrence[fc].keys():
					if w1 not in self.cooccurrence_p_values[fc].keys():
						self.cooccurrence_p_values[fc][w1] = {}
					for w2 in self.cooccurrence[fc].keys():
						if w1 != w2:
							self.cooccurrence_p_values[fc][w1][w2] = significance(
								self.word_occurrence[fc][self.dictionary[self.per_fc_keys_to_all_keys[fc][w1]]],
								self.word_occurrence[fc][self.dictionary[self.per_fc_keys_to_all_keys[fc][w2]]],
								self.cooccurrence[fc][w1][w2] if w2 in self.cooccurrence[fc][w1] else 0,
								self.docs_per_fc[fc]
							)
		else:
			self.cooccurrence_p_values = {}
			for w1 in self.cooccurrence.keys():
				if w1 not in self.cooccurrence_p_values.keys():
					self.cooccurrence_p_values[w1] = {}
				for w2 in self.cooccurrence.keys():
					if w1 != w2:
						self.cooccurrence_p_values[w1][w2] = significance(
							self.word_occurrence[self.dictionary[w1]],
							self.word_occurrence[self.dictionary[w2]],
							self.cooccurrence[w1][w2] if w2 in self.cooccurrence[w1] else 0,
							self.total_docs
						)

	def calc_cooccurrence_significance_parallel(self):
		if len(self.famcats):
			for fc in self.famcats:
				self.cooccurrence_p_values[fc] = {}
				for w1 in self.cooccurrence[fc].keys():
					if w1 not in self.cooccurrence_p_values[fc].keys():
						self.cooccurrence_p_values[fc][w1] = {}
					for w2 in self.cooccurrence[fc].keys():
						if w1 != w2:
							self.cooccurrence_p_values[fc][w1][w2] = significance(
								self.word_occurrence[fc][self.dictionary[self.per_fc_keys_to_all_keys[fc][w1]]],
								self.word_occurrence[fc][self.dictionary[self.per_fc_keys_to_all_keys[fc][w2]]],
								self.cooccurrence[fc][w1][w2] if w2 in self.cooccurrence[fc][w1] else 0,
								self.docs_per_fc[fc]
							)
		else:
			sigs_to_compute = []
			self.cooccurrence_p_values = {}
			for w1 in self.cooccurrence.keys():
				if w1 not in self.cooccurrence_p_values.keys():
					self.cooccurrence_p_values[w1] = {}
				for w2 in self.cooccurrence.keys():
					if w1 != w2:
						sigs_to_compute.append((w1,
												w2,
												self.word_occurrence[self.dictionary[w1]],
												self.word_occurrence[self.dictionary[w2]],
												self.cooccurrence[w1][w2] if w2 in self.cooccurrence[w1] else 0,
												self.total_docs))
			computed_sigs = Parallel(n_jobs=-1)(delayed(significance_on_tuple)(sig) for sig in sigs_to_compute)
			for sig,p in zip(sigs_to_compute,computed_sigs):
				self.cooccurrence_p_values[sig[0]][sig[1]] = p

class ACMDL_DocReader(DocReader):
	def __init__(self,path, title_column, text_column, id_column, famcat_path=None, run_name=None, use_sglove=False):
		self.title_column = title_column
		self.text_column = text_column
		self.id_column = id_column
		DocReader.__init__(self,path,famcat_path, run_name=run_name, use_sglove=use_sglove)

	def __iter__(self):
		if self.first_pass and self.famcat_filepath is not None:
			with io.open(self.famcat_filepath+".csv",mode="r",encoding='ascii',errors="ignore") as famcat_file:
				reader = csv.reader(famcat_file)
				#famcats = {row[0]:(row[1:] if len(row) > 1 else []) for row in reader}

				#Hacks for working with fake author-based famcats
				famcats = {row[0]:([n[0] for n in row[1:] if len(n)] if len(row) > 1 else ["None"]) for row in reader}
				#famcats = {row[0]:["1"] if random.random() > 0.5 else ["1","2"] for row in reader}
		with io.open(self.filepath+".csv",mode="r",encoding='ascii',errors="ignore") as i_f:
			for row in csv.DictReader(i_f):
				docwords = [singularize(w) for w in self.tokeniser.tokenize((row[self.title_column]+" "+row[self.text_column]).lower()) if w not in self.stop]

				#tag+lemmatize
				#docwords = nltk.pos_tag(self.tokeniser.tokenize(row["Abstract"].lower()))
				#docwords = [self.lem.lemmatize(w,pos=get_wordnet_pos(t)) for w,t in docwords if w not in self.stop]

				if self.first_pass:
					self.total_docs += 1.0
					self.doc_ids.append(row[self.id_column])
					self.doc_titles.append(row[self.title_column])
					self.doc_raws.append(row[self.text_column])
					if self.famcat_filepath is not None:
						self.doc_famcats.append(famcats[row[self.id_column]])
				yield docwords
		self.first_pass = False

class WikiPlot_DocReader(DocReader):
	def __init__(self,path):
		DocReader.__init__(self,path)

	def __iter__(self):
		with io.open(self.filepath,mode="r",encoding='ascii',errors="ignore") as i_f:
			if self.first_pass:
				doc_raw = ""
				t_f = io.open(self.filepath+"_titles",mode="r",encoding='ascii',errors="ignore")
			docwords = []
			for line in i_f.readlines():
				if line[:5] == "<EOS>":
					if self.first_pass:
						self.total_words += len(docwords)
						self.doc_ids.append(self.total_docs)
						self.total_docs += 1
						self.doc_titles.append(t_f.readline())
						self.doc_raws.append(doc_raw)
						doc_raw = ""
					yield docwords
					docwords = []
				else:
					docwords += [singularize(w) for w in self.tokeniser.tokenize(line) if w not in self.stop]
					if self.first_pass:
						doc_raw += line
		if self.first_pass:
			t_f.close()
		self.first_pass = False

def glovex_model(filepath, argstring, cooccurrence, p_values, dims=100, alpha=0.75, x_max=100, force_overwrite = False, suffix = ".glovex", use_sglove=False):
	model_path = filepath+argstring
	model_files = glob.glob(model_path+"_epochs*"+suffix)
	if not len(model_files) or force_overwrite:
		if use_sglove:
			model = s_glove.Glove(cooccurrence, p_values, d=dims, alpha=alpha)
		else:
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

def load_personalised_models(filepath, docreader):
	models = []
	for fc in docreader.famcats:
		models.append(glovex_model(filepath, docreader.argstring+"_fc"+str(fc), docreader.cooccurrence[fc]))
	return models

def print_top_n_surps(model, acm):

	top_surps = []
	for doc in acm.documents:
		if len(doc):
			top_surps += evaluate.estimate_document_surprise_pairs(doc, model, acm)[:10]
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
		est_coocs.append(evaluate.estimate_word_pair_cooccurrence(wk1, wk2, model, acm.cooccurrence))
		w1_w2_cooccurrence = acm.cooccurrence[wk1][wk2]
		obs_coocs.append(w1_w2_cooccurrence)
		obs_surp = evaluate.word_pair_surprise(w1_w2_cooccurrence, w1_occ, w2_occ, len(acm.documents))
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
	parser.add_argument("--dataset", default="acm",type=str, help="Which dataset to assume.  Currently 'acm' or 'plots'")
	parser.add_argument("--name", default=None, type=str, help= "Name of this run (used when saving files.)")
	parser.add_argument("--dims", default = 100, type=int, help="The number of dimensions in the GloVe vectors.")
	parser.add_argument("--epochs", default = 26, type=int, help="The number of epochs to train GloVe for.")
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
	parser.add_argument("--use_sglove", action="store_true",
						help="Use the modified version of the GloVe algorithm that favours surprise rather than co-occurrence.")
	parser.add_argument("--familiarity_categories", default=None, type=str,
						help='The (optional) path to the file containing IDs and familiarity categories (omit the ".csv")')

	args = parser.parse_args()
	if args.dataset == "acm":
		reader = ACMDL_DocReader(args.inputfile, "title", "abstract", "ID", famcat_path=args.familiarity_categories, run_name=args.name, use_sglove=args.use_sglove)
	elif args.dataset == "plots":
		reader = WikiPlot_DocReader(args.inputfile)
	else:
		logger.info("You've tried to load a dataset we don't know about.  Sorry.")
		sys.exit()
	reader.preprocess(no_below=args.no_below, no_above=args.no_above, force_overwrite=args.overwrite_preprocessing)
	if args.familiarity_categories is None:
		model = glovex_model(args.inputfile, reader.argstring, reader.cooccurrence, reader.cooccurrence_p_values, args.dims, args.glove_alpha, args.glove_x_max,
							 args.overwrite_model, use_sglove=args.use_sglove)
		logger.info(" ** Training GloVe")
		init_step_size = args.learning_rate
		step_size_decay = 10.0
		cores = multiprocessing.cpu_count() - 2
		for epoch in range(args.epochs):
			err = model.train(workers=cores, batch_size=1000, step_size=init_step_size/(1.0+epoch/step_size_decay))
			logger.info("   **** Training GloVe: epoch %d, error %.5f" % (epoch, err))
			if epoch and epoch % 25 == 0:
				print_top_n_surps(model, reader)
				save_model(model, args.inputfile, "_below"+str(args.no_below)+"_above"+str(args.no_above)+"_epochs"+str(epoch))
		save_model(model, args.inputfile, "_below"+str(args.no_below)+"_above"+str(args.no_above)+"_epochs"+str(epoch))
	else:
		for fc,fc_cooccurrence in reader.cooccurrence.iteritems():
			model = glovex_model(args.inputfile, reader.argstring+"_fc"+fc, fc_cooccurrence, reader.cooccurrence_p_values[fc], args.dims, args.glove_alpha, args.glove_x_max,
								 args.overwrite_model, use_sglove=args.use_sglove)
			logger.info(" ** Training GloVe for "+fc)
			init_step_size = args.learning_rate
			step_size_decay = 10.0
			cores = multiprocessing.cpu_count() - 1
			for epoch in range(args.epochs):
				err = model.train(workers=cores, batch_size=100, step_size=init_step_size/(1.0+epoch/step_size_decay))
				logger.info("   **** Training GloVe for "+fc+": epoch %d, error %.5f" % (epoch, err))
				if epoch and epoch % 25 == 0:
					print_top_n_surps(model, reader)
					save_model(model, args.inputfile, reader.argstring+"_epochs"+str(epoch))
			save_model(model, args.inputfile, reader.argstring+"_epochs"+str(epoch))