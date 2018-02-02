import argparse, logging, scipy, itertools, sys, multiprocessing

import preprocessor

import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("glovex")

def eval_dataset_surprise(model, reader, top_n_per_doc = 0, log_every=1000, ignore_order=True):
	logger.info("  ** Evaluating dataset.")
	dataset_surps = []
	count = 0
	# for id,title,doc,raw_doc in zip(reader.doc_ids, reader.doc_titles, reader.documents, reader.doc_raws):
	for id, doc, raw_doc in zip(reader.doc_ids, reader.documents, reader.doc_raws):
		if count and count % log_every == 0:
			logger.info("    **** Evaluated "+str(count)+" documents.")
		if len(doc):
			surps = estimate_document_surprise_pairs(doc, model, reader.cooccurrence, reader.word_occurrence, reader.dictionary, reader.documents, use_sglove=reader.use_sglove, ignore_order=ignore_order)
			if top_n_per_doc and len(surps) > top_n_per_doc:
				surps = surps[:top_n_per_doc]
			# dataset_surps.append({"id": id,"title":title,"raw":raw_doc, "surprises":surps, "surprise": document_surprise(surps)})
			dataset_surps.append({"id": id,"raw":raw_doc, "surprises":surps, "surprise": document_surprise(surps)})
		else:
			# dataset_surps.append({"id": id,"title":title,"raw":raw_doc, "surprises":[], "surprise": float("inf")})
			dataset_surps.append({"id": id,"raw":raw_doc, "surprises":[], "surprise": float("-inf")})
		count+=1
	logger.info("  ** Evaluation complete.")
	return dataset_surps

def document_surprise(surps, percentile=95):
	if len(surps):
		return np.percentile([x[2] for x in surps], percentile) #note that percentile calculates the highest.
	return float("-inf")

def estimate_document_surprise_pairs(doc, model, cooccurrence, word_occurrence, dictionary, documents, use_sglove=False, top_n_per_doc = 0, ignore_order=True):
	est_cooc_mat = estimate_document_cooccurrence_matrix(doc, model, cooccurrence, use_sglove=use_sglove)
	surps = document_cooccurrence_to_surprise(doc, est_cooc_mat, word_occurrence, dictionary, len(documents), ignore_order=ignore_order)
	surps.sort(key = lambda x: x[2], reverse=True)
	if top_n_per_doc and len(surps) > top_n_per_doc:
		return surps[:top_n_per_doc]
	return surps

def word_pair_surprise(w1_w2_cooccurrence, w1_occurrence, w2_occurrence, n_docs, offset = 0.5):
	# Offset is Laplacian smoothing
	w1_w2_cooccurrence = min(min(w1_occurrence,w2_occurrence),max(0,w1_w2_cooccurrence)) #Capped due to estimates being off sometimes
	p_w1_given_w2 = (w1_w2_cooccurrence + offset) / (w2_occurrence + offset)
	p_w1 = (w1_occurrence + offset) / (n_docs + offset)
	return -np.log2(p_w1_given_w2 / p_w1)

def extract_document_cooccurrence_matrix(doc, coocurrence):
	cooc_mat = np.zeros([len(doc),len(doc)])
	for i1,i2 in itertools.combinations(range(len(doc)),2):
		d1 = doc[i1][0]
		d2 = doc[i2][0]
		cooc_mat[i1,i2] = coocurrence[d1][d2]
	return cooc_mat

def estimate_word_pair_cooccurrence(wk1, wk2, model, cooccurrence, use_sglove = False):
	# take dot product of vectors
	cooc = np.dot(model.W[wk1],model.ContextW[wk2]) + model.b[wk1] + model.ContextB[wk2]
	if use_sglove:
		# correct for the significantly-less-than-unconditional-likelihood scaling in s_glove
		cooc = cooc #Not currently scaling these in any way
	elif cooccurrence[wk1][wk2] < model.x_max:
		# correct for the rare feature scaling described in https://nlp.stanford.edu/pubs/glove.pdf
		cooc *= 1.0/pow(cooccurrence[wk1][wk2] / model.x_max,model.alpha)
	return cooc[0]

def estimate_document_cooccurrence_matrix(doc, model, cooccurrence, use_sglove = False):
	cooc_mat = np.zeros([len(doc),len(doc)])
	for i1,i2 in itertools.combinations(range(len(doc)),2):
		d1 = doc[i1][0]
		d2 = doc[i2][0]
		cooc_mat[i1,i2] = estimate_word_pair_cooccurrence(d1, d2, model, cooccurrence, use_sglove=use_sglove)
	return np.triu(np.exp(cooc_mat), k=1)

def top_n_surps_from_doc(doc, model, cooccurrence, word_occurrence, dictionary, n_docs, top_n = 10, use_sglove = False):
	if len(doc):
		est_cooc_mat = estimate_document_cooccurrence_matrix(doc,model,cooccurrence, use_sglove=use_sglove)
		surps = document_cooccurrence_to_surprise(doc, est_cooc_mat, word_occurrence, dictionary, n_docs)
		surps.sort(key = lambda x: x[2], reverse=False)
		return surps[:min(top_n,len(surps))]
	else:
		return []

#Returns an ordered list of most-to-least surprising word combinations as (w1,w2,surprise) tuples
def document_cooccurrence_to_surprise(doc, cooc_mat, word_occurrence, dictionary, n_docs, ignore_order=True):
	surp_list = []
	for i1,i2 in zip(*np.triu_indices(cooc_mat.shape[0],k=1,m=cooc_mat.shape[1])):
		w1 = doc[i1][0]
		w2 = doc[i2][0]
		if not w1 == w2:
			s = word_pair_surprise(cooc_mat[i1,i2], word_occurrence[dictionary[w1]], word_occurrence[dictionary[w2]], n_docs)
			surp_list.append((dictionary[w1], dictionary[w2],s))
	surp_list.sort(key = lambda x: x[2], reverse=True)
	return surp_list

#Return the surprises from the given list that have the most similar feature word (i.e. w1) to the one in the given surp.
def most_similar_features(surp, surp_list, model, dictionary, n = 10):
	results = []
	for surp2 in surp_list:
		if not surp[:2] == surp2[:2]:
			surp_feat = model.W[dictionary.token2id[surp[0]]]
			surp2_feat = model.W[dictionary.token2id[surp2[0]]]
			results.append([surp,surp2,scipy.spatial.distance.euclidean(surp_feat, surp2_feat)])
	results.sort(key = lambda x: x[2])
	if n and n < len(results):
		return results[:n]
	return results

#Return the surprises from the given list that have the most similar context word(s) (i.e. w2) to the one in the given surp.
def most_similar_contexts(surp, surp_list, model, dictionary, n = 10):
	results = []
	for surp2 in surp_list:
		if not surp[:2] == surp2[:2]:
			surp_context = model.W[dictionary.token2id[surp[1]]]
			surp2_context = model.W[dictionary.token2id[surp2[1]]]
			results.append([surp,surp2,scipy.spatial.distance.euclidean(surp_context, surp2_context)])
	results.sort(key = lambda x: x[2])
	if n and n < len(results):
		return results[:n]
	return results

#Return the surprises from the given list that have the most similar vector difference to the one in the given surp.
def most_similar_differences(surp, surp_list, model, dictionary, n = 10):
	results = []
	for surp2 in surp_list:
		if not surp[:2] == surp2[:2]:
			surp_diff = model.W[dictionary.token2id[surp[0]]] -  model.W[dictionary.token2id[surp[1]]]
			surp2_diff = model.W[dictionary.token2id[surp2[0]]] -  model.W[dictionary.token2id[surp2[1]]]
			results.append([surp,surp2,surp_diff,surp2_diff,scipy.spatial.distance.euclidean(surp_diff, surp2_diff)])
	results.sort(key = lambda x: x[4])
	if n and n < len(results):
		return results[:n]
	return results

if __name__ == "__main__":
	# Parse arguments from the command
	parser = argparse.ArgumentParser(description="Evaluate a dataset using a trained GloVex model.")
	parser.add_argument("inputfile", help='The input file path to work with (omit the args and suffix)')
	parser.add_argument("--dataset", default="acm",type=str, help="Which dataset to assume.  Currently 'acm' or 'plots'")
	parser.add_argument("--name", default=None, type=str, help= "Name of this run (used when saving files.)")
	parser.add_argument("--dims", default = 100, type=int, help="The number of dimensions in the GloVe vectors.")
	parser.add_argument("--epochs", default = 25, type=int, help="The number of epochs to train GloVe for.")
	parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate for SGD.")
	parser.add_argument("--learning_rate_decay", default=25.0, type=float, help="LR is halved after this many epochs, divided by three after twice this, by four after three times this, etc.")
	parser.add_argument("--print_surprise_every", default=25, type=int, help="Evaluate the whole dataset and print the most surprising every this number of epochs (time consuming).")
	parser.add_argument("--glove_x_max", default = 100.0, type=float, help="x_max parameter in GloVe.")
	parser.add_argument("--glove_alpha", default = 0.75, type=float, help="alpha parameter in GloVe.")
	parser.add_argument("--no_below", default = 0.001, type=float,
						help="Min fraction of documents a word must appear in to be included.")
	parser.add_argument("--no_above", default = 0.5, type=float,
						help="Max fraction of documents a word can appear in to be included.")
	parser.add_argument("--overwrite_model", action="store_true",
						help="Ignore (and overwrite) existing .glovex file.")
	parser.add_argument("--overwrite_preprocessing", action="store_true",
						help="Ignore (and overwrite) existing .preprocessed file.")
	parser.add_argument("--use_sglove", action="store_true",
						help="Use the modified version of the GloVe algorithm that favours surprise rather than co-occurrence.")
	args = parser.parse_args()

	# Read the documents according to its type
	if args.dataset == "acm":
		reader = preprocessor.ACMDL_DocReader(args.inputfile, "title", "abstract", "ID",None, run_name=args.name, use_sglove=args.use_sglove)
	elif args.dataset == "plots":
		reader = preprocessor.WikiPlot_DocReader(args.inputfile)
	elif args.dataset == "recipes":
		reader = preprocessor.Recipe_Reader(args.inputfile, "Title and Ingredients", "ID",None)
	else:
		logger.info("You've tried to load a dataset we don't know about.  Sorry.")
		sys.exit()

	# Preprocess the data
	reader.preprocess(no_below=args.no_below, no_above=args.no_above, force_overwrite=args.overwrite_preprocessing)

	# Construct the glovex_model
	model = preprocessor.glovex_model(args.inputfile, reader.argstring, reader.cooccurrence, args.dims, args.glove_alpha,
						 args.glove_x_max, args.overwrite_model, use_sglove=args.use_sglove,
						 p_values=reader.cooccurrence_p_values if args.use_sglove else None)

	if np.sum(model.gradsqb)/len(model.gradsqb) == 1: # Nothing in the model object says how many epochs it's been trained for.  However, if the bias nodes are all 1, it's untrained
		logger.info(" ** Training created model.")
		preprocessor.train_glovex(model, reader, args)
	else:
		logger.info(" ** Loaded GloVe model.")

	# Evaluate it
	dataset_surps = eval_dataset_surprise(model, reader, top_n_per_doc=25)
	dataset_surps.sort(key = lambda x: x["surprise"], reverse=True)
	unique_surps = set((p for s in dataset_surps for p in s["surprises"]))
	print 'dataset_surps', len(dataset_surps)
	for doc in dataset_surps[:10]:
		# print doc["id"]+":", doc["title"]
		print doc["id"]+":"
		print "  ** 95th percentile surprise:",doc["surprise"]
		print "  ** Abstract:",doc["raw"]
		print "  ** Surprising pairs:",doc["surprises"]
		most_similar = most_similar_differences(doc["surprises"][0],unique_surps,model, reader.dictionary)
		print "  ** Most similar to top surprise:("+str(doc["surprises"][0])+")"
		for pair in most_similar:
			print "    ** ",pair[4],":",pair[1]
		print
