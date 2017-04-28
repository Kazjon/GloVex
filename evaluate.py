import argparse, logging, scipy, itertools

import preprocessor

import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("glovex")

def eval_dataset_surprise(model, acm, top_n_per_doc = 0, log_every=1000, ignore_order=True):
	logger.info("  ** Evaluating dataset.")
	dataset_surps = []
	count = 0
	for id,title,doc in zip(acm.doc_ids, acm.doc_titles, acm.documents):
		if count and count % log_every == 0:
			logger.info("    **** Evaluated "+str(count)+" documents.")
		if len(doc):
			surps = estimate_document_surprise_pairs(doc, model, acm, ignore_order=ignore_order)
			if top_n_per_doc and len(surps) > top_n_per_doc:
				surps = surps[:top_n_per_doc]
			dataset_surps.append((id,title,surps,document_surprise(surps)))
		else:
			dataset_surps.append((id,title,[],3))
		count+=1
	logger.info("  ** Evaluation complete.")
	return dataset_surps

def document_surprise(surps, percentile=95):
	if len(surps):
		return np.percentile([x[2] for x in surps], percentile)
	return float("inf")

def estimate_document_surprise_pairs(doc, model, acm, top_n_per_doc = 0, ignore_order=True):
	est_cooc_mat = estimate_document_cooccurrence_matrix(doc,model,acm.cooccurrence)
	surps = document_cooccurrence_to_surprise(doc, est_cooc_mat, acm.word_occurrence, acm.dictionary, len(acm.documents), ignore_order=ignore_order)
	surps.sort(key = lambda x: x[2])
	if top_n_per_doc and len(surps) > top_n_per_doc:
		return surps[:top_n_per_doc]
	return surps

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
def document_cooccurrence_to_surprise(doc, cooc_mat, word_occurrence, dictionary, n_docs, ignore_order=True):
	surp_list = []
	for i1,i2 in zip(*np.triu_indices(cooc_mat.shape[0],k=1,m=cooc_mat.shape[1])):
		w1 = doc[i1][0]
		w2 = doc[i2][0]
		if not w1 == w2:
			s = word_pair_surprise(cooc_mat[i1,i2], word_occurrence[dictionary[w1]], word_occurrence[dictionary[w2]], n_docs)
			surp_list.append((dictionary[w1], dictionary[w2],s))
	surp_list.sort(key = lambda x: x[2], reverse=False)
	return surp_list

#Return the surprises from the given list that have the most similar feature word (i.e. w1) to the one in the given surp.
def most_similar_features(surp, surp_list, model, dictionary, n = 10):
	results = []
	for surp2 in surp_list:
		if surp[:2] is not surp2[:2]:
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
		if surp[:2] is not surp2[:2]:
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
		if surp[:2] is not surp2[:2]:
			surp_diff = model.W[dictionary.token2id[surp[0]]] -  model.W[dictionary.token2id[surp[1]]]
			surp2_diff = model.W[dictionary.token2id[surp2[0]]] -  model.W[dictionary.token2id[surp2[1]]]
			results.append([surp,surp2,surp_diff,surp2_diff,scipy.spatial.distance.euclidean(surp_diff, surp2_diff)])
	results.sort(key = lambda x: x[4])
	if n and n < len(results):
		return results[:n]
	return results

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Evaluate a dataset using a trained GloVex model.")
	parser.add_argument("inputfile", help='The input file path to work with (omit the args and suffix)')
	parser.add_argument("--no_below", default = 0.001, type=float,
						help="Min fraction of documents a word must appear in to be included.")
	parser.add_argument("--no_above", default = 0.75, type=float,
						help="Max fraction of documents a word can appear in to be included.")
	args = parser.parse_args()
	acm = preprocessor.ACMDL_DocReader(args.inputfile)
	acm.preprocess(no_below=args.no_below, no_above=args.no_above)
	model = preprocessor.glovex_model(args.inputfile, acm.argstring, acm.cooccurrence)
	logger.info(" ** Loaded GloVe")
	dataset_surps = eval_dataset_surprise(model, acm, top_n_per_doc=25)
	dataset_surps.sort(key = lambda x: x[1])
	unique_surps = set((p for s in dataset_surps for p in s[2]))
	for doc in dataset_surps[:10]:
		print doc[0]+":", doc[1]
		print doc[2]
		most_similar = most_similar_differences(doc[2][0],unique_surps,model, acm.dictionary)
		print "  ** Most similar to top surprise:("+str(doc[2][0])+")"
		for pair in most_similar:
			print "    * ",pair[4],":",pair[1]
		print


