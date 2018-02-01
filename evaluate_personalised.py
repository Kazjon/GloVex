import argparse, logging, scipy, itertools, random, sys

import preprocessor, evaluate

import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("glovex")

def eval_personalised_dataset_surprise(models, reader, user, log_every=1000, ignore_order=True):
# def eval_personalised_dataset_surprise(models, acm, user, top_n_per_doc=25, log_every=1000, ignore_order=True):
	logger.info("  ** Evaluating dataset..")
	dataset_surps = []
	count = 0
	# for id,title,doc,raw_doc in zip(acm.doc_ids, acm.doc_titles, acm.documents, acm.doc_raws):
	for id, doc, raw_doc in zip(reader.doc_ids, reader.documents, reader.doc_raws):
		if count and count % log_every == 0:
			logger.info("    **** Evaluated "+str(count)+" documents.")
		if len(doc):
			surps = estimate_personalised_document_surprise_pairs(doc, models, reader, user, ignore_order=ignore_order)
			# dataset_surps.append({"id": id,"title":title,"raw":raw_doc, "surprises":surps, "surprise": document_surprise(surps)})
			dataset_surps.append({"id": id, "raw":raw_doc, "surprises":convert_personalised_to_combined_surprise(surps, reader.dictionary), "surprise": document_surprise(surps)})
		else:
			# dataset_surps.append({"id": id,"title":title,"raw":raw_doc, "surprises":[], "surprise": float("inf")})
			dataset_surps.append({"id": id, "raw":raw_doc, "surprises":[], "surprise": float("-inf")})
		count+=1
	logger.info("  ** Evaluation complete.")
	return dataset_surps

def convert_personalised_to_combined_surprise(surps, dictionary):
	surplist = []
	for k1,v1 in surps.iteritems():
		for k2,v2 in v1.iteritems():
			surplist.append((dictionary.id2token[k1], dictionary.id2token[k2], dict(v2)["combined"]))
	return surplist

def document_surprise(surps, percentile=95):
	if len(surps):
		surp_ratings = [w1_w2 for w1 in surps.values() for w1_w2 in w1.values()]
		return np.percentile([r[1] for l in surp_ratings for r in l if r[0] == "combined"], percentile) #note that percentile calculates the highest.
	return float("-inf")

def estimate_personalised_document_surprise_pairs(doc, models, reader, user, top_n_per_doc = 0, ignore_order=True):
	surps = {}
	for model,fc in zip(models,reader.famcats):
		#rekeyed_cooccurrence = {acm.per_fc_keys_to_all_keys[fc][k]:{acm.per_fc_keys_to_all_keys[fc][k2]:v2 for k2,v2 in v.iteritems()} for k,v in acm.cooccurrence[fc].iteritems()}
		rekeyed_doc = [(reader.all_keys_to_per_fc_keys[fc][k],v) for k,v in doc if k in reader.all_keys_to_per_fc_keys[fc].keys()]
		est_cooc_mat = estimate_document_cooccurrence_matrix(rekeyed_doc,model,reader.cooccurrence[fc])
		document_cooccurrence_to_surprise(surps, fc, rekeyed_doc, est_cooc_mat, reader.word_occurrence[fc], reader.dictionary, reader.per_fc_keys_to_all_keys[fc], reader.docs_per_fc[fc], ignore_order=ignore_order)
	combine_surprise(surps, {fc:f for fc,f in zip(reader.famcats,user)})
	return surps

def estimate_personalised_document_surprise_pairs_one_fc(doc, model, fc, reader, top_n_per_doc = 0, ignore_order=True):
	surps = {}
	rekeyed_doc = [(reader.all_keys_to_per_fc_keys[fc][k],v) for k,v in doc if k in reader.all_keys_to_per_fc_keys[fc].keys()]
	est_cooc_mat = estimate_document_cooccurrence_matrix(rekeyed_doc,model,reader.cooccurrence[fc])
	surps_list = document_cooccurrence_to_surprise(surps, fc, rekeyed_doc, est_cooc_mat, reader.word_occurrence[fc], reader.dictionary, reader.per_fc_keys_to_all_keys[fc], reader.docs_per_fc[fc], ignore_order=ignore_order)

	#May also need to un-re-key on the way out
	# This is what's in the non-pers version, in which the above call seems to return a list:
	surps_list.sort(key = lambda x: x[2], reverse=True)
	if top_n_per_doc and len(surps_list) > top_n_per_doc:
		return surps_list[:top_n_per_doc]
	return surps_list

def combine_surprise(surps, user):
	for w1,w1_pairs in surps.iteritems():
		for w2 in w1_pairs:
			surps[w1][w2].append(("combined",combine_surprise_across_famcats_for_user(surps[w1][w2],user, method="weighted_sum")))

#Different methods for combining the surprise predictions from each famcat. Current method is a placeholder only.
#Note: Need to re-investigate exactly what the reported surprise values are before this can be used properly.
def combine_surprise_across_famcats_for_user(surprises,user, method="weighted_sum"):
	if method == "weighted_sum":
			return sum(s[1]*user[s[0]] for s in surprises)
	else:
		raise NotImplementedError


def extract_document_cooccurrence_matrix(doc, coocurrence):
	cooc_mat = np.zeros([len(doc),len(doc)])
	for i1,i2 in itertools.combinations(range(len(doc)),2):
		d1 = doc[i1][0]
		d2 = doc[i2][0]
		cooc_mat[i1,i2] = coocurrence[d1][d2]
	return cooc_mat

def estimate_word_pair_cooccurrence(wk1, wk2, model, cooccurrence, offset = 0.5):
	# take dot product of vectors
	cooc = np.dot(model.W[wk1],model.ContextW[wk2]) + model.b[wk1] + model.ContextB[wk2]
	try:
		actual_cooc = cooccurrence[wk1][wk2]
	except KeyError:
		actual_cooc = offset

	# correct for the rare feature scaling described in https://nlp.stanford.edu/pubs/glove.pdf
	if actual_cooc < model.x_max:
		cooc *= 1.0/pow(actual_cooc / model.x_max,model.alpha)
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
		surps.sort(key = lambda x: x[2], reverse=True)
		return surps[:min(top_n,len(surps))]
	else:
		return []

#Returns an ordered list of most-to-least surprising word combinations as (w1,w2,surprise) tuples, also updates surp dictionary it's provided
def document_cooccurrence_to_surprise(surps, fc, doc, cooc_mat, word_occurrence, dictionary, key_map, n_docs, ignore_order=True):
	surp_list = []
	for i1,i2 in zip(*np.triu_indices(cooc_mat.shape[0],k=1,m=cooc_mat.shape[1])):
		w1 = doc[i1][0]
		w2 = doc[i2][0]
		if not w1 == w2 and w1 in key_map.keys() and w2 in key_map.keys(): #if the words aren't in this famcat's model, then don't make any predictions on them.
			s = evaluate.word_pair_surprise(cooc_mat[i1,i2], word_occurrence[dictionary[key_map[w1]]], word_occurrence[dictionary[key_map[w2]]], n_docs)
			if key_map[w1] not in surps.keys():
				surps[key_map[w1]] = {key_map[w2]:[(fc,s)]}
			elif key_map[w2] not in surps[key_map[w1]].keys():
				surps[key_map[w1]][key_map[w2]] = [(fc,s)]
			else:
				surps[key_map[w1]][key_map[w2]].append((fc,s))
			surp_list.append((dictionary[key_map[w1]], dictionary[key_map[w2]], s))
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
def most_similar_differences_personal(surp, surp_list, models, dictionary, n = 10):
	results = []
	for surp2 in surp_list:
		if not surp[:2] == surp2[:2]:
			surp_diffs = []
			surp2_diffs = []
			dists = []
			for model in models:
				surp_diffs.append(model.W[dictionary.token2id[surp[0]]] -  model.W[dictionary.token2id[surp[1]]])
				surp2_diffs.append(model.W[dictionary.token2id[surp2[0]]] -  model.W[dictionary.token2id[surp2[1]]])
				dists.append(scipy.spatial.distance.euclidean(surp_diffs[-1], surp2_diffs[-1]))
			results.append([surp,surp2,np.average(surp_diffs),np.average(surp2_diffs),np.average(dists)])
	results.sort(key = lambda x: x[4])
	if n and n < len(results):
		return results[:n]
	return results

if __name__ == "__main__":
	# Parse arguments from the command
	parser = argparse.ArgumentParser(description="Evaluate a dataset using a trained GloVex model.")
	parser.add_argument("inputfile", help='The input file path to work with (omit the args and suffix)')
	parser.add_argument("--dataset", default="acm", type=str, help="Which dataset to assume.  Currently 'acm' or 'plots'")
	parser.add_argument("--name", default=None, type=str, help="Name of this run (used when saving files.)")
	parser.add_argument("--dims", default = 100, type=int, help="The number of dimensions in the GloVe vectors.")
	parser.add_argument("--epochs", default = 25, type=int, help="The number of epochs to train GloVe for.")
	parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate for SGD.")
	parser.add_argument("--learning_rate_decay", default=25.0, type=float, help="LR is halved after this many epochs, divided by three after twice this, by four after three times this, etc.")
	parser.add_argument("--print_surprise_every", default=25, type=int, help="Evaluate the whole dataset and print the most surprising every this number of epochs (time consuming).")
	parser.add_argument("--glove_x_max", default = 100.0, type=float, help="x_max parameter in GloVe.")
	parser.add_argument("--glove_alpha", default = 0.5, type=float, help="alpha parameter in GloVe.")
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
	args = parser.parse_args()

	# Read the documents according to its type
	if args.dataset == "acm":
		reader = preprocessor.ACMDL_DocReader(args.inputfile, "title", "abstract", "ID", famcat_column="category", run_name=args.name, use_sglove=args.use_sglove)
	elif args.dataset == "plots":
		reader = preprocessor.WikiPlot_DocReader(args.inputfile)
	elif args.dataset == "recipes":
		reader = preprocessor.Recipe_Reader(args.inputfile, "Title and Ingredients", "ID", famcat_column="cuisine")
	else:
		logger.info("You've tried to load a dataset we don't know about.  Sorry.")
		sys.exit()

	# Preprocess the data
	reader.preprocess(no_below=args.no_below, no_above=args.no_above, force_overwrite=args.overwrite_preprocessing)

	# print reader.famcats
	# print 'doc_famcats', reader.doc_famcats
	# print 'doc_raws', reader.doc_raws
	# print 'cooccurrence', reader.cooccurrence.iteritems()

	# Load personalized models
	models = preprocessor.load_personalised_models(args.inputfile, reader)

	for model,fc in zip(models,reader.famcats):
		if np.sum(model.gradsqb)/len(model.gradsqb) == 1: # Nothing in the model object says how many epochs it's been trained for.  However, if the bias nodes are all 1, it's untrained
			logger.info(" ** Training created "+fc+" model.")
			preprocessor.train_glovex(model, reader, args, famcat=fc)
		else:
			logger.info(" ** Loaded GloVe model for "+fc+".")

	logger.info(" ** Loaded GloVe")

	# Get familiarity category of the data
	# The user's familiarity scores for each cuisine (score: between 0-1) [[0.3, 0.3, 0.5, ], [], [] ... ]
	# Try with one user first
	# [mexican, chinese, greek, indian, thai, italian]
	# user = [random.random() for fc in reader.famcats]
	user = [0.5, 0.6, 0.8, 0.6, 0.3, 0.7]
	print 'user', user
	logger.info(" ** Generated fake user familiarity profile: " + ", ".join([str(fc)+": "+str(f) for f,fc in zip(user, reader.famcats)]))

# # 	Repeat for each user
# 	for each_user in users:
	# Evaluate personalized surprise model
	# dataset_surps = eval_personalised_dataset_surprise(models, acm, user, top_n_per_doc=25)
	# dataset_surps = eval_personalised_dataset_surprise(models, reader, user, top_n_per_doc=25)
	dataset_surps = eval_personalised_dataset_surprise(models, reader, user)
	dataset_surps.sort(key = lambda x: x["surprise"], reverse=True)
	unique_surps = set((p for s in dataset_surps for p in s["surprises"]))
	for doc in dataset_surps[:10]:
		print doc["id"]+": (no titles currently)"#, doc["title"]
		print "  ** 95th percentile surprise:",doc["surprise"]
		print "  ** Document text:",doc["raw"]
		print "  ** Surprising pairs:",doc["surprises"]

		#if len(doc["surprises"]):
		#	# most_similar = most_similar_differences(doc["surprises"][0],unique_surps, model, reader.dictionary)
		#	most_similar = most_similar_differences_personal(doc["surprises"][0],unique_surps, models, reader.dictionary)
		#	print "  ** Most similar to top surprise:("+str(doc["surprises"][0])+")"
		#	for pair in most_similar:
		#		print "    ** ",pair[4],":",pair[1]
		print
