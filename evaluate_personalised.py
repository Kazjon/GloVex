import os, re, argparse, logging, scipy.spatial, itertools, sys, numpy as np, pandas as pd, pprint, pickle, json, collections
pp = pprint.PrettyPrinter(indent=4)

import preprocessor, evaluate

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("glovex")

def eval_personalised_dataset_surprise(models, reader, user, log_every=1000, ignore_order=True, offset=0):
	logger.info("  ** Evaluating dataset..")
	dataset_surps = []
	count = 0
	for id, doc, raw_doc in zip(reader.doc_ids[offset:], reader.documents[offset:], reader.doc_raws[offset:]):
		if count and count % log_every == 0:
			logger.info("    **** Evaluated "+str(count)+" documents.")
		if len(doc):
			surps = estimate_personalised_document_surprise_pairs(doc, models, reader, user, ignore_order=ignore_order)
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
			surps_without_combined = tuple([s for s in v2 if s[0] is not "combined"])
			surplist.append((dictionary.id2token[k1], dictionary.id2token[k2], dict(v2)["combined"], surps_without_combined))
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
	#combine_surprise_weighted_sum(surps, {fc: f for fc, f in zip(reader.famcats, user)})
	combine_surprise_normalised(surps, reader, {fc:f for fc,f in zip(reader.famcats,user)})
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

def combine_surprise_weighted_sum(surps, user):
	for w1,w1_pairs in surps.iteritems():
		for w2 in w1_pairs:
			# Multiply the surprise for each category by the user's familiarity with that category, sum up and divide by the number of categories.
			surps[w1][w2].append(("combined",sum(s[1] * user[s[0]] for s in surps) / len(surps)))

def combine_surprise_normalised(surps, reader, user):
	for w1, w1_pairs in surps.iteritems():
		for w2 in w1_pairs:
			w1_str = reader.dictionary[w1]
			w2_str = reader.dictionary[w2]
			w1_occs_by_famcat = [reader.word_occurrence[fc][w1_str] if w1_str in reader.word_occurrence[fc] else 0 for fc in reader.famcats]
			w2_occs_by_famcat = [reader.word_occurrence[fc][w2_str] if w2_str in reader.word_occurrence[fc] else 0 for fc in reader.famcats]
			coocs_by_famcat = []
			for fc in reader.famcats:
				try:
					cooc = reader.cooccurrence[fc][reader.all_keys_to_per_fc_keys[fc][w1]][reader.all_keys_to_per_fc_keys[fc][w2]]
				except KeyError:
					cooc = 0
				coocs_by_famcat.append(cooc)
			# Multiply word occurrences in each category by the user's familiarity with that category, sum them all up and calculate surprise
			cooc = sum([c * user[fc] for c, fc in zip(coocs_by_famcat, reader.famcats)])
			w1_occs = sum([c * user[fc] for c, fc in zip(w1_occs_by_famcat, reader.famcats)])
			w2_occs = sum([c * user[fc] for c, fc in zip(w2_occs_by_famcat, reader.famcats)])

			surps[w1][w2].append(("combined", evaluate.word_pair_surprise(cooc, w1_occs, w2_occs, reader.total_docs)))


def combine_surprise_across_famcats_for_user_normalised(w1_occs_by_famcat, w2_occs_by_famcat,coocs_by_famcat, n_docs, user):
	# Multiply word occurrences in each category by the user's familiarity with that category, sum them all up and calculate surprise
	cooc = sum([c*fc for c,fc in zip(coocs_by_famcat,user)])
	w1_occs = sum([c*fc for c,fc in zip(w1_occs_by_famcat,user)])
	w2_occs = sum([c*fc for c,fc in zip(w2_occs_by_famcat,user)])
	return evaluate.word_pair_surprise(cooc,w1_occs,w2_occs,n_docs)

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

def survey_reader(qchef_surveydata_fn):
	# Add .csv to the file path
	qchef_surveydata_fn = qchef_surveydata_fn + '.csv'
	# Read the CSV file
	qchef_surveydata_df = pd.read_csv(qchef_surveydata_fn)
	# Lower case the comments_1 column
	qchef_surveydata_df['comments_1'] = qchef_surveydata_df['comments_1'].str.lower()
	# Select the first timers dem_1 == 1
	first_timer_df = qchef_surveydata_df[qchef_surveydata_df['dem_1'] == 1]
	# Set up the condition for filtering out the second timers: if the keywords in the comments contains 'kaz' or 'test'
	condition1 = first_timer_df['comments_1'].str.contains('test') == False
	condition2 = first_timer_df['comments_1'].str.contains('test').isnull()
	condition3 = first_timer_df['comments_1'].str.contains('kaz') == False
	condition4 = first_timer_df['comments_1'].str.contains('kaz').isnull()
	# Filter out the users who mentioned in their comments that it's the second time
	first_timer_df = first_timer_df[(condition1 | condition2) & (condition3 | condition4)]
	# print first_timer_df['comments_1'].unique()
	# How Much Do You Know About These Foods?
	# hab_11_01: American, hab_11_02: Chinese, hab_11_03: Mexican, hab_11_04: Italian, hab_11_05: Japanese, hab_11_06: Greek, hab_11_07: French, hab_11_08: Thai, hab_11_09: Spanish, hab_11_10: Indian, hab_11_11: Mediterranean
	fam_cols = ['hab_11_01', 'hab_11_02', 'hab_11_03', 'hab_11_04', 'hab_11_05', 'hab_11_06', 'hab_11_07', 'hab_11_08',
				'hab_11_09', 'hab_11_10', 'hab_11_11']
	# To-do: Modern is either American or the average of all of the cuisines
	# Filter in the user familiarity columns only
	user_fam_df = first_timer_df[fam_cols]
	# Reorder cuisines as following to match the reader.famcat: mexican, chinese, greek, indian, thai, italian
	cuisine_cols = ['hab_11_03', 'hab_11_02', 'hab_11_01', 'hab_11_06', 'hab_11_10', 'hab_11_08', 'hab_11_04']
	user_fam_df = user_fam_df[cuisine_cols]
	# Get list of lists of users' familiarity rarings
	user_fam_score_arr = user_fam_df.values.tolist()
	# Scale the ratings to range between 0-1
	user_fam_scaled_arr = np.array(user_fam_score_arr) / 5.0
	# Get the surprise scores
	# Get columns that ask about surprise
	surp_cols = []
	for each_col in first_timer_df.columns:
		if 'surp_' in each_col and int(re.search(r'\d+', each_col).group()) % 2 != 0:
			surp_cols.append(each_col)
	# Get the list of lists
	users_surp_ratings_arr = first_timer_df[surp_cols].values.tolist()
	# Return as a list instead of a numpy array
	return user_fam_scaled_arr.tolist(), users_surp_ratings_arr

if __name__ == "__main__":
	# Get the current dir's path
	cwd = os.getcwd()
	# Parse arguments from the command
	parser = argparse.ArgumentParser(description="Evaluate a dataset using a trained GloVex model.")
	parser.add_argument("inputfile", help='The input file path to work with (omit the args and suffix)')
	parser.add_argument("--user_survey", default=None, type=str, help='The input file path to the user survey')
	parser.add_argument("--surprise_recipes", default=None, type=str, help='The input file path to the surprise recipes')
	# Argument to chekc the function used for modern
	parser.add_argument("--dataset", default="acm", type=str, help="Which dataset to assume.  Currently 'acm' or 'plots'")
	parser.add_argument("--name", default=None, type=str, help="Name of this run (used when saving files.)")
	parser.add_argument("--dims", default = 100, type=int, help="The number of dimensions in the GloVe vectors.")
	parser.add_argument("--epochs", default = 25, type=int, help="The number of epochs to train GloVe for.")
	parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate for SGD.")
	parser.add_argument("--learning_rate_decay", default=25.0, type=float, help="LR is halved after this many epochs, divided by three after twice this, by four after three times this, etc.")
	parser.add_argument("--print_surprise_every", default=100, type=int, help="Evaluate the whole dataset and print the most surprising every this number of epochs (time consuming).")
	parser.add_argument("--glove_x_max", default = 100.0, type=float, help="x_max parameter in GloVe.")
	parser.add_argument("--glove_alpha", default = 0.5, type=float, help="alpha parameter in GloVe.")
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

	# Load/train personalised models
	models = []
	model_famcat = dict()
	for fc in reader.famcats:
		# Change glovex_model to return whether it was loaded or made
		model = preprocessor.glovex_model(args.inputfile, reader.argstring+"_fc"+str(fc), reader.cooccurrence[fc],
										  args.dims, args.glove_alpha, args.glove_x_max,args.overwrite_model,
										  use_sglove=args.use_sglove, p_values=reader.cooccurrence_p_values if args.use_sglove else None)
		if np.sum(model.gradsqb)/len(model.gradsqb) == 1:
			# Nothing in the model object says how many epochs it's been trained for. However, if the bias nodes are all 1, it's untrained
			logger.info(" ** Training created "+fc+" model.")
			preprocessor.train_glovex(model, reader, args, famcat=fc)
		else:
			logger.info(" ** Loaded GloVe model for "+fc+".")
		# Store the mode in an array
		models.append(model)
		# Store the model in a dict
		model_famcat[fc] = model

	# Get all of the surprises for all of the combinations
	top_surps = []
	all_comb_surps_per_cuisine = collections.defaultdict(dict)
	for famcat in model_famcat:
		print('Store all surprise combinations for', famcat)
		all_comb_suprs = preprocessor.save_surps(model_famcat[famcat], reader, famcat=famcat)
		# pp.pprint(all_comb_suprs)
		for each_comb in all_comb_suprs:
			# print(str(each_comb))
			all_comb_surps_per_cuisine[frozenset(each_comb)][famcat] = all_comb_suprs[each_comb]
		# print(all_comb_surps_per_cuisine)
	print('Number of combinations in all_comb_surps_per_cuisine:', len(all_comb_surps_per_cuisine))
	# print(all_comb_surps_per_cuisine)
	# # Store all of the combination surprises in a JSON format
	# all_comb_suprs_fn = cwd + '/GloVex/results/new/all_comb_surps_per_cuisine.json'
	# json.dump(all_comb_surps_per_cuisine, open(all_comb_suprs_fn, 'w'), indent=4)

	# Get familiarity category of the data from the user survey data
	# The user's familiarity scores for each cuisine (score: between 0-1) [[0.3, 0.3, 0.5, ], [], [] ... ]
	# Order of cuisines[mexican, chinese, modern, greek, indian, thai, italian]
	if args.user_survey != None:
		users_fam, users_surp_ratings_arr = survey_reader(args.user_survey)
	else:
		print 'There is no user data, an assumed user will be modeled instead'
		users_fam = [
			[0.5, 0.6, 0.8, 0.6, 0.3, 0.7, 0.5],
			[0.4, 0.3, 0.2, 0.8, 0.6, 0.7, 0.2]
		]
		print 'users_fam', users_fam
		logger.info(" ** Generated fake user familiarity profile: " + ", ".join(
			[str(fc) + ": " + str(f) for f, fc in zip(users_fam[0], reader.famcats)]))

	# Evaluate personalized surprise model
	print 'Number of users:', len(users_fam)
	# Initialize the user surprise estimates
	user_suprise_estimates = {}
	all_comb_surps_per_user = collections.defaultdict(dict)
	print('Store all_comb_surps_per_user:')
	# Repeat for each user
	for user_idx, user_fam_cat in enumerate(users_fam):
		print "User's familiarity", user_idx, user_fam_cat
		# Store the fam cat into the dict
		user_suprise_estimates['user_' + str(user_idx)] = {'user_fam_cat': user_fam_cat, 'recipes_surp': []}
		# Store the surprises; there are two ways to evaluate the surprise recipes
		# dataset_surps = eval_personalised_dataset_surprise(models, surprise_recipe_reader, user_fam_cat)
		dataset_surps = eval_personalised_dataset_surprise(models, reader, user_fam_cat, offset=73106)
		dataset_surps.sort(key = lambda x: x["surprise"], reverse=True)
		# Get the unique set of surprises for this user
		unique_surps = set((p for s in dataset_surps for p in s["surprises"]))
		# print(unique_surps)
		# Iterate over the unique surps to store in all_comb_surps_per_user
		for each_comb in unique_surps:
			# print(str((each_comb[0], each_comb[1])))
			# Store the combined the surprise scores of all cuisines (double keys: primary: ingredient combinations, secondary: user ID)
			all_comb_surps_per_user[frozenset((each_comb[0], each_comb[1]))]['user_' + str(user_idx)] = each_comb[2]
		# print(all_comb_surps_per_user)
		# For the surprising receipes, store their recipe ID, surprise, raw doc and surprise cuisine scores
		recipe_surp_dict = {}
		# for doc in dataset_surps[:10]:
		for doc in dataset_surps:
			# pp.pprint(doc)
			recipe_surp_dict['recipe_id'] = doc['id']
			recipe_surp_dict['95th_percentile'] = doc['surprise']
			recipe_surp_dict['ingredients'] = doc['raw']
			recipe_surp_dict['surprise_cuisine'] = []
			for surprise_combination in doc['surprises']:
				recipe_surp_dict['surprise_cuisine'].append(surprise_combination)
			# Append the recipe_surp_dict to the user_suprise_estimates
			user_suprise_estimates['user_' + str(user_idx)]['recipes_surp'].append(recipe_surp_dict)
			# Renew/empty the dict for next iteration
			recipe_surp_dict = {}
	print('Number of combinations in all_comb_surps_per_user:',len(all_comb_surps_per_user))
	# print(all_comb_surps_per_user)
	# # Store the user_suprise_estimates in a JSON
	# all_comb_surps_per_user_fn = cwd + '/GloVex/results/new/all_comb_surps_per_user.json'
	# json.dump(all_comb_surps_per_user, open(all_comb_surps_per_user_fn, 'w'), indent=4)
	# Store the user_suprise_estimates in a pickle
	user_suprise_estimates_pickle_fn = cwd + '/GloVex/results/new/user_suprise_estimates.pickle'
	pickle.dump(user_suprise_estimates, open(user_suprise_estimates_pickle_fn, 'wb'))

	# Test if all combinations are the same in both dicts
	print('Number of combinations in all_comb_surps_per_cuisine:', len(all_comb_surps_per_cuisine))
	print('Number of combinations in all_comb_surps_per_user:', len(all_comb_surps_per_user))
	print('Are all keys in the two dicts equal?',
		set(all_comb_surps_per_cuisine.keys()) == set(all_comb_surps_per_user.keys()))
	combinations_instersection = \
		set(all_comb_surps_per_cuisine.keys()).intersection(set(all_comb_surps_per_user.keys()))
	print('Number of intersecting keys between all_comb_surps_per_cuisine and all_comb_surps_per_user',
		len(combinations_instersection))
	# Combine all_surp per cuisine and per user
	all_comb_surps_dict = collections.defaultdict(dict)
	# Iterate over the ingredient combinations
	# for each_comb in all_comb_surps_per_cuisine.keys():
	for each_comb in combinations_instersection:
		# print(each_comb)
		# Add the the per_cuisine data
		all_comb_surps_dict[str(each_comb)]['per_cuisine'] = all_comb_surps_per_cuisine[each_comb]
		all_comb_surps_dict[str(each_comb)]['per_user'] = all_comb_surps_per_user[each_comb]
		# # If there is combination exists in the all_comb_surps_per_user, add the the per_user data
		# if each_comb in all_comb_surps_per_user:
		# 	all_comb_surps_dict[str(each_comb)]['per_user'] = all_comb_surps_per_user[each_comb]
	# Store the all_comb_surps_dict in a JSON
	all_comb_surps_fn = cwd + '/GloVex/results/new/all_comb_surps.json'
	json.dump(all_comb_surps_dict, open(all_comb_surps_fn, 'w'), indent=4)

"""
('Number of combinations in all_comb_surps_per_cuisine:', 39821)
('Number of combinations in all_comb_surps_per_user:', 471)
('Are all keys in the two dicts equal?', False)
('Number of intersecting keys between all_comb_surps_per_cuisine and all_comb_surps_per_user', 438)

('Number of combinations in all_comb_surps_per_user:', 471)
('Number of combinations in all_comb_surps_per_cuisine:', 74881)
('Number of combinations in all_comb_surps_per_user:', 471)
('Are all keys in the two dicts equal?', False)
('Number of intersecting keys between all_comb_surps_per_cuisine and all_comb_surps_per_user', 471)
"""
