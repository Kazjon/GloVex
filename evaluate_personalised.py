# ToDo: Argument to check the function used for modern
import os, re, time, argparse, logging, scipy.spatial, itertools, sys, numpy as np, pandas as pd, pprint, pickle, json, collections
from collections import defaultdict
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
			dataset_surps.append({"id": id, "raw":raw_doc,
								  "surprises":convert_personalised_to_combined_surprise(surps, reader.dictionary),
								  "surprise_95": document_surprise(surps),
								  "surprise_90": document_surprise(surps, 90)})
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

########################################################################################################################
# Survey reader class, its variables and functions

# col_str2tuple: Function to convert knowledge answers to scores
def col_str2tuple(answer):
	return tuple(answer.replace(' ', '').split(','))
# end col_str2tuple

class survey_reader(object):
	def __init__(self):
		self.food_cuisine_survey_df = pd.DataFrame()
		# Create a map between the ratings and the scores for the familiarity and surprise
		# Map between direct familiarity of the user that they claim for each cuisine
		self.familiarity_direct_map = {
			'Not sure': -1,
			'Not at all': 1,
			'Somewhat': 2,
			'Moderately': 3,
			'Very much': 4,
			'Extremely': 5
		}
		# Map between the familiarity questions and their correct answers
		self.familiarity_knowledge_map_original_list = {
			'chinese': {
				'common': ['Bamboo shoots', 'Ginger', 'Water chestnuts'],
				'uncommon': ['Nutmeg', 'Sour cream']
			},
			'mexican': {
				'common': ['Tortillas', 'Green chilli peppers', 'Pinto beans'],
				'uncommon': ['Sesame oil', 'Strawberries']
			},
			'italian': {
				'common': ['Pasta', 'Prosciutto', 'Ricotta'],
				'uncommon': ['Turmeric', 'Soy sauce']
			},
			'greek': {
				'common': ['Lamb', 'Feta cheese', 'Phyllo pastry'],
				'uncommon': ['Ginger', 'Brown sugar']
			},
			'indian': {
				'common': ['Ghee', 'Turmeric', 'Cumin'],
				'uncommon': ['Honey', 'Thyme']
			},
			'thai': {
				'common': ['Fish sauce', 'Lemongrass', 'Coconut milk'],
				'uncommon': ['Cheese', 'Vanilla']
			}
		}
		# Remove spaces from all of the ingredients and use that dict
		self.familiarity_knowledge_map_list = defaultdict()
		for each_cuisine in self.familiarity_knowledge_map_original_list:
			self.familiarity_knowledge_map_list[each_cuisine] = {}
			for each_list in self.familiarity_knowledge_map_original_list[each_cuisine]:
				self.familiarity_knowledge_map_list[each_cuisine][each_list] = []
				for each_ingredient in self.familiarity_knowledge_map_original_list[each_cuisine][each_list]:
					self.familiarity_knowledge_map_list[each_cuisine][each_list].append(
						''.join(each_ingredient.split()).lower())
	# end of __init__

	# get_cuisine_fam_pref
	def get_cuisine_fam_pref(self, search_str, col_str, this_col):
		# Get familiarity column
		familiar_col = re.search(search_str + '(.*) cuisine', this_col)
		if not familiar_col is None:
			# Get cuisine name and its suffix
			cuisine_name = familiar_col.group(1).lower() + col_str
			# Append cuisine name and its suffix
			self.food_cuisine_survey_df[cuisine_name] = self.food_cuisine_survey_df[this_col].map(self.familiarity_direct_map)
	# end get_cuisine_fam_pref

	# get_surprise_fam_pref
	def get_surprise_fam_pref(self, search_str, col_str, this_col):
		surprise_col = re.search(search_str, this_col)
		if not surprise_col is None:
			# Get surprise recipe's number
			each_col_num = re.findall('\\b\\d+\\b', this_col)
			if len(each_col_num) == 0: surprise_recipe_num = 0
			else: surprise_recipe_num = each_col_num[0]
			# Name the surprise rating column
			surprise_rating_col = 'recipe' + str(surprise_recipe_num) + col_str
			# Convert the users' inputs to ints
			self.food_cuisine_survey_df[surprise_rating_col] = self.food_cuisine_survey_df[this_col].map(self.familiarity_direct_map)
	# end get_surprise_fam_pref

	# get_knowledge_fam
	def get_knowledge_fam(self, knowledge_str, this_col):
		knowledge_col = re.search(knowledge_str + '(.*) cuisine', this_col)
		if not knowledge_col is None:
			# Get cuisine name and its suffix
			cuisine_name = knowledge_col.group(1).lower()
			# Convert the column from a one string to a tuple
			self.food_cuisine_survey_df[this_col] = self.food_cuisine_survey_df[this_col].apply(col_str2tuple)
			# Iterate over the set of all possible values for each of the cuisine columns
			for each_ingredient in frozenset().union(*self.food_cuisine_survey_df[this_col]):
				# Make the column name as the cuisine name and the ingredient separated by an underscore
				col_name = cuisine_name + '_knowledge_ingredient_' + each_ingredient
				# Append this name into an array
				# Transform the names of the ingredients
				self.food_cuisine_survey_df[col_name] = self.food_cuisine_survey_df.apply(lambda _: int(each_ingredient in _[this_col]), axis=1)
				# Invert to the opposite for uncommon ingreidients of the cuisine to make it if unchosen then its corrent (1) and vise versa
				if each_ingredient.lower() in self.familiarity_knowledge_map_list[cuisine_name]['uncommon']:
					self.food_cuisine_survey_df[col_name] = (~self.food_cuisine_survey_df[col_name].astype(bool)).astype(int)
	# end get_knowledge_fam

	# create_users_input_dict: Get dictionary of users' responses separately
	def create_users_input_dict(self, required_cols, scale):
		# Put the surprise scores in a DF
		users_df = self.food_cuisine_survey_df[required_cols]
		# Get the surprise list and put in a column
		users_df['col_list'] = (np.array(users_df.values.tolist()) / scale).tolist()
		# Create a dict with the keys as the index and the values as the surprise ratings
		return pd.Series(users_df['col_list'].values, index=users_df.index).to_dict()
	# end of create_users_input_dict

	# read_survey
	def read_survey(self, food_cuisine_survey_fn, _fam_cats_sorted_):
		# Add .csv to the file path
		food_cuisine_survey_fn = food_cuisine_survey_fn + '.csv'
		# Read the CSV file
		self.food_cuisine_survey_df = pd.read_csv(food_cuisine_survey_fn)
		# Filter out users who didn't pay attention
		attention_question = 'Select only the ingredient that is Broccoli'
		self.food_cuisine_survey_df = self.food_cuisine_survey_df[self.food_cuisine_survey_df[attention_question] == 'Broccoli']
		# Parse cuisine's direct and indirect (knowledge) familiarities and preferences, surprise recipes' scores and preferences
		# Initialize variables
		familiar_str = 'How familiar are you with '
		preference_str = 'How much do you enjoy eating '
		knowledge_str = 'Which of the following ingredients are commonly found in '
		surprise_rating_str = 'How surprising did you find this recipe?'
		surprise_preference_str = "How much do you think you'd enjoy eating this dish?"
		# Iterate over the columns
		for each_col in self.food_cuisine_survey_df.columns:
			# Get cuisine familiarity column
			self.get_cuisine_fam_pref(familiar_str, '_fam_dir', each_col)
			# Get cuisine preference column
			self.get_cuisine_fam_pref(preference_str, '_cuisine_pref', each_col)
			# Get surprise ratings column
			self.get_surprise_fam_pref(surprise_rating_str, '_surprise_rating', each_col)
			# Get surprise preference column
			self.get_surprise_fam_pref(surprise_preference_str, '_surprise_preference', each_col)
			# Get knowledge column
			self.get_knowledge_fam(knowledge_str, each_col)
		# Calculate modern familiarity direct (self-reported)
		_familiar_cols_ = [col for col in self.food_cuisine_survey_df if 'fam_dir' in col]
		self.food_cuisine_survey_df['modern_fam_dir'] = self.food_cuisine_survey_df[_familiar_cols_].mean(axis=1)
		# Calculate knowledge score per cuisine (0-5)
		knowledge_cols = [col for col in self.food_cuisine_survey_df if 'knowledge' in col]
		cuisine_names = set([each_col.split('_')[0] for each_col in knowledge_cols])
		for each_cuisine in cuisine_names:
			cuisine_ingredient_cols = [col for col in knowledge_cols if each_cuisine in col]
			self.food_cuisine_survey_df[each_cuisine + '_cuisine_knowledge'] = self.food_cuisine_survey_df[cuisine_ingredient_cols].sum(axis=1)
		# Calculate knowledge for modern
		_cuisine_knowledge_cols_ = [col for col in self.food_cuisine_survey_df if 'cuisine_knowledge' in col]
		self.food_cuisine_survey_df['modern_cuisine_knowledge'] = self.food_cuisine_survey_df[_cuisine_knowledge_cols_].mean(axis=1)
		# Get the column names for each of the user input set
		# Fix this array as the columns for direct familiarity in this order
		_familiar_cols_ = [_each_ + '_fam_dir' for _each_ in _fam_cats_sorted_]
		_preference_cols_ = [col for col in self.food_cuisine_survey_df if 'cuisine_pref' in col]
		_knowledge_ingredient_cols_ = [col for col in self.food_cuisine_survey_df if '_knowledge_ingredient_' in col]
		# _cuisine_knowledge_cols_ = [col for col in self.food_cuisine_survey_df if '_cuisine_knowledge' in col]
		# _cuisine_knowledge_cols_ = ['mexican_cuisine_knowledge', 'chinese_cuisine_knowledge', 'modern_cuisine_knowledge',
		# 	'greek_cuisine_knowledge', 'indian_cuisine_knowledge', 'thai_cuisine_knowledge', 'italian_cuisine_knowledge']
		_cuisine_knowledge_cols_ = [_each_ + '_cuisine_knowledge' for _each_ in _fam_cats_sorted_]
		_surprise_preference_cols_ = [col for col in self.food_cuisine_survey_df if 'surprise_preference' in col]
		_surprise_rating_cols_ = [col for col in self.food_cuisine_survey_df if 'surprise_rating' in col]
		# Create dictionaries for the users' input
		users_fam_dir_dict = self.create_users_input_dict(_familiar_cols_, 5.0)
		users_cuisine_pref_dict = self.create_users_input_dict(_preference_cols_, 5.0)
		users_knowledge_ingredient_dict = self.create_users_input_dict(_knowledge_ingredient_cols_, 5.0)
		users_knowledge_cuisine_dict = self.create_users_input_dict(_cuisine_knowledge_cols_, 5.0)
		users_surp_ratings_dict = self.create_users_input_dict(_surprise_rating_cols_, 5.0)
		users_surp_pref_dict = self.create_users_input_dict(_surprise_preference_cols_, 5.0)
		# Return all of the dictionaries
		return users_fam_dir_dict, _familiar_cols_, users_cuisine_pref_dict, _preference_cols_, \
			   users_knowledge_ingredient_dict, _knowledge_ingredient_cols_, users_knowledge_cuisine_dict, _cuisine_knowledge_cols_, \
			   users_surp_ratings_dict, _surprise_rating_cols_, users_surp_pref_dict, _surprise_preference_cols_

########################################################################################################################

if __name__ == "__main__":
	start_time = time.time()
	# Get the current dir's path
	cwd = os.getcwd()
	# Parse arguments from the command
	parser = argparse.ArgumentParser(description="Evaluate a dataset using a trained GloVex model.")
	parser.add_argument("inputfile", help='The input file path to work with (omit the args and suffix)')
	parser.add_argument("--user_survey", default=None, type=str, help='The input file path to the user survey')
	parser.add_argument("--surprise_recipes", default=None, type=str, help='The input file path to the surprise recipes')
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
	fam_cat_sorted = reader.famcats

	# Load/train personalised models
	models = []
	model_famcat = dict()
	for fc in fam_cat_sorted:
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

	# # Get all of the surprises for all of the combinations
	# top_surps = []
	# all_comb_surps_per_cuisine = collections.defaultdict(dict)
	# for famcat in model_famcat:
	# 	print('Store all surprise combinations for', famcat)
	# 	all_comb_suprs = preprocessor.save_surps(model_famcat[famcat], reader, famcat=famcat)
	# 	# pp.pprint(all_comb_suprs)
	# 	for each_comb in all_comb_suprs:
	# 		# print(str(each_comb))
	# 		all_comb_surps_per_cuisine[frozenset(each_comb)][famcat] = all_comb_suprs[each_comb]
	# 	# print(all_comb_surps_per_cuisine)
	# print('Number of combinations in all_comb_surps_per_cuisine:', len(all_comb_surps_per_cuisine))
	# # print(all_comb_surps_per_cuisine)

	# Get familiarity category of the data from the user survey data
	# The user's familiarity scores for each cuisine (score: between 0-1) [[0.3, 0.3, 0.5, ], [], [] ... ]
	# Order of cuisines[mexican, chinese, modern, greek, indian, thai, italian]
	if not args.user_survey is None:
		# Construct a survey_reader object
		survey_reader_obj = survey_reader()
		# Read the survey
		users_fam_dir, familiar_cols, users_cuisine_pref, preference_cols, \
		users_knowledge_ingredient, knowledge_ingredient_cols, users_knowledge_cuisine, cuisine_knowledge_cols, \
		users_surp_ratings, surprise_rating_cols, users_surp_pref, surprise_preference_cols = \
			survey_reader_obj.read_survey(args.user_survey, fam_cat_sorted)
	else:
		print 'There is no user data, an assumed user will be modeled instead'
		user_fam_arr = [
			[0.5, 0.6, 0.8, 0.6, 0.3, 0.7, 0.5],
			[0.4, 0.3, 0.2, 0.8, 0.6, 0.7, 0.2]
		]
		# Temporarily assign an empty dict
		users_fam_dir, users_knowledge_cuisine = {}, {}
		print 'users_fam', user_fam_arr
		logger.info(" ** Generated fake user familiarity profile: " + ", ".join(
			[str(fc) + ": " + str(f) for f, fc in zip(user_fam_arr[0], reader.famcats)]))

	# Evaluate personalized surprise model
	print 'Number of users:', len(users_knowledge_cuisine)
	# Initialize the user surprise estimates
	user_suprise_estimates = {}
	all_comb_surps_per_user = collections.defaultdict(dict)
	print 'Store all_comb_surps_per_user:'
	# Repeat for each user
	for user_idx in users_knowledge_cuisine:
		user_fam_scores = users_knowledge_cuisine[user_idx]
		print "User's familiarity", user_idx, user_fam_scores
		# Store the fam cat into the dict
		user_suprise_estimates[user_idx] = {'user_fam_scores': user_fam_scores, 'recipes_surp': {}}
		# Store the surprises; there are two ways to evaluate the surprise recipes
		dataset_surps = eval_personalised_dataset_surprise(models, reader, user_fam_scores, offset=73106)
		# Get the unique set of surprises for this user
		unique_surps = set((p for s in dataset_surps for p in s["surprises"]))
		# Iterate over the unique surps to store in all_comb_surps_per_user
		for each_comb in unique_surps:
			# Store the combined the surprise scores of all cuisines (double keys: primary: ingredient combinations, secondary: user ID)
			all_comb_surps_per_user[frozenset((each_comb[0], each_comb[1]))][user_idx] = each_comb[2]
		# For the surprising receipes, store their recipe ID, surprise, raw doc and surprise cuisine scores
		recipe_surp_dict = {}
		for doc in dataset_surps:
			# pp.pprint(doc)
			recipe_surp_dict['95th_percentile'] = doc['surprise_95']
			recipe_surp_dict['90th_percentile'] = doc['surprise_90']
			recipe_surp_dict['ingredients'] = doc['raw']
			recipe_surp_dict['surprise_cuisine'] = []
			for surprise_combination in doc['surprises']:
				recipe_surp_dict['surprise_cuisine'].append(surprise_combination)
			# Append the recipe_surp_dict to the user_suprise_estimates
			# user_suprise_estimates[user_idx]['recipes_surp'].append(recipe_surp_dict)
			doc_id_int = int(doc['id'])
			user_suprise_estimates[user_idx]['recipes_surp'][doc_id_int] = recipe_surp_dict
			# Renew/empty the dict for next iteration
			recipe_surp_dict = {}
	print 'Number of combinations in all_comb_surps_per_user:', len(all_comb_surps_per_user)

	# Store the user_suprise_estimates in a pickle
	# user_suprise_estimates_pickle_fn = cwd + '/GloVex/results/second_survey/user_suprise_estimates.pickle'
	print 'Storing user_suprise_estimates:'
	user_suprise_estimates_pickle_fn = cwd + '/GloVex/results/' + args.dataset + '_knowledge/user_suprise_estimates.pickle'
	pickle.dump(user_suprise_estimates, open(user_suprise_estimates_pickle_fn, 'wb'))

	# # Test if all combinations are the same in both dicts
	# print('Number of combinations in all_comb_surps_per_cuisine:', len(all_comb_surps_per_cuisine))
	# print('Number of combinations in all_comb_surps_per_user:', len(all_comb_surps_per_user))
	# print('Are all keys in the two dicts equal?',
	# 	set(all_comb_surps_per_cuisine.keys()) == set(all_comb_surps_per_user.keys()))
	# combinations_instersection = \
	# 	set(all_comb_surps_per_cuisine.keys()).intersection(set(all_comb_surps_per_user.keys()))
	# print('Number of intersecting keys between all_comb_surps_per_cuisine and all_comb_surps_per_user',
	# 	len(combinations_instersection))
	# # Combine all_surp per cuisine and per user
	# all_comb_surps_dict = collections.defaultdict(dict)
	# # Iterate over the ingredient combinations
	# # for each_comb in all_comb_surps_per_cuisine.keys():
	# for each_comb in combinations_instersection:
	# 	# print(each_comb)
	# 	# Add the the per_cuisine data
	# 	all_comb_surps_dict[str(each_comb)]['per_cuisine'] = all_comb_surps_per_cuisine[each_comb]
	# 	all_comb_surps_dict[str(each_comb)]['per_user'] = all_comb_surps_per_user[each_comb]
	# 	# # If there is combination exists in the all_comb_surps_per_user, add the the per_user data
	# 	# if each_comb in all_comb_surps_per_user:
	# 	# 	all_comb_surps_dict[str(each_comb)]['per_user'] = all_comb_surps_per_user[each_comb]
	# # Store the all_comb_surps_dict in a JSON
	# # all_comb_surps_fn = cwd + '/GloVex/results/second_survey/all_comb_surps.json'
	# all_comb_surps_fn = cwd + '/GloVex/results/' + args.dataset + '_knowledge/all_comb_surps.json'
	# json.dump(all_comb_surps_dict, open(all_comb_surps_fn, 'w'), indent=4)
	# print 'Time taken to run the whole script:', (time.time() - start_time) / 60, 'minutes'
