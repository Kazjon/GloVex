
# coding: utf-8

# Libraries

# In[1]:


import os, pickle, numpy as np
from GloVex.evaluate_personalised import survey_reader
from scipy.stats import pearsonr, spearmanr
cwd = os.getcwd()


# Helper functions

# In[2]:


def averageing_fn(surp_list):
	_avg_ = []
	for each_ in surp_list:
		_avg_.append(reduce(lambda x, y: x + y, each_) / len(each_))
	return _avg_


# Get the surprises of the users in the survey

# In[3]:

print 'Get the surprises of the users in the survey:'
survey_reader_obj = survey_reader()
food_cuisine_survey_fn = cwd + '/personalized-surprise/Food and Cuisine Preferences Survey (Responses) - Form Responses 2'
fam_cat_sorted = ['mexican', 'chinese', 'modern', 'greek', 'indian', 'thai', 'italian']
users_fam_dir, familiar_cols, \
users_cuisine_pref, preference_cols, \
users_knowledge_ingredient, knowledge_ingredient_cols,\
users_knowledge_cuisine, cuisine_knowledge_cols,\
users_surp_ratings, surprise_rating_cols,\
users_surp_pref, surprise_preference_cols = \
	survey_reader_obj.read_survey(food_cuisine_survey_fn, fam_cat_sorted)

# Prepare array of familiarity scores from the dict

# In[4]:

print 'Prepare array of familiarity scores from the dict:'
# Take the familiarity scores from the dict into arrays
user_fam_arr = []
for each_user in sorted(users_fam_dir):
	user_fam_arr.append(users_fam_dir[each_user])
# Take the surprise scores from the dict into arrays
users_surp_ratings_arr = []
for each_user in sorted(users_surp_ratings):
	users_surp_ratings_arr.append(users_surp_ratings[each_user])


# Transpose for recipe level comparison

# In[5]:

print 'Transpose for recipe level comparison:'
users_surp_ratings_userlvl = users_surp_ratings_arr
users_surp_ratings_recipelvl = np.transpose(users_surp_ratings_arr).tolist()


# Flatten the arrays in both dimenssions

# In[6]:


users_surp_ratings_user_flat = [item for sublist in users_surp_ratings_userlvl for item in sublist]
users_surp_ratings_recipe_flat = [item for sublist in users_surp_ratings_recipelvl for item in sublist]


# Divide the users into foodies and novices using the median of sum of familiarities

# In[7]:


# Array holding all of the sums
sum_arr = []
# Iterate over the users' familiarities
# for user_idx in sorted(users_fam_dir.keys()):
for user_idx in sorted(users_knowledge_cuisine.keys()):
	# Get the familiarity scores
	# each_fam = users_fam_dir[user_idx]
	each_fam = users_knowledge_cuisine[user_idx]
	# Calculate the sum of familiarity for each user
	sum_fam = sum(each_fam)
	# Append the sum of familiarity of each user to the sum array
	sum_arr.append(sum_fam)
median_fam = np.median(sum_arr)
# Split the users into foodeis and novices using the median
foodie_fam_dict = {}
foodie_surp_ratings_dict = {}
novice_fam_dict = {}
novice_surp_ratings_dict = {}
# Iterate over the users' familiarities
# for user_idx in sorted(users_fam_dir.keys()):
for user_idx in sorted(users_knowledge_cuisine.keys()):
	# Get the familiarity and surprise scores
	# each_fam = users_fam_dir[user_idx]
	each_fam = users_knowledge_cuisine[user_idx]
	each_surp = users_surp_ratings[user_idx]
	# Calculate the sum of familiarity for each user
	sum_fam = sum(each_fam)
	if sum_fam > median_fam:
		# print 'above median', each_fam
		foodie_fam_dict[user_idx] = each_fam
		foodie_surp_ratings_dict[user_idx] = each_surp
	else:
		# print 'below median', each_fam
		novice_fam_dict[user_idx] = each_fam
		novice_surp_ratings_dict[user_idx] = each_surp
# List the foodies and novices in two separate lists
foodie_users = sorted(foodie_fam_dict.keys())
novice_users = sorted(novice_fam_dict.keys())


# Get the personal surprise estimates by the model

# In[8]:

print 'Get the personal surprise estimates by the model:'
# familiarity_measure = 'recipes_self_reported'
# familiarity_measure = 'recipes_knowledge'
familiarity_measure = 'recipes_averaged'
user_suprise_estimates_pickle_fn = cwd + '/GloVex/results/' + familiarity_measure + '/user_suprise_estimates.pickle'
user_suprise_estimates_dict = pickle.load(open(user_suprise_estimates_pickle_fn,'rb'))


# Get the 90th, 95th percentils and max surprise estimates

# In[9]:


# Initialize surprise estimates for 95th percentile and max score
personalized_surp_estimates_90perc = []
personalized_surp_estimates_90perc_dict = {}
personalized_surp_estimates_95perc = []
personalized_surp_estimates_95perc_dict = {}
personalized_surp_estimates_max = []
personalized_surp_estimates_max_dict = {}
# Iterate over the user surprise estimeates dict
for each_user in sorted(user_suprise_estimates_dict.keys()):
	# print 'each_user', each_user
	# Initialize the inner arrays for 95th percentile and max score
	users_surp_estimates_90perc = []
	users_surp_estimates_95perc = []
	users_surp_estimates_max = []
	# Iterate over the surprises for each user's surprise estimates
	for each_recipe in sorted(user_suprise_estimates_dict[each_user]['recipes_surp']):
		# print 'each_recipe', each_recipe
		# Get the 90th and 95th percentiles
		user_suprise_90th_percentile = user_suprise_estimates_dict[each_user]['recipes_surp'][each_recipe]['90th_percentile']
		user_suprise_95th_percentile = user_suprise_estimates_dict[each_user]['recipes_surp'][each_recipe]['95th_percentile']
		# Get the max
		user_suprise_max = max([each_ingr_comb[2] 
			for each_ingr_comb in user_suprise_estimates_dict[each_user]['recipes_surp'][each_recipe]['surprise_cuisine']])
		# Append the 90th and 95th percentile and maxfor the surprise estimates
		users_surp_estimates_90perc.append(user_suprise_90th_percentile)
		users_surp_estimates_95perc.append(user_suprise_95th_percentile)
		users_surp_estimates_max.append(user_suprise_max)
	# Append the user's surprise estimates to the personalized main 2D array (90th and 95th percentile and max score)
	personalized_surp_estimates_90perc.append(users_surp_estimates_90perc)
	personalized_surp_estimates_95perc.append(users_surp_estimates_95perc)
	personalized_surp_estimates_max.append(users_surp_estimates_max)
	# Add the user's surprise estimates to the dict
	personalized_surp_estimates_90perc_dict[each_user] = users_surp_estimates_90perc
	personalized_surp_estimates_95perc_dict[each_user] = users_surp_estimates_95perc
	personalized_surp_estimates_max_dict[each_user] = users_surp_estimates_max


# Transpose for recipe level comparison

# In[10]:


personalized_surp_estimates_recipelvl_90perc = np.transpose(personalized_surp_estimates_90perc).tolist()
personalized_surp_estimates_recipelvl_95perc = np.transpose(personalized_surp_estimates_95perc).tolist()
personalized_surp_estimates_recipelvl_max = np.transpose(personalized_surp_estimates_max).tolist()


# Flatten the arrays in both dimnessions

# In[11]:


personalized_surp_estimates_userflat_90perc = [item for sublist in personalized_surp_estimates_90perc for item in sublist]
personalized_surp_estimates_userflat_95perc = [item for sublist in personalized_surp_estimates_95perc for item in sublist]
personalized_surp_estimates_userflat_max = [item for sublist in personalized_surp_estimates_max for item in sublist]
personalized_surp_estimates_recipeflat_90perc = [item for sublist in personalized_surp_estimates_recipelvl_90perc for item in sublist]
personalized_surp_estimates_recipeflat_95perc = [item for sublist in personalized_surp_estimates_recipelvl_95perc for item in sublist]
personalized_surp_estimates_recipeflat_max = [item for sublist in personalized_surp_estimates_recipelvl_max for item in sublist]


# Separate the foodies from novices

# In[12]:


# Get the foodies for the 90th and 95th percentiles and the max
foodie_personalized_surp_estimates_90perc_dict = {each_: personalized_surp_estimates_90perc_dict[each_] 
	for each_ in personalized_surp_estimates_90perc_dict.keys() if each_ in foodie_surp_ratings_dict.keys()}
foodie_personalized_surp_estimates_95perc_dict = {each_: personalized_surp_estimates_95perc_dict[each_] 
	for each_ in personalized_surp_estimates_95perc_dict.keys() if each_ in foodie_surp_ratings_dict.keys()}
foodie_personalized_surp_estimates_max_dict = {each_: personalized_surp_estimates_max_dict[each_] 
	for each_ in personalized_surp_estimates_max_dict.keys() if each_ in foodie_surp_ratings_dict.keys()}
# Get the novices for the 90th and 95th percentiles and the max
novice_personalized_surp_estimates_90perc_dict = {each_: personalized_surp_estimates_90perc_dict[each_] 
	for each_ in personalized_surp_estimates_90perc_dict.keys() if each_ in novice_surp_ratings_dict.keys()}
novice_personalized_surp_estimates_95perc_dict = {each_: personalized_surp_estimates_95perc_dict[each_] 
	for each_ in personalized_surp_estimates_95perc_dict.keys() if each_ in novice_surp_ratings_dict.keys()}
novice_personalized_surp_estimates_max_dict = {each_: personalized_surp_estimates_max_dict[each_] 
	for each_ in personalized_surp_estimates_max_dict.keys() if each_ in novice_surp_ratings_dict.keys()}


# Transfer the foodies' and novices' into arrays using the same iterator's index for the ratings and estimates (at the same time to geuarantee synchronization)

# In[13]:


# Foodies
foodie_surp_ratings = []
foodie_personalized_surp_estimates_90perc = []
foodie_personalized_surp_estimates_95perc = []
foodie_personalized_surp_estimates_max = []
for each_foodie in sorted(foodie_users):
	foodie_surp_ratings.append(foodie_surp_ratings_dict[each_foodie])
	foodie_personalized_surp_estimates_90perc.append(foodie_personalized_surp_estimates_90perc_dict[each_foodie])
	foodie_personalized_surp_estimates_95perc.append(foodie_personalized_surp_estimates_95perc_dict[each_foodie])
	foodie_personalized_surp_estimates_max.append(foodie_personalized_surp_estimates_max_dict[each_foodie])
# Novices
novice_surp_ratings = []
novice_personalized_surp_estimates_90perc = []
novice_personalized_surp_estimates_95perc = []
novice_personalized_surp_estimates_max = []
for each_novice in sorted(novice_users):
	novice_surp_ratings.append(novice_surp_ratings_dict[each_novice])
	novice_personalized_surp_estimates_90perc.append(novice_personalized_surp_estimates_90perc_dict[each_novice])
	novice_personalized_surp_estimates_95perc.append(novice_personalized_surp_estimates_95perc_dict[each_novice])
	novice_personalized_surp_estimates_max.append(novice_personalized_surp_estimates_max_dict[each_novice])


# Foodies and novices user level and recipe levels arrays and flatten them for surprise ratings

# In[14]:


# Foodies surprise ratings user and recipe level
foodie_surp_ratings_userlvl = foodie_surp_ratings
foodie_surp_ratings_recipelvl = np.transpose(foodie_surp_ratings_userlvl).tolist()
# Foodies surprise ratings flattened on user and recipe level
foodie_surp_ratings_user_flat = [item for sublist in foodie_surp_ratings_userlvl for item in sublist]
foodie_surp_ratings_recipe_flat = [item for sublist in foodie_surp_ratings_recipelvl for item in sublist]
# Novices surprise ratings user and recipe level
novice_surp_ratings_userlvl = novice_surp_ratings
novice_surp_ratings_recipelvl = np.transpose(novice_surp_ratings_userlvl).tolist()
# Novices surprise ratings flattened on user and recipe level
novice_surp_ratings_user_flat = [item for sublist in novice_surp_ratings_userlvl for item in sublist]
novice_surp_ratings_recipe_flat = [item for sublist in novice_surp_ratings_recipelvl for item in sublist]


# Flatten the foodies' and novices' personalized for surprise estimates

# In[15]:


# Foodies surprise estimates on user level
foodie_personalized_surp_estimates_userflat_90perc = [item for sublist in foodie_personalized_surp_estimates_90perc for item in sublist]
foodie_personalized_surp_estimates_userflat_95perc = [item for sublist in foodie_personalized_surp_estimates_95perc for item in sublist]
foodie_personalized_surp_estimates_userflat_max = [item for sublist in foodie_personalized_surp_estimates_max for item in sublist]
# Foodies surprise estimates on recipe level
foodie_personalized_surp_estimates_recipelvl_90perc = np.transpose(foodie_personalized_surp_estimates_90perc).tolist()
foodie_personalized_surp_estimates_recipeflat_90perc = [item for sublist in foodie_personalized_surp_estimates_recipelvl_90perc for item in sublist]
foodie_personalized_surp_estimates_recipelvl_95perc = np.transpose(foodie_personalized_surp_estimates_95perc).tolist()
foodie_personalized_surp_estimates_recipeflat_95perc = [item for sublist in foodie_personalized_surp_estimates_recipelvl_95perc for item in sublist]
foodie_personalized_surp_estimates_recipelvl_max = np.transpose(foodie_personalized_surp_estimates_max).tolist()
foodie_personalized_surp_estimates_recipeflat_max = [item for sublist in foodie_personalized_surp_estimates_recipelvl_max for item in sublist]
# Novices surprise estimates on user level
novice_personalized_surp_estimates_userflat_90perc = [item for sublist in novice_personalized_surp_estimates_90perc for item in sublist]
novice_personalized_surp_estimates_userflat_95perc = [item for sublist in novice_personalized_surp_estimates_95perc for item in sublist]
novice_personalized_surp_estimates_userflat_max = [item for sublist in novice_personalized_surp_estimates_max for item in sublist]
# Novices surprise estimates on recipe level
novice_personalized_surp_estimates_recipelvl_90perc = np.transpose(novice_personalized_surp_estimates_90perc).tolist()
novice_personalized_surp_estimates_recipeflat_90perc = [item for sublist in novice_personalized_surp_estimates_recipelvl_90perc for item in sublist]
novice_personalized_surp_estimates_recipelvl_95perc = np.transpose(novice_personalized_surp_estimates_95perc).tolist()
novice_personalized_surp_estimates_recipeflat_95perc = [item for sublist in novice_personalized_surp_estimates_recipelvl_95perc for item in sublist]
novice_personalized_surp_estimates_recipelvl_max = np.transpose(novice_personalized_surp_estimates_max).tolist()
novice_personalized_surp_estimates_recipeflat_max = [item for sublist in novice_personalized_surp_estimates_recipelvl_max for item in sublist]


# Get the oracle surprise estimates by the model

# In[16]:

print 'Get the oracle surprise estimates by the model:'
oracle_suprise_pickle_fn = cwd + '/GloVex/results/recipes_oracle/oracle_suprise_run4.pickle'
oracle_suprise_dict = pickle.load(open(oracle_suprise_pickle_fn, 'rb'))


# Get the oracle's surprise estimates (90th and 95th percentiles and max score)

# In[17]:


# Initialize arrays
oracle_surp_estimates_90perc = []
oracle_surp_estimates_95perc = []
oracle_surp_estimates_max = []
observed_surp_estimates_90perc = []
observed_surp_estimates_95perc = []
observed_surp_estimates_max = []
# Iterate over the dict
for each_recipe in sorted(oracle_suprise_dict):
	# print 'each_recipe', each_recipe
	# Get all of the ingredients combinations' surprise estimates
	oracle_ingr_surp_estimates = [each_ingr_comb[2] for each_ingr_comb in oracle_suprise_dict[each_recipe]['surprise_cuisine']]
	observed_ingr_surp_estimates = [each_ingr_comb[2] for each_ingr_comb in oracle_suprise_dict[each_recipe]['observed_surprise_cuisine']]
	# Append the 90th and 95th_percentiles and max surprise estimates
	oracle_surp_estimates_90perc.append(oracle_suprise_dict[each_recipe]['90th_percentile'])
	oracle_surp_estimates_95perc.append(oracle_suprise_dict[each_recipe]['95th_percentile'])
	oracle_surp_estimates_max.append(max(oracle_ingr_surp_estimates))
	observed_surp_estimates_90perc.append(oracle_suprise_dict[each_recipe]['observed_90th_percentile'])
	observed_surp_estimates_95perc.append(oracle_suprise_dict[each_recipe]['observed_95th_percentile'])
	observed_surp_estimates_max.append(max(observed_ingr_surp_estimates))


# Find correlation between surprise scores

# Function for calculating correlation per user

# In[18]:


# Function for calculating the correlation per user
def corr_per_user(oracle_surp, personalized_surp_estimates, users_surp_ratings_userlvl):
	for user_idx, (each_personal_score, each_users_rating) in enumerate(zip(personalized_surp_estimates, users_surp_ratings_userlvl)):
		# print user_idx, each_personal_score, each_users_rating
		# print 'oracle_surp: spearmanr', spearmanr(oracle_surp, each_users_rating)
		# print 'each_personal_score: spearmanr', spearmanr(each_personal_score, each_users_rating)
		oracles_pearsonr_corr, oracles_pearsonr_pvalue = pearsonr(oracle_surp, each_users_rating)
		personals_pearsonr_corr, personals_pearsonr_pvalue = pearsonr(each_personal_score, each_users_rating)
		if oracles_pearsonr_pvalue <= 0.05:
			print user_idx, 'oracle_surp: pearsonr', oracles_pearsonr_corr, oracles_pearsonr_pvalue
		if personals_pearsonr_pvalue <= 0.05:
			print user_idx, 'each_personal_score: pearsonr', personals_pearsonr_corr, personals_pearsonr_pvalue


# All users personalized with: Oracle and Observed and users' surprise ratings (90th, 95th and max)

# In[ ]:


corr_per_user(oracle_surp_estimates_90perc, personalized_surp_estimates_90perc, users_surp_ratings_userlvl)


# In[ ]:


corr_per_user(observed_surp_estimates_90perc, personalized_surp_estimates_90perc, users_surp_ratings_userlvl)


# In[ ]:


corr_per_user(oracle_surp_estimates_95perc, personalized_surp_estimates_95perc, users_surp_ratings_userlvl)


# In[ ]:


corr_per_user(observed_surp_estimates_95perc, personalized_surp_estimates_95perc, users_surp_ratings_userlvl)


# In[ ]:


corr_per_user(oracle_surp_estimates_max, personalized_surp_estimates_max, users_surp_ratings_userlvl)


# In[ ]:


corr_per_user(observed_surp_estimates_max, personalized_surp_estimates_max, users_surp_ratings_userlvl)


# Foodies personalized with: Oracle and Observed and users' surprise ratings (90th, 95th and max)

# In[ ]:


corr_per_user(oracle_surp_estimates_90perc, foodie_personalized_surp_estimates_90perc, foodie_surp_ratings_userlvl)


# In[ ]:


corr_per_user(observed_surp_estimates_90perc, foodie_personalized_surp_estimates_90perc, foodie_surp_ratings_userlvl)


# In[ ]:


corr_per_user(oracle_surp_estimates_95perc, foodie_personalized_surp_estimates_95perc, foodie_surp_ratings_userlvl)


# In[ ]:


corr_per_user(observed_surp_estimates_95perc, foodie_personalized_surp_estimates_95perc, foodie_surp_ratings_userlvl)


# In[ ]:


corr_per_user(oracle_surp_estimates_max, foodie_personalized_surp_estimates_max, foodie_surp_ratings_userlvl)


# In[ ]:


corr_per_user(observed_surp_estimates_max, foodie_personalized_surp_estimates_max, foodie_surp_ratings_userlvl)


# Novices personalized with: Oracle and Observed and users' surprise ratings (90th, 95th and max)

# In[ ]:


corr_per_user(oracle_surp_estimates_90perc, novice_personalized_surp_estimates_90perc, novice_surp_ratings_userlvl)


# In[ ]:


corr_per_user(observed_surp_estimates_90perc, novice_personalized_surp_estimates_90perc, novice_surp_ratings_userlvl)


# In[ ]:


corr_per_user(oracle_surp_estimates_95perc, novice_personalized_surp_estimates_95perc, novice_surp_ratings_userlvl)


# In[ ]:


corr_per_user(observed_surp_estimates_95perc, novice_personalized_surp_estimates_95perc, novice_surp_ratings_userlvl)


# In[ ]:


corr_per_user(oracle_surp_estimates_max, novice_personalized_surp_estimates_max, novice_surp_ratings_userlvl)


# In[ ]:


corr_per_user(observed_surp_estimates_max, novice_personalized_surp_estimates_max, novice_surp_ratings_userlvl)


# Function for calculating correlation per recipe

# In[19]:


def corr_per_recipe(personalized_surp_estimates_recipelvl, users_surp_ratings_recipelvl):
	for user_idx, (each_personal_score, each_users_rating) in enumerate(zip(personalized_surp_estimates_recipelvl,
	  users_surp_ratings_recipelvl)):
		# print user_idx, each_personal_score, each_users_rating
		# print 'each_personal_score: spearmanr', spearmanr(each_personal_score, each_users_rating)
		personals_pearson_corr, personals_pearson_pvalue = pearsonr(each_personal_score, each_users_rating)
		if personals_pearson_pvalue <= 0.05:
			print user_idx, 'each_personal_score: pearsonr', personals_pearson_corr, personals_pearson_pvalue


# All users personalized with: users' surprise ratings (90th, 95th and max)

# In[ ]:


corr_per_recipe(personalized_surp_estimates_recipelvl_95perc, users_surp_ratings_recipelvl)
corr_per_recipe(personalized_surp_estimates_recipelvl_90perc, users_surp_ratings_recipelvl)
corr_per_recipe(personalized_surp_estimates_recipelvl_max, users_surp_ratings_recipelvl)


# Foodies personalized with: users' surprise ratings (90th, 95th and max)

# In[ ]:


corr_per_recipe(foodie_personalized_surp_estimates_recipelvl_95perc, foodie_surp_ratings_recipelvl)
corr_per_recipe(foodie_personalized_surp_estimates_recipelvl_90perc, foodie_surp_ratings_recipelvl)
corr_per_recipe(foodie_personalized_surp_estimates_recipelvl_max, foodie_surp_ratings_recipelvl)


# Novices personalized with: users' surprise ratings (90th, 95th and max)

# In[ ]:


corr_per_recipe(novice_personalized_surp_estimates_recipelvl_90perc, novice_surp_ratings_recipelvl)
corr_per_recipe(novice_personalized_surp_estimates_recipelvl_95perc, novice_surp_ratings_recipelvl)
corr_per_recipe(novice_personalized_surp_estimates_recipelvl_max, novice_surp_ratings_recipelvl)


# Get averages for the recipe level arrays

# In[20]:


# All, foodies and novices users ratings
users_surp_ratings_recipelvl_avg = averageing_fn(users_surp_ratings_recipelvl)
foodie_surp_ratings_recipelvl_avg = averageing_fn(foodie_surp_ratings_recipelvl)
novice_surp_ratings_recipelvl_avg = averageing_fn(novice_surp_ratings_recipelvl)
# All 90th 95th perc and max estimates
personalized_surp_estimates_recipelvl_90perc_avg = averageing_fn(personalized_surp_estimates_recipelvl_90perc)
personalized_surp_estimates_recipelvl_95perc_avg = averageing_fn(personalized_surp_estimates_recipelvl_95perc)
personalized_surp_estimates_recipelvl_max_avg = averageing_fn(personalized_surp_estimates_recipelvl_max)
# Foodies 90th 95th perc and max estimates
foodie_personalized_surp_estimates_recipelvl_90perc_avg = averageing_fn(foodie_personalized_surp_estimates_recipelvl_90perc)
foodie_personalized_surp_estimates_recipelvl_95perc_avg = averageing_fn(foodie_personalized_surp_estimates_recipelvl_95perc)
foodie_personalized_surp_estimates_recipelvl_max_avg = averageing_fn(foodie_personalized_surp_estimates_recipelvl_max)
# Novices 90th 95th perc and max estimates
novice_personalized_surp_estimates_recipelvl_90perc_avg = averageing_fn(novice_personalized_surp_estimates_recipelvl_90perc)
novice_personalized_surp_estimates_recipelvl_95perc_avg = averageing_fn(novice_personalized_surp_estimates_recipelvl_95perc)
novice_personalized_surp_estimates_recipelvl_max_avg = averageing_fn(novice_personalized_surp_estimates_recipelvl_max)


# Correlation of the average

# All users' personalized surprise estimates with users' ratings

# In[21]:

print "# All users' personalized surprise estimates with users' ratings"
# print 'spearmanr', spearmanr(personalized_surp_estimates_recipelvl_90perc_avg, users_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(personalized_surp_estimates_recipelvl_90perc_avg, users_surp_ratings_recipelvl_avg)
# print 'spearmanr', spearmanr(personalized_surp_estimates_recipelvl_95perc_avg, users_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(personalized_surp_estimates_recipelvl_95perc_avg, users_surp_ratings_recipelvl_avg)
# print 'spearmanr', spearmanr(personalized_surp_estimates_recipelvl_max_avg, users_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(personalized_surp_estimates_recipelvl_max_avg, users_surp_ratings_recipelvl_avg)


# Foodies personalized surprise estimates with users' ratings

# In[22]:

print "Foodies personalized surprise estimates with users' ratings"
# print 'spearmanr', spearmanr(foodie_personalized_surp_estimates_recipelvl_90perc_avg, foodie_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(foodie_personalized_surp_estimates_recipelvl_90perc_avg, foodie_surp_ratings_recipelvl_avg)
# print 'spearmanr', spearmanr(foodie_personalized_surp_estimates_recipelvl_95perc_avg, foodie_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(foodie_personalized_surp_estimates_recipelvl_95perc_avg, foodie_surp_ratings_recipelvl_avg)
# print 'spearmanr', spearmanr(foodie_personalized_surp_estimates_recipelvl_max_avg, foodie_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(foodie_personalized_surp_estimates_recipelvl_max_avg, foodie_surp_ratings_recipelvl_avg)


# Novices personalized surprise estimates with users' ratings

# In[23]:

print "Novices personalized surprise estimates with users' ratings"
# print 'spearmanr', spearmanr(novice_personalized_surp_estimates_recipelvl_90perc_avg, novice_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(novice_personalized_surp_estimates_recipelvl_90perc_avg, novice_surp_ratings_recipelvl_avg)
# print 'spearmanr', spearmanr(novice_personalized_surp_estimates_recipelvl_95perc_avg, novice_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(novice_personalized_surp_estimates_recipelvl_95perc_avg, novice_surp_ratings_recipelvl_avg)
# print 'spearmanr', spearmanr(novice_personalized_surp_estimates_recipelvl_max_avg, novice_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(novice_personalized_surp_estimates_recipelvl_max_avg, novice_surp_ratings_recipelvl_avg)


# Oracles surprise estimates of with users' ratings

# In[24]:

print "Oracles surprise estimates of with users' ratings"
# print 'spearmanr', spearmanr(oracle_surp_estimates_90perc, users_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(oracle_surp_estimates_90perc, users_surp_ratings_recipelvl_avg)
# print 'spearmanr', spearmanr(oracle_surp_estimates_95perc, users_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(oracle_surp_estimates_95perc, users_surp_ratings_recipelvl_avg)
# print 'spearmanr', spearmanr(oracle_surp_estimates_max, users_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(oracle_surp_estimates_max, users_surp_ratings_recipelvl_avg)


# Oracles surprise estimates with Foodies ratings

# In[1]:

print "Oracles surprise estimates with Foodies ratings"
# print 'spearmanr', spearmanr(oracle_surp_estimates_90perc, foodie_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(oracle_surp_estimates_90perc, foodie_surp_ratings_recipelvl_avg)
# print 'spearmanr', spearmanr(oracle_surp_estimates_95perc, foodie_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(oracle_surp_estimates_95perc, foodie_surp_ratings_recipelvl_avg)
# print 'spearmanr', spearmanr(oracle_surp_estimates_max, foodie_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(oracle_surp_estimates_max, foodie_surp_ratings_recipelvl_avg)


# Oracles surprise estimates with Novices ratings

# In[62]:

print "Oracles surprise estimates with Novices ratings"
# print 'spearmanr', spearmanr(oracle_surp_estimates_90perc, novice_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(oracle_surp_estimates_90perc, novice_surp_ratings_recipelvl_avg)
# print 'spearmanr', spearmanr(oracle_surp_estimates_95perc, novice_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(oracle_surp_estimates_95perc, novice_surp_ratings_recipelvl_avg)
# print 'spearmanr', spearmanr(oracle_surp_estimates_max, novice_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(oracle_surp_estimates_max, novice_surp_ratings_recipelvl_avg)


# Observed surprise estimates with users' ratings

# In[63]:

print "Observed surprise estimates with users' ratings"
# print 'spearmanr', spearmanr(observed_surp_estimates_90perc, users_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(observed_surp_estimates_90perc, users_surp_ratings_recipelvl_avg)
# print 'spearmanr', spearmanr(observed_surp_estimates_95perc, users_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(observed_surp_estimates_95perc, users_surp_ratings_recipelvl_avg)
# print 'spearmanr', spearmanr(observed_surp_estimates_max, users_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(observed_surp_estimates_max, users_surp_ratings_recipelvl_avg)


# Observed surprise with Foodies ratings

# In[65]:

print "Observed surprise with Foodies ratings"
# print 'spearmanr', spearmanr(observed_surp_estimates_90perc, foodie_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(observed_surp_estimates_90perc, foodie_surp_ratings_recipelvl_avg)
# print 'spearmanr', spearmanr(observed_surp_estimates_95perc, foodie_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(observed_surp_estimates_95perc, foodie_surp_ratings_recipelvl_avg)
# print 'spearmanr', spearmanr(observed_surp_estimates_max, foodie_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(observed_surp_estimates_max, foodie_surp_ratings_recipelvl_avg)


# Observed surprise with Novices ratings

# In[66]:

print "Observed surprise with Novices ratings"
# print 'spearmanr', spearmanr(observed_surp_estimates_90perc, novice_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(observed_surp_estimates_90perc, novice_surp_ratings_recipelvl_avg)
# print 'spearmanr', spearmanr(observed_surp_estimates_95perc, novice_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(observed_surp_estimates_95perc, novice_surp_ratings_recipelvl_avg)
# print 'spearmanr', spearmanr(observed_surp_estimates_max, novice_surp_ratings_recipelvl_avg)
print 'pearsonr', pearsonr(observed_surp_estimates_max, novice_surp_ratings_recipelvl_avg)

#
# # Correlations users flattened
#
# # In[28]:
#
#
# # All
# print '90th percentile'
# print 'spearmanr', spearmanr(personalized_surp_estimates_userflat_90perc, users_surp_ratings_user_flat)
# print 'pearsonr', pearsonr(personalized_surp_estimates_userflat_90perc, users_surp_ratings_user_flat)
# print '95th percentile'
# print 'spearmanr', spearmanr(personalized_surp_estimates_userflat_95perc, users_surp_ratings_user_flat)
# print 'pearsonr', pearsonr(personalized_surp_estimates_userflat_95perc, users_surp_ratings_user_flat)
# print 'Max score'
# print 'spearmanr', spearmanr(personalized_surp_estimates_userflat_max, users_surp_ratings_user_flat)
# print 'pearsonr', pearsonr(personalized_surp_estimates_userflat_max, users_surp_ratings_user_flat)
#
#
# # In[29]:
#
#
# # Foodies
# print '90th percentile'
# print 'spearmanr', spearmanr(foodie_personalized_surp_estimates_userflat_90perc, foodie_surp_ratings_user_flat)
# print 'pearsonr', pearsonr(foodie_personalized_surp_estimates_userflat_90perc, foodie_surp_ratings_user_flat)
# print '95th percentile'
# print 'spearmanr', spearmanr(foodie_personalized_surp_estimates_userflat_95perc, foodie_surp_ratings_user_flat)
# print 'pearsonr', pearsonr(foodie_personalized_surp_estimates_userflat_95perc, foodie_surp_ratings_user_flat)
# print 'Max score'
# print 'spearmanr', spearmanr(foodie_personalized_surp_estimates_userflat_max, foodie_surp_ratings_user_flat)
# print 'pearsonr', pearsonr(foodie_personalized_surp_estimates_userflat_max, foodie_surp_ratings_user_flat)
#
#
# # In[30]:
#
#
# # Novices
# print '90th percentile'
# print 'spearmanr', spearmanr(novice_personalized_surp_estimates_userflat_90perc, novice_surp_ratings_user_flat)
# print 'pearsonr', pearsonr(novice_personalized_surp_estimates_userflat_90perc, novice_surp_ratings_user_flat)
# print '95th percentile'
# print 'spearmanr', spearmanr(novice_personalized_surp_estimates_userflat_95perc, novice_surp_ratings_user_flat)
# print 'pearsonr', pearsonr(novice_personalized_surp_estimates_userflat_95perc, novice_surp_ratings_user_flat)
# print 'Max score'
# print 'spearmanr', spearmanr(novice_personalized_surp_estimates_userflat_max, novice_surp_ratings_user_flat)
# print 'pearsonr', pearsonr(novice_personalized_surp_estimates_userflat_max, novice_surp_ratings_user_flat)
#
#
# # Correlations recipes flattened
#
# # In[31]:
#
#
# # All
# print '90th percentile'
# print 'spearmanr', spearmanr(personalized_surp_estimates_recipeflat_90perc, users_surp_ratings_recipe_flat)
# print 'pearsonr', pearsonr(personalized_surp_estimates_recipeflat_90perc, users_surp_ratings_recipe_flat)
# print '95th percentile'
# print 'spearmanr', spearmanr(personalized_surp_estimates_recipeflat_95perc, users_surp_ratings_recipe_flat)
# print 'pearsonr', pearsonr(personalized_surp_estimates_recipeflat_95perc, users_surp_ratings_recipe_flat)
# print 'Max score'
# print 'spearmanr', spearmanr(personalized_surp_estimates_recipeflat_max, users_surp_ratings_recipe_flat)
# print 'pearsonr', pearsonr(personalized_surp_estimates_recipeflat_max, users_surp_ratings_recipe_flat)
#
#
# # In[32]:
#
#
# # Foodies
# print '90th percentile'
# print 'spearmanr', spearmanr(foodie_personalized_surp_estimates_recipeflat_90perc, foodie_surp_ratings_recipe_flat)
# print 'pearsonr', pearsonr(foodie_personalized_surp_estimates_recipeflat_90perc, foodie_surp_ratings_recipe_flat)
# print '95th percentile'
# print 'spearmanr', spearmanr(foodie_personalized_surp_estimates_recipeflat_95perc, foodie_surp_ratings_recipe_flat)
# print 'pearsonr', pearsonr(foodie_personalized_surp_estimates_recipeflat_95perc, foodie_surp_ratings_recipe_flat)
# print 'Max score'
# print 'spearmanr', spearmanr(foodie_personalized_surp_estimates_recipeflat_max, foodie_surp_ratings_recipe_flat)
# print 'pearsonr', pearsonr(foodie_personalized_surp_estimates_recipeflat_max, foodie_surp_ratings_recipe_flat)
#
#
# # In[33]:
#
#
# # Novices
# print '90th percentile'
# print 'spearmanr', spearmanr(novice_personalized_surp_estimates_recipeflat_90perc, novice_surp_ratings_recipe_flat)
# print 'pearsonr', pearsonr(novice_personalized_surp_estimates_recipeflat_90perc, novice_surp_ratings_recipe_flat)
# print '95th percentile'
# print 'spearmanr', spearmanr(novice_personalized_surp_estimates_recipeflat_95perc, novice_surp_ratings_recipe_flat)
# print 'pearsonr', pearsonr(novice_personalized_surp_estimates_recipeflat_95perc, novice_surp_ratings_recipe_flat)
# print 'Max score'
# print 'spearmanr', spearmanr(novice_personalized_surp_estimates_recipeflat_max, novice_surp_ratings_recipe_flat)
# print 'pearsonr', pearsonr(novice_personalized_surp_estimates_recipeflat_max, novice_surp_ratings_recipe_flat)
#
#
# # Replicate the oracle by the number of the users to be able to correlate with the users' ratings (the arrays have to be the same size)
#
# # In[34]:
#
#
# oracle_surp_estimates_repeated_90perc = [oracle_surp_estimates_90perc] * len(users_surp_ratings_userlvl)
# oracle_surp_estimates_repeated_95perc = [oracle_surp_estimates_95perc] * len(users_surp_ratings_userlvl)
# oracle_surp_estimates_repeated_max = [oracle_surp_estimates_max] * len(users_surp_ratings_userlvl)
# oracle_surp_estimates_userflat_90perc = [item for sublist in oracle_surp_estimates_repeated_90perc for item in sublist]
# oracle_surp_estimates_userflat_95perc = [item for sublist in oracle_surp_estimates_repeated_95perc for item in sublist]
# oracle_surp_estimates_userflat_max = [item for sublist in oracle_surp_estimates_repeated_max for item in sublist]
#
#
# # Correlations oracle flattened
#
# # In[35]:
#
#
# print '90th percentile'
# print 'spearmanr', spearmanr(oracle_surp_estimates_userflat_90perc, users_surp_ratings_user_flat)
# print 'pearsonr', pearsonr(oracle_surp_estimates_userflat_90perc, users_surp_ratings_user_flat)
# print '95th percentile'
# print 'spearmanr', spearmanr(oracle_surp_estimates_userflat_95perc, users_surp_ratings_user_flat)
# print 'pearsonr', pearsonr(oracle_surp_estimates_userflat_95perc, users_surp_ratings_user_flat)
# print 'Max score'
# print 'spearmanr', spearmanr(oracle_surp_estimates_userflat_max, users_surp_ratings_user_flat)
# print 'pearsonr', pearsonr(oracle_surp_estimates_userflat_max, users_surp_ratings_user_flat)
#
