# GloVex
Expectation modelling with GloVe

# Usage

    python preprocessor.py 'filepath' [--dataset 'dataset name'] [override_preprocessing] [—override_model]

    'filepath': the path to the data file
    
    filepaths to ACM datasets:
        Complete: '../../Datasets/ACM Digital Library/Abstract only/ACMDL_complete.csv'
        First 1000: '../../Datasets/ACM Digital Library/Abstract only/ACMDL_1000.csv'
        First 10000: '../../Datasets/ACM Digital Library/Abstract only/ACMDL_10000.csv'
    filepaths to Recipes datasets:
        Complete: '/Users/oeltayeb/Google Drive/Live/PQE/Omar/RecipeDataSummer2017/recipe_cuisine_all_tagged'
        First 1000: '/Users/oeltayeb/Google Drive/Live/PQE/Omar/RecipeDataSummer2017/recipe_cuisine_all_tagged_1000'

    --dataset: string, default acm
        ACM: is the default dataset, which is expressed using the 'acm' string
        WikiPlot: 'plots'
        Recipes: 'recipes'

    -override_preprocessing: use it if a new preprocessing is needed
    
    —override_model: use it if a new model is needed

	--familiarity_categories: default None
		"fam_cat_file_path": the familiarity category file path; used with the input is ACMDL or WikiPlot datasets
		"True": to add the categories from the same dataset file; used with the recipe dataset only
