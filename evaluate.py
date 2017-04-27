import argparse, logging

import preprocessor

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("glovex")

def eval_dataset_surprise(model, acm, top_n_per_doc = 0):
	dataset_surps = []
	for id,title,doc in zip(acm.doc_ids, acm.doc_titles, acm.documents):
		if len(doc):
			surps = preprocessor.estimate_document_surprise_pairs(doc, model, acm, top_n_per_doc = top_n_per_doc)
			dataset_surps.append((id,title,surps,document_surprise(surps)))
		else:
			dataset_surps.append((id,title,[],3))
	return dataset_surps

def document_surprise(surps):
	if len(surps):
		return min([s[2] for s in surps])
	return float("inf")

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
	dataset_surps = eval_dataset_surprise(model, acm, top_n_per_doc=5)
	dataset_surps.sort(key = lambda x: x[1])
	for doc in dataset_surps[:10]:
		print doc[0]+":", doc[1]
		print doc[2]
		print


