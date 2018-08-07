import sys, csv, random
import numpy as np
from scipy.spatial import distance

def print_most_similar(vectors, id2w, ids, i, occs, num_to_print=5, ignore=[]):
	#distances = distance.cdist(np.atleast_2d(vectors[i]),np.array([vectors[id] for id in ids if id is not i]), "cosine").T
	distances = [distance.cosine(vectors[i],vectors[id]) for id in ids]
	sorted_distance_tuples = [(float(d),id) for d,id in zip(distances,ids) if id is not i and id not in ignore]
	sorted_distance_tuples.sort(key = lambda x: x[0])
	print "word:",id2w[i],"("+str(occs[i])+")"
	for d,id in sorted_distance_tuples[:num_to_print]:
		print "  ",id2w[id],"("+str(occs[id])+") : ",d

def calculate_differences(vectors, ids, dims):
	diffs = np.zeros((len(ids),len(ids), dims))
	for id in ids:
		for id2 in ids:
				diffs[id,id2,:] = vectors[id] - vectors[id2]
	return diffs

def interpolate_between_surprises(sv1, sv2, alpha):
	if alpha > 1.0 or alpha < 0:
		print "Bad interpolation alpha!  Sadness!"
		sys.exit()
	return alpha * sv1 + (1-alpha) * sv2

def print_most_similar_differences(diff_vectors, wi, wi2, id2w, ids, obs_surps, occs, num_to_print=5, ignore=[], metric="euclidean"):
	dists = np.zeros((len(ids),len(ids)))
	for id in ids:
		for id2 in ids:
			if not id == id2 and not id in [wi,wi2] and not id2 in [wi,wi2]:
				if metric == "euclidean":
					dists[id,id2] = distance.euclidean(diff_vectors[wi,wi2,:],diff_vectors[id,id2,:])
				elif metric == "cosine":
					dists[id,id2] = distance.cosine(diff_vectors[wi,wi2,:],diff_vectors[id,id2,:])
	it = np.nditer(dists, flags=["multi_index"])
	sorted_distance_tuples = []
	while not it.finished:
		if not it.multi_index[0] == it.multi_index[1] and not it.multi_index[0] in [wi,wi2]+ignore and not it.multi_index[1] in [wi,wi2]+ignore:
			sorted_distance_tuples.append((it[0],it.multi_index[0],it.multi_index[1]))
		it.iternext()
	sorted_distance_tuples.sort(key = lambda x: x[0])
	print "surprise pair:",id2w[wi],"("+str(occs[wi])+")",id2w[wi2],"("+str(occs[wi2])+"), obs_surp:",str(round(obs_surps[wi,wi2],2))
	for d,id,id2 in sorted_distance_tuples[:num_to_print]:
		print "  ",id2w[id],"("+str(occs[id])+")",id2w[id2],"("+str(occs[id2])+")"," dist:",str(round(d,5)), " obs_surp:",str(round(obs_surps[id,id2],2))


def read_vector_list(path, verbose = False):
	ings = {}
	ings_context = {}
	ids = []
	id2w = {}
	with open(path) as f:
		reader = csv.reader(f)
		for row in reader:
			id = int(row[0])
			ids.append(id)
			dims = (len(row) - 2) / 2
			id2w[id] = row[1]
			ings[id] = np.asarray(row[2:dims + 2]).astype(np.float)
			ings_context[id] = np.asarray(row[dims + 2:]).astype(np.float)
			if verbose:
				print id, id2w[id], len(ings[id]), len(ings_context[id])
	return ings,ings_context,id2w,ids, dims

def read_surps(path, verbose = False):
	return np.loadtxt(path, np.float32, delimiter=",")

def read_dictionary(path,id2w, verbose=False):
	occs = {}
	w2id = {v:k for k,v in id2w.iteritems()}
	with open(path) as f:
		reader = csv.reader(f)
		for row in reader:
			occs[row[0]] = int(row[1])
	occs_list = [0] * len(w2id.keys())
	for w,c in occs.iteritems():
		occs_list[w2id[w]] = c
	return occs_list, w2id

if __name__ == "__main__":
	ings, ings_context,id2w, ids, dims = read_vector_list(sys.argv[1]+"_vectors.csv")
	ings_combined = {wi:np.concatenate((w,ings_context[wi])) for wi,w in ings.iteritems()}
	occs, w2id = read_dictionary(sys.argv[1]+"_dictionary.csv", id2w)

	ings_to_ignore = ["topping", "filling", "sodium", "calories", "salad_dressing", "vghca", "van_geffen", "pieces", "green", "sauce", "tomato", "tobasco", 'vorheis', "leaves", "gluten", "mintzias", "cubes", "greens"]
	ignored_indices = [i for i in ids if id2w[i] in ings_to_ignore]

	for i in range(5):
		w = random.choice(ings.keys())
		while w in ignored_indices:
			w = random.choice(ings.keys())
		print_most_similar(ings, id2w, ids, w, occs, ignore=ignored_indices, num_to_print=15)

	ings_to_print = ["pine_nuts", "cucumbers", "cottage_cheese", "cayenne_pepper", "apples", "lentils", "chocolate"]

	print "Ings to print:"
	for i in ings_to_print:
		print_most_similar(ings, id2w, ids, w2id[i], occs, ignore=ignored_indices, num_to_print=15)

	diffs = calculate_differences(ings,ids, dims)
	#est_surps = read_surps(sys.argv[1]+"_estimated_surprise.csv")
	obs_surps = read_surps(sys.argv[1]+"_observed_surprise.csv")

	'''
	print "Ten randoms."
	for i in range(10):
		w = random.choice(ings.keys())
		while w in ignored_indices:
			w = random.choice(ings.keys())
		w2 = random.choice(ings.keys())
		while w == w2 or w2 in ignored_indices:
			w2 = random.choice(ings.keys())
		print_most_similar_differences(diffs, w, w2, id2w, ids, obs_surps, occs, num_to_print=10, ignore=ignored_indices, metric="cosine")
	'''

	obs_surps_tuples = []
	it = np.nditer(obs_surps, flags=["multi_index"])
	while not it.finished:
		if it.multi_index[0] < it.multi_index[1]:
			obs_surps_tuples.append((it[0],it.multi_index[0],it.multi_index[1]))
		it.iternext()
	obs_surps_tuples.sort(key = lambda x: x[0], reverse=True)

	pairs_to_match = [("parmesan", "coriander"), ("banana","basil"),("soy_sauce", "vanilla"), ("worcestershire_sauce", "vanilla"),
					  ("parmesan", "vanilla"), ("soy_sauce", "chocolate"),("soy_sauce", "vanilla"),("mozzarella", "brown_sugar")]
	print
	print "Listed matches:"
	for i, j in pairs_to_match:
		print_most_similar_differences(diffs, w2id[i], w2id[j], id2w, ids, obs_surps, occs, num_to_print=50, ignore=ignored_indices, metric="euclidean")


	print
	print "Top 100:"
	for d,i,j in obs_surps_tuples[:100]:
		print_most_similar_differences(diffs, i, j, id2w, ids, obs_surps, occs, num_to_print=50, ignore=ignored_indices, metric="cosine")


	'''
		for i in range(10):
			s1w1 = random.choice(ings.keys())
			s1w2 = random.choice(ings.keys())
			while s1w2 == s1w1:
				s1w2 = random.choice(ings.keys())
			s2w1 = random.choice(ings.keys())
			while s2w1 in [s1w1,s1w2]:
				s2w1 = random.choice(ings.keys())
			s2w2 = random.choice(ings.keys())
			while s2w2 in [s1w1,s1w2,s2w1]:
				s2w2 = random.choice(ings.keys())
	'''
