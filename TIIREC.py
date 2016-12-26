#-*- utf-8 -*-

import numpy as np
import sys
import random, math
import time, argparse
import collections, logging
import itertools, multiprocessing


DIMENSION = 0.
LEARNRATIO = 0.
REGULARIZATION = 0.01

# threshold to define sparse items
PARTION_POINT = 0

# max iterations
MaxEpoch = 0

# number of processes to calculate evaluation results
NUMWORKERs = 24

# calculate recall with k in reck_list
reck_list = [5, 10, 15, 20, 25, 30]

# define feature vectors
V, T, S, U, A = [0 for i in range(0, 5)]

artists, sparseItems, itemsQuery = [0 for i in range(0, 3)]

# define implicit feedback variables
item_im, tag_im = [dict(), dict()]
user_tag_item_im, item_tag_im = [dict(), dict()]

# testing cases
total_test_counts, sparse_test_counts = 0, 0

# usr-item dict
USRARTIST = dict()

#global CONSTRAINT,DIMENSION,LEARNRATIO,RECALLK,V,T,S,U,artists,vset,tset
def TIIREC_PlusPlus(DIMEN=5, LEARNR=0.01,REGUL=0.01, MAXEPOCH=30, pppoint=5):
	""" Triple Interaction Item Recommendation presents a novel appraoch to blends collaborative
	    recommendation and information retrieval.
	"""
	global DIMENSION, LEARNRATIO, REGULARIZATION, PARTION_POINT, MaxEpoch

	DIMENSION = DIMEN
	LEARNRATIO = LEARNR
	REGULARIZATION = REGUL
	PARTION_POINT = pppoint
	MaxEpoch = MAXEPOCH

def iniParMatrix(fp=0):
	""" Read data from certain file path
	    Initilize parameters for entities
	""" 
	global V,T,S,U,A, artists, USRARTIST, sparseItems, itemsQuery

	V, T, S, U, A, artists, USRARTIST, sparseItems, itemsQuery = load(fp = fp)

def exe(traset=0, teset_path=0, dataset=0):
	""" Learn the values of parameters from the data by SGD optimization
		
		@param
		 traset - data set file path
		 vaset - validation data set file path
		 teset - testing data set file path
	"""
	global total_test_counts, sparse_test_counts

	logging.info("Begin to initialize features")
	iniParMatrix(fp = traset)
	
	logging.info("Loading Testing Cases")
	tset, total_test_counts, sparse_test_counts = validSet(test = teset_path)

	logging.info("{0} testing cases, {1} sparse item testing cases".format(total_test_counts, sparse_test_counts))	
	train(traset = traset, vset = tset)

def train(traset=0, vset=0):
	""" Learn the values of parameters from the data by SGD optimization
	Parameter
	#traset - training data set file path
	#vset - validation data set
	"""
	global CONSTRAINT,DIMENSION,LEARNRATIO, V,T,S,U,A,artists,USRARTIST, MaxEpoch

#	print "Begin to optimize loss function"
	epoch = 0
	while MaxEpoch > 0:
		epoch += 1
		MaxEpoch -= 1

		with open(traset, "r") as src:
			for line in src.readlines():
				uid, query, aid = line.strip().split("\t")
				QueryL2Rank(uid, query, aid)
		
		src.close()

		globalRecalls, sparseRecall = MapReduceEvaluation(tset = vset, num_workers = NUMWORKERs)
		globalRecalls = toStringRecalls(globalRecalls)
		sparseRecalls = toStringRecalls(sparseRecall)

		logging.info("GLOBAL:" + globalRecalls)
		logging.info("SPARSE:" + sparseRecalls)

def toStringRecalls(recallsDict):
	""" print recall values
	"""

	info = []
	recalls = recallsDict.items()
	recalls.sort(key = lambda x: x[0])

	for k, rec in recalls:
		info.append("%.4f"%rec)
	info = ":".join(info)

	return info

def QueryL2Rank(uid, query, aid):
	global LEARNRATIO, REGULARIZATION,V,T,S,U,A, item_im, tag_im, user_tag_item_im, item_tag_im
	
	vu = V[uid]
	ta = T[aid]	
	sq = S[query]
	uu = U
	aa = A[aid]

	Listened = USRARTIST[uid]

	modelA = len(artists) - 1

	# bayesian personalized sampling
	while True:
		index = random.randint(0, modelA)
		bid = artists[index]
		if bid not in Listened:
			break
	
	ab = A[bid]
	tb = T[bid]

	# features with implicit feedback
	utiim = user_tag_item_im[uid]

	# user side
	utdenominator, utim = utiim["tim"]
	uidenominator, uiim = utiim["iim"]

	# \hat{vu}
	hatvu = vu + utim + uiim

	# item side
	aitdenominator, aitim = item_tag_im[aid]
	bitdenominator, bitim = item_tag_im[bid]

	# \hat{Ti}
	hatta = ta + aitim
	hattb = tb + bitim

	# difference between \hat{Ti} and \hat{Tj}
	dhattatb = hatta - hattb

	# difference between Aa and Ab
	daabb = aa - ab

	# <St, daabb>
	stdaabb = np.dot(sq, daabb)

	squ = np.dot(sq, uu)

	# inner-product of sq and aa encoder
	sqaa = np.dot(sq,aa)

	# inner-product of sq and ab encoder
	sqab = np.dot(sq, ab)

	error = np.dot(squ, dhattatb) + np.dot(hatvu, dhattatb) + np.dot(sqaa, hatvu) - np.dot(sqab, hatvu)
	try:
		ep = math.exp(-error)
		loss = ep / (1 + ep)
	except OverflowError:
		loss = 1.

	#updating corresponding parameters

	# Updating user u
	V[uid] = vu + LEARNRATIO * (loss * (dhattatb + stdaabb) - REGULARIZATION * vu)

	# Updating query q
	S[query] = sq + LEARNRATIO * (loss * (np.dot(uu, dhattatb) + np.dot(daabb, hatvu)) - REGULARIZATION * sq)

	# Updating artist a
	squhatvu = squ + hatvu
	T[aid] = ta + LEARNRATIO * (loss * squhatvu - REGULARIZATION * ta)
	
	# Updating artist b
	T[bid] = tb - LEARNRATIO * ( loss * squhatvu + REGULARIZATION * tb)

	# user encoder
	di = np.matrix(dhattatb)
	sq = np.matrix(sq)
	m = np.array(sq.T * di)
	U = uu + LEARNRATIO * (loss * np.array(m) - REGULARIZATION * uu )

	# item encoder
	sq = np.matrix(sq)
	m = np.array(sq.T * hatvu)

	A[bid] = ab - LEARNRATIO * (loss * np.array(m) + REGULARIZATION * ab)
	A[aid] = aa + LEARNRATIO * (loss * np.array(m) - REGULARIZATION * aa)

	# Implicit feedback
	hattijstaij = (dhattatb + stdaabb)

	#updating yi
	iaim = item_im[aid]
	item_im[aid] = iaim + LEARNRATIO * (loss * uidenominator * hattijstaij - REGULARIZATION * iaim)

	# updating yt
	uttim = tag_im[query]
	tag_im[query] = uttim + LEARNRATIO * (loss * utdenominator * hattijstaij - REGULARIZATION * uttim)

	# Updating \user_tag_item_im
	ut = math.pow(utdenominator, 0.5) * utim
	ut -= uttim
	ut += tag_im[query]

	ut = utdenominator * ut

	ui = math.pow(uidenominator, 0.5) * uiim
	ui -= iaim
	ui += item_im[aid]
	ui = uidenominator * ui

	utiim["iim"] = [uidenominator, ui]
	utiim["tim"] = [utdenominator, ut]

	# Updating \item_tag_im
	it = math.pow(aitdenominator, 0.5) * aitim
	it -= uttim
	it += tag_im[query]
	it = aitdenominator * it

	item_tag_im[aid] = [aitdenominator, it]

def MapReduceEvaluation(tset=0, num_workers=2):
	""" There are many possible evaluation metrics. In this case, we use recall@K
	    and weighted recall@K to measure the performance of a algorithm.

	Definition V=0, U=0, T=0, S=0
	"""
	global CONSTRAINT,DIMENSION,LEARNRATIO,V,T,S,U,artists, total_test_counts, sparse_test_counts

	# logging.info("MapReduce Evaluation")

	map_inputs = tset
	# Defninition processing pool to deal with mapped inputs
	mappool = multiprocessing.Pool(num_workers)
	reduce_pool = multiprocessing.Pool(num_workers)
	try:
		map_responses = mappool.map(map_fnc, map_inputs,
								 chunksize = 1)
		mappool.close()
		mappool.join()

		partitioned_data = partition(itertools.chain(*map_responses))

		reduced_values = reduce_pool.map(reduce_fnc, partitioned_data.items())
		reduce_pool.close()
		reduce_pool.join()

		reduced_values = dict(reduced_values)

		globalRecalls = reduced_values["globalRecall"]
		# normalize global recall value
		for k in globalRecalls:
			globalRecalls[k] /= float(total_test_counts)
		
		# calculate recall value of sparse items
		sparseCases = sparse_test_counts
		recallSparseItems = reduced_values["recallSparseItems"]
		for k in recallSparseItems:
			recallSparseItems[k] /= float(sparse_test_counts)

		return globalRecalls, recallSparseItems

	except KeyboardInterrupt:
		sys.exit(0)

def map_fnc(user_query):
	""" Input user-query pairs and return a sequence of (key, occurrences) values.

		@return
		 Recall#K [5, 10, 15, 20, 25, 30]
	"""
	global reck_list,sparseItems,V,T,S,U,A,item_im, tag_im, user_tag_item_im, item_tag_im
	
	uid, query, samples = user_query

	vu = V[uid]
	uu = U

	# features with implicit feedback
	utiim = user_tag_item_im[uid]

	# user side
	utdenominator, utim = utiim["tim"]
	uidenominator, uiim = utiim["iim"]

	# \hat{vu}
	hatvu = vu + utim + uiim

	# recall at k
	reck = dict()
	for k in reck_list:
		reck.setdefault(k, 0)
	
	# recall sparse items
	recallSparseItems = dict()
	for k in reck_list:
		recallSparseItems.setdefault(k, 0)

	sq = S[query]
	qusr = np.dot(sq , uu) + hatvu

	rankList = []
	for aid in T:
		ta = T[aid]
		aitdenominator, aitim = item_tag_im[aid]
		hatta = ta + aitim
		aa = A[aid]

		quaua = np.dot(qusr, hatta)
		qau = np.dot(np.dot(sq , aa), hatvu)
		score =  quaua + qau

		rankList.append([aid, score])

	rankList.sort(key = lambda x:x[1], reverse = True)

	for k in reck_list:
		top_K = dict(rankList[:k])
		for aid in samples:
			if aid in top_K:
				reck[k] += 1
				if aid in sparseItems:
					recallSparseItems[k] += 1

	return [("globalRecall", reck.items()), ("recallSparseItems", recallSparseItems.items())]


def reduce_fnc(items):
	""" Convert the partitioned dat for a word to a tuple containing the key and the
	    number of occurrences.
	"""
	key, occurrences = items
	elements_sum = dict()
	#init
	for record in occurrences:
		for index, value in record:
			elements_sum.setdefault(index, 0)
			elements_sum[index] += value

	return (key, elements_sum)		

def partition(mapped_values):
	""" Organize the mapped values by their key.
	"""

	partitioned_data = collections.defaultdict(list)
	try:
		for key, value in mapped_values:
			partitioned_data[key].append(value)
	except:
		print records, mapped_values[0]
		sys.exit(0)

	return partitioned_data


def validSet(valid=0, test=0):
	""" Validation data set

	Return
	#tset - dictionary
	"""
	global V, S, T, sparseItems

	tset = dict()

	t_counts = 0
	sparse_test_cases = 0

	with open(test, "r") as src:
		
		for line in src.readlines():
			uid, query, aid = line.strip().split("\t")
			
			if (uid in V) and (query in S) and (aid in T):
				tset.setdefault(uid, {})
				tset[uid].setdefault(query, [])
				tset[uid][query].append(aid)

				t_counts += 1

	inputs = []

	# Mapping inputs to intermediate data.
	# Return a tuple with the key and a value to be reduced.
	for uid in tset:
		samples = tset[uid]
		for query in samples:
			s = []
			for aid in samples[query]:
				s.append( aid )
				
				if aid in sparseItems:
					sparse_test_cases += 1

			inputs.append((uid, query, s) )

	return inputs, t_counts, sparse_test_cases

def load(fp=0):
	""" Load samples to construct parameters matrices.
	    Each sample contains UserID,Query,Artist,WCount,QueryType

		@param
	     fp - file path

		Return:
		#V - user feature matrix
		#S - query feature matrix
		#T - atrist feature matrix
		#U - a dictionary includes all users' specific encoding matrices
		#artists - includes all artists appearing in the training data
	"""
	global DIMENSION, item_im, tag_im, user_tag_item_im, item_tag_im

	V, S, T, U, A = [dict() for i in range(0, 5)]

	usrs = dict()

	# temple dict to store user-tag relation
	t_user_tag = dict()

	# item-query matrix
	itemsQuery = dict()

	with open(fp, "r") as src:
		
		for line in src.readlines():
			uid, query, aid = line.strip().split("\t")
			usrs.setdefault(uid, {})
			usrs[uid].setdefault(aid, 1)

			t_user_tag.setdefault(uid, {})
			t_user_tag[uid].setdefault(query, 1)

			if uid not in V:
				
				V[uid] = np.random.uniform(-0.02, 0.02, DIMENSION)
				
			if query not in S:
				
				S[query] = np.random.uniform(-0.02, 0.02, DIMENSION)
				

			if aid not in T:
				
				T[aid] = np.random.uniform(-0.02, 0.02, DIMENSION)					
				
			if aid not in A:
				A[aid] = np.random.uniform(-0.02, 0.02, (DIMENSION,DIMENSION) )

			
			U = np.random.uniform(-0.02, 0.02, (DIMENSION,DIMENSION) )

			item_im.setdefault(aid, np.random.uniform(-0.02, 0.02, DIMENSION))
			tag_im.setdefault(query, np.random.uniform(-0.02, 0.02, DIMENSION))

			itemsQuery.setdefault(aid, {})

			itemsQuery[aid].setdefault(query, 1)

		artists = itemsQuery.keys()
	
	# Initlialize user's implicit feedback features
	# user_tag_item_im
	for uid in usrs:
		uitems = usrs[uid]
		denominator = math.pow(len(uitems), -0.5)

		user_tag_item_im.setdefault(uid, {})

		t_iim = 0

		for iid in uitems:
			t_iim += item_im[iid]

		t_iim = denominator * t_iim

		user_tag_item_im[uid]["iim"] = [denominator, t_iim]

		# user tag implicit feedback
		utags = t_user_tag[uid]
		denominator = math.pow(len(utags), -0.5)

		t_tim = 0
		for tid in utags:
			t_tim += tag_im[tid]
		t_tim = denominator * t_tim

		user_tag_item_im[uid]["tim"] = [denominator, t_tim]

	# Find sparse items with less than PARTION_POINT keywords
	sparseItems = dict()
	for iid in itemsQuery:
		if len(itemsQuery[iid]) <= PARTION_POINT:
			sparseItems.setdefault(iid, 1)

		# initialize item tag implicit features
		item_tag_im.setdefault(iid, [])

		itags = itemsQuery[iid]
		tit_im = 0
		denominator = math.pow(len(itags), -0.5)
		for tid in itags:
			tit_im += tag_im[tid]
		tit_im = denominator * tit_im

		item_tag_im[iid] = [denominator, tit_im]

	return V, T, S, U, A, artists, usrs, sparseItems, itemsQuery

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run TIIREC")

	parser.add_argument('--lr', type = float, default = 0.001, help="learning rate")
	parser.add_argument('--reg', type = float, default = 0.01, help="regularization parameter")

	parser.add_argument('--trainpath', default = "./data/train.data", help="your training data path")
	parser.add_argument('--testpath', default = "./data/test.data", help = "your test dataset path")
	parser.add_argument('--dataname', default = "unknown", help = "your data set name")
	parser.add_argument('--dimension', type=int, default=10, help = "number of dimension")
	parser.add_argument('--sparse_items_threshold', type=int, default=5, help="threshold to define sparse items")
	parser.add_argument('--maxepoch', type=int, default=100, help = "maximum iterations")
	parser.add_argument('--workers', type=int, default=6, help = "number of parallel processes")
	
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()

	NUMWORKERs = args.workers
	# Config logging output format
	# logging.basicConfig(format = '%(asctime)s:%(levelname)s:%(message)s',
	# 					level = logging.INFO)

	logging.basicConfig(format = '%(asctime)s:%(levelname)s:%(message)s',
						level = logging.INFO,
						filename =  "./logs/{0}_TIIREC_{1}_RECALLS.log".format(args.dataname.upper(), args.dimension),
						filemode = "w")

	TIIREC_PlusPlus(DIMEN=args.dimension,
				LEARNR=args.lr,
				REGUL = args.reg,
				MAXEPOCH = args.maxepoch,
				pppoint = args.sparse_items_threshold)

	exe(traset = args.trainpath,
		teset_path = args.testpath,
		dataset = args.dataname)

