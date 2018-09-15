### Import libraries
import time
import sys
from pyspark import SparkContext
import itertools
import random


def generate_utility_matrix(iterator):
	global unique_movies
	iterator = list(iterator)
	user_id = iterator[0]
	user_rated_movies = list(iterator[1])
	return_array = []
	for movie_id in unique_movies:
		if movie_id in user_rated_movies:
			return_array.append(1)
		else:
			return_array.append(0)
	assert len(return_array) == len(unique_movies)
	yield (user_id, return_array)

def generate_hash_functions():
	global a_matrix
	global b_matrix
	global n_hash_functions
	seed = random.randint(0, len(unique_movies))
	for i in xrange(n_hash_functions):
		randIndex = random.randint(0, len(unique_movies))
		while randIndex in a_matrix:
			randIndex = random.randint(0, len(unique_movies))
		a_matrix.append(randIndex)
		randIndex = random.randint(0, len(unique_movies))
		while randIndex in a_matrix:
			randIndex = random.randint(0, len(unique_movies))
		b_matrix.append(randIndex)


def zip_columns(splitIndex, iterator):
	global unique_movies
	chunk = list(iterator)
	column_dict = {}
	candidate_pairs = []
	
	#   zipped_columns = zip(chunk[0][1], chunk[1][1], chunk[2][1], chunk[3][1])
	zipped_columns = zip(*[chunk[i][1] for i in range(len(chunk))])
	#   yield (splitIndex, zipped_columns[0])

	for key, column in zip(unique_movies, zipped_columns): 
		column_dict[key] = column

	for combination in itertools.combinations(unique_movies, 2):
		if column_dict[combination[0]] == column_dict[combination[1]]:
			candidate_pairs.append(combination)
			
	yield candidate_pairs

def compute_minHash_signatures(row):
	'''
	row : each row of the previously generated utility matrix
	n : number of randomly chosen hash functions
	'''
	global unique_userids
	global a_matrix
	global b_matrix
	
	userId = row[0]
	utility_matrix_row = row[1]
	
	hash_values = []
	
	# compute the various hash values for the current movieId
	# hash function of the form (ax + b)%m => a = random integer, b=random integer, m=number of movies

	for i in range(len(a_matrix)):
		hash_values.append((a_matrix[i] * userId + b_matrix[i]) % len(unique_userids))
			
	#     assert len(set(hash_values)) == len(hash_values)
	#     assert len(hash_values) == 40
	
	for signature_column, rating_status in enumerate(utility_matrix_row):
		# if userId == 1:
			# print(signature_column)
		if rating_status == 1:
			for signature_row, row_in_hash in enumerate(hash_values):
				yield ((signature_row, signature_column), row_in_hash)

def zip_columns_lsh(splitIndex, iterator):
	global unique_movies
	chunk = list(iterator)
	column_dict = {}
	global candidate_pairs
	return_values = []
	
	zipped_columns = zip(*[chunk[i][1] for i in range(len(chunk))])
	
	for key, column in zip(unique_movies, zipped_columns):
		column_dict[key] = column
	
	for pair in candidate_pairs:
		yield (pair, count_intersection_and_union(column_dict[pair[0]], column_dict[pair[1]]))

def toList(listoftuples):
	return_list = [0 for i in xrange(len(listoftuples))]
	for val in listoftuples:
		return_list[val[0]] = val[1]
	return return_list


def count_intersection_and_union(column1, column2):
	union_count = 0
	intersection_count = 0
	# assert set(column1).issubset(set([0, 1])) and set(column2).issubset(set([0, 1]))
	for value1, value2 in zip(column1, column2):
		if value1 + value2 == 2:
			union_count += 1
			intersection_count += 1
		elif value1 + value2 == 1:
			union_count += 1
	return (intersection_count, union_count)


if __name__=="__main__":

	sc = SparkContext(appName="Assignment 3 LSH Task 1")

	### Get the start_time
	start_time = time.time()

	### Get the path of ratings.csv file from the input arguments
	data_path = sys.argv[1]

	### data_path = sc.textFile("../Description/data/ratings.csv")
	data = sc.textFile(data_path, minPartitions=2)
	header = data.first()
	data = data.filter(lambda row : row != header)

	### Unique userids
	unique_userids = data.map(lambda row : int(row.split(",")[0])).distinct().collect()
	unique_userids.sort()

	### Filter only the user, movie columns from the data
	utility_matrix = data.map(lambda row : [int(i) for i in row.split(",")[:2]])

	### Unique movieIds
	unique_movies = utility_matrix.map(lambda row : row[1]).distinct().collect()
	unique_movies.sort()

	### Generate utility matrix
	utility_matrix = utility_matrix.groupByKey()\
	.map(generate_utility_matrix)\
	.flatMap(lambda x:x)\
	.sortByKey()

	### Generate hash functions
	n_hash_functions = 40
	a_matrix = []
	b_matrix = []
	generate_hash_functions()

	### Generate signature matrix
	signature_matrix = utility_matrix.flatMap(lambda row: compute_minHash_signatures(row))\
	.reduceByKey(lambda x,y:x if x<y else y)\
	.map(lambda x: (x[0][0], [x[0][1], x[1]])).groupByKey()\
	.map(lambda x : (x[0], toList(x[1]))).collect()

	### Create bands
	bands = sc.parallelize(signature_matrix, 20)
    
	### Generate candidate_pairs
	candidate_pairs = bands.mapPartitionsWithIndex(zip_columns)\
	.flatMap(lambda x : x).distinct().collect()

	### Get similar items
	similar_items = utility_matrix\
	.mapPartitionsWithIndex(zip_columns_lsh)\
	.reduceByKey(lambda x, y:(x[0] + y[0], x[1] + y[1]))\
	.mapValues(lambda values : float(values[0]) / values[1])\
	.filter(lambda x : x[1] >= 0.5).sortByKey().collect()

	### Import the ground truth file
	# ground_truth = sc.textFile("../../Description/data/SimilarMovies.GroundTruth.05.csv")
	# ground_truth = ground_truth\
	# .map(lambda x : tuple(sorted([int(i) for i in x.split(",")])))\
	# .collect()

	# ### Create similar_items_list
	# similar_items_list = [s[0] for s in similar_items]

	# ### Compute true positives, false positives and false negatives 
	# true_positives = len(set(similar_items_list))
	# false_negatives = len(set(ground_truth)) - len(set(similar_items_list).intersection(set(ground_truth)))
	# false_positives =len(similar_items_list) - len(set(similar_items_list).intersection(set(ground_truth)))

	# ### Compute precision and recall
	# precision = float(true_positives) / (true_positives + false_positives)
	# recall = float(true_positives) / (true_positives + false_negatives)

	# ### Compute end time
	end_time = time.time()

	# ### Print the results on the terminal
	# print("Precision = {}".format(precision))
	# print("Recall = {}".format(recall))
	print("Time taken = {}".format(end_time - start_time))

	### Save the contents of similar_items list of tuples file to a txt file
	with open("Vishal_Seshagiri_SimilarMovies_Jaccard.txt", "w") as file:
		for tup in similar_items:
			file.write("{}, {}, {}\n".format(tup[0][0], tup[0][1], tup[1]))


# Result Stats
# Precision = 1.0
# Recall = 0.999786587826
# Time taken = 199.905250072
