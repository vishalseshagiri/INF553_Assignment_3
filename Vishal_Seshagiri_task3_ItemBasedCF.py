from pyspark import SparkContext
from collections import OrderedDict
import sys
import itertools
import time
import random
# from Vishal_Seshagiri_task2_UserBasedCF import pearsons_correlation, calc_absolute_differences_and_mse
# from Vishal_Seshagiri_task1_Jaccard import generate_utility_matrix, generate_hash_functions, zip_columns, compute_minHash_signatures, zip_columns_lsh, toList, count_intersection_and_union 

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


def pearsons_correlation(arg1, arg2):
	arg1_avg = sum(arg1)/len(arg1)
	arg2_avg = sum(arg2)/len(arg2)
	arg1 = [i-arg1_avg for i in arg1]
	arg2 = [i-arg2_avg for i in arg2]
	numerator = 0
	denomL = 0
	denomR = 0
	
	for (l, r) in zip(arg1, arg2):
		numerator += l*r
		denomL += l**2
		denomR += r**2
	denomL = denomL ** 0.5
	denomR = denomR ** 0.5
	denom = denomL * denomR
	if denom == 0:
		return 0
	
	return numerator / denom

def calc_absolute_differences_and_mse(predictions, true_values):
	global mse
	global absolute_difference
	
	for a,b in zip(predictions, true_values):
		diff = abs(a[2] - b[2])
		
		if diff < 1:
			absolute_difference[">=0 and <1"] += 1
		elif diff < 2:
			absolute_difference[">=1 and <2"] += 1
		elif diff < 3:
			absolute_difference[">=2 and <3"] += 1
		elif diff < 4:
			absolute_difference[">=3 and <4"] += 1
		else:
			absolute_difference[">=4"] += 1
				
		mse += (a[2] - b[2]) ** 2
	mse /= len(predictions)


if __name__ == "__main__":
	sc = SparkContext(appName="Assignment 3 LSH Task 1")

	### Get the start_time
	start_time = time.time()

	### Get the path of ratings.csv and testing files from the input arguments
	data_path = sys.argv[1]
	testing_path = sys.argv[2]

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
	.filter(lambda x : x[1] >= 0.5).sortByKey()

	### Creating the movies_rdd file
	movies_rdd = data.map(lambda x: [i for i in x.split(",")[:3]])\
	.map(lambda x:(int(x[1]), (int(x[0]), float(x[2]))))\
	.groupByKey()\
	.mapValues(lambda values : [v for v in values])

	m_rdd = dict(movies_rdd.collect())

	predicted_LSH = similar_items\
	.map(lambda x : [x[0][0], x[0][1]])\
	.collect()

	### Generating the to_be_predicted list
	prediction_latest_small_data = sc.textFile(testing_path)
	header_prediction = prediction_latest_small_data.first()
	prediction_latest_small_data = prediction_latest_small_data.filter(lambda row : row != header_prediction)
	prediction_latest_small_data = prediction_latest_small_data.map(lambda x : tuple([int(i) for i in x.split(",")]))
	to_be_predicted = prediction_latest_small_data.collect()

	### Generating the similarity matrix
	similarity_matrix = {}
	count = 0
	for (movieId1, movieId2) in predicted_LSH:
		count += 1
		row1 = dict(list(m_rdd[movieId1]))
		row2 = dict(list(m_rdd[movieId2]))
		intersection = set(row1.keys()).intersection(set(row2.keys()))
		if intersection:
			arg1 = []
			arg2 = []
			for key in intersection:
				arg1.append(row1[key])
				arg2.append(row2[key])
			sim = pearsons_correlation(arg1, arg2)
			# print(sim)
			if not similarity_matrix.get(movieId1):
				similarity_matrix[movieId1] = [(movieId2, sim)]
			else:
				similarity_matrix[movieId1].append((movieId2, sim))
			if not similarity_matrix.get(movieId2):
				similarity_matrix[movieId2] = [(movieId1, sim)]
			else:
				similarity_matrix[movieId2].append((movieId1, sim))
		else:
			if not similarity_matrix.get(movieId1):
				similarity_matrix[movieId1] = [(movieId2, 0)]
			else:
				similarity_matrix[movieId1].append((movieId2, 0))
			if not similarity_matrix.get(movieId2):
				similarity_matrix[movieId2] = [(movieId1, 0)]
			else:
				similarity_matrix[movieId2].append((movieId1, 0))

	predictions = []
	cant_be_predicted = []
	for item in to_be_predicted:
		userId = item[0]
		movieId = item[1]

		most_similar_movies = similarity_matrix.get(item[1])
		if not most_similar_movies:
			cant_be_predicted.append((userId, movieId))
			pred = 0
			predictions.append((userId, movieId, pred))
		else:
			numerator = 0
			denominator = 0
			for (movie, similarity) in most_similar_movies:
				c_rating = dict(m_rdd[movie]).get(userId)

				if c_rating:
					numerator += c_rating * similarity
					denominator += abs(similarity)
			if userId == 1 and movieId == 1172:
				print(numerator, denominator)
			if denominator == 0:
				cant_be_predicted.append((userId, movieId))
				pred = 0
			else:
				pred = numerator / denominator
			predictions.append((userId, movieId, pred))

	ml_latest_small_data = data\
	.map(lambda x : [float(i) if index == 2 else int(i) for index, i in enumerate(x.split(",")[:3])])

	true_values = ml_latest_small_data\
	.filter(lambda x:(x[0], x[1]) in to_be_predicted)\
	.map(lambda x:(x[0], x[1], x[2])).collect()

	absolute_difference = {">=0 and <1":0, ">=1 and <2":0, ">=2 and <3":0, ">=3 and <4":0, ">=4":0}
	mse = 0

	calc_absolute_differences_and_mse(predictions=predictions, true_values=true_values)

	predictions = sorted(predictions, key=lambda x:(x[0], x[1]))

	save_path = "Vishal_Seshagiri_ItemBasedCF_with_LSH.txt"

	with open(save_path, "w") as file:
		for tup in predictions:
			file.write("{}, {}, {}\n".format(tup[0], tup[1], tup[2]))

	print("RMSE {}".format(mse**0.5))
	print(absolute_difference)


	### Generating the similarity matrix
	similarity_matrix = {}
	count = 0
	for (movieId1, movieId2) in itertools.combinations(m_rdd, 2):
		count += 1
		row1 = dict(list(m_rdd[movieId1]))
		row2 = dict(list(m_rdd[movieId2]))
		intersection = set(row1.keys()).intersection(set(row2.keys()))
		if intersection:
			arg1 = []
			arg2 = []
			for key in intersection:
				arg1.append(row1[key])
				arg2.append(row2[key])
			sim = pearsons_correlation(arg1, arg2)
			# print(sim)
			if not similarity_matrix.get(movieId1):
				similarity_matrix[movieId1] = [(movieId2, sim)]
			else:
				similarity_matrix[movieId1].append((movieId2, sim))
			if not similarity_matrix.get(movieId2):
				similarity_matrix[movieId2] = [(movieId1, sim)]
			else:
				similarity_matrix[movieId2].append((movieId1, sim))
		else:
			if not similarity_matrix.get(movieId1):
				similarity_matrix[movieId1] = [(movieId2, 0)]
			else:
				similarity_matrix[movieId1].append((movieId2, 0))
			if not similarity_matrix.get(movieId2):
				similarity_matrix[movieId2] = [(movieId1, 0)]
			else:
				similarity_matrix[movieId2].append((movieId1, 0))

	predictions = []
	cant_be_predicted = []
	for item in to_be_predicted:
		userId = item[0]
		movieId = item[1]

		most_similar_movies = similarity_matrix.get(item[1])
		if not most_similar_movies:
			cant_be_predicted.append((userId, movieId))
			pred = 0
			predictions.append((userId, movieId, pred))
		else:
			numerator = 0
			denominator = 0
			for (movie, similarity) in most_similar_movies:
				c_rating = dict(m_rdd[movie]).get(userId)

				if c_rating:
					numerator += c_rating * similarity
					denominator += abs(similarity)
			if userId == 1 and movieId == 1172:
				print(numerator, denominator)
			if denominator == 0:
				cant_be_predicted.append((userId, movieId))
				pred = 0
			else:
				pred = numerator / denominator
			predictions.append((userId, movieId, pred))

	ml_latest_small_data = data\
	.map(lambda x : [float(i) if index == 2 else int(i) for index, i in enumerate(x.split(",")[:3])])

	true_values = ml_latest_small_data\
	.filter(lambda x:(x[0], x[1]) in to_be_predicted)\
	.map(lambda x:(x[0], x[1], x[2])).collect()

	absolute_difference = {">=0 and <1":0, ">=1 and <2":0, ">=2 and <3":0, ">=3 and <4":0, ">=4":0}
	mse = 0

	calc_absolute_differences_and_mse(predictions=predictions, true_values=true_values)

	predictions = sorted(predictions, key=lambda x:(x[0], x[1]))

	save_path = "Vishal_Seshagiri_ItemBasedCF_without_LSH.txt"

	with open(save_path, "w") as file:
		for tup in predictions:
			file.write("{}, {}, {}\n".format(tup[0], tup[1], tup[2]))

	print("RMSE {}".format(mse**0.5))
	print(absolute_difference)