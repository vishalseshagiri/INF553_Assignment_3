from pyspark import SparkContext
from collections import OrderedDict
import sys
import itertools
import time

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

if __name__=="__main__":
	sc = SparkContext(appName="Assignment 3 Collaborative Filtering User Based")

	start_time = time.time()

	ratings_path = sys.argv[1]
	testing_file_path = sys.argv[2]

	### Data to be loaded

	ml_latest_small_data = sc.textFile(ratings_path)

	prediction_latest_small_data = sc.textFile(testing_file_path)

	header_prediction = prediction_latest_small_data.first()
	prediction_latest_small_data = prediction_latest_small_data\
	.filter(lambda row : row != header_prediction)\
	.map(lambda x : tuple([int(i) for i in x.split(",")]))

	header = ml_latest_small_data.first()
	ml_latest_small_data = ml_latest_small_data\
	.filter(lambda row : row != header)\
	.map(lambda x : [float(i) if index == 2 else int(i) for index, i in enumerate(x.split(",")[:3])])

	to_be_predicted = sorted(prediction_latest_small_data.collect())

	ratings = ml_latest_small_data\
	.filter(lambda x:(x[0], x[1]) not in to_be_predicted)\
	.map(lambda x:(x[0], (x[1], x[2]))).groupByKey()

	ratings = OrderedDict(sorted(ratings.collect(), key=lambda x:x[0]))

	similarity_matrix = {}

	for (userId1, userId2) in itertools.combinations(ratings, 2):
		row1 = dict(list(ratings[userId1]))
		row2 = dict(list(ratings[userId2]))
		intersection = set(row1.keys()).intersection(set(row2.keys()))

		if intersection:
			arg1 = []
			arg2 = []

			for key in intersection:
				arg1.append(row1[key])
				arg2.append(row2[key])

			sim = pearsons_correlation(arg1, arg2)
			if not similarity_matrix.get(userId1):
				similarity_matrix[userId1] = [(userId2, sim, sum(arg2)/len(arg2))]
			else:
				similarity_matrix[userId1].append((userId2, sim, sum(arg2)/len(arg2)))
			if not similarity_matrix.get(userId2):
				similarity_matrix[userId2] = [(userId1, sim, sum(arg2)/len(arg1))]
			else:
				similarity_matrix[userId2].append((userId1, sim, sum(arg1)/len(arg1)))
		else:
			if not similarity_matrix.get(userId1):
				similarity_matrix[userId1] = [(userId2, 0, 0)]
			else:
				similarity_matrix[userId1].append((userId2, 0, 0))
			if not similarity_matrix.get(userId2):
				similarity_matrix[userId2] = [(userId1, 0, 0)]
			else:
				similarity_matrix[userId2].append((userId1, 0, 0))


	predictions = []
	cant_be_predicted = []
	for item in to_be_predicted:
		userId = item[0]
		movieId = item[1]

		most_similar_users = sorted(similarity_matrix[item[0]], key=lambda x:x[1], reverse=True)

		numerator = 0
		denominator = 0

		for (user, similarity, avg_c_user_rating) in most_similar_users:
			c_rating = dict(ratings[user]).get(movieId)

			if c_rating:
				values = dict(ratings[user]).values()

				numerator += (c_rating - avg_c_user_rating) * similarity

				denominator += abs(similarity)
		# if userId == 1 and movieId == 1172:
			# print(numerator, denominator)
	
		user_ratings = dict(ratings[userId]).values()
		avg_ratings = sum(user_ratings)/len(user_ratings)
		if denominator == 0:
			cant_be_predicted.append((userId, movieId))
			pred = 0
		else:
			pred = numerator / denominator
	
		predictions.append((userId, movieId, avg_ratings + pred))

	absolute_difference = OrderedDict({">=0 and <1":0, ">=1 and <2":0, ">=2 and <3":0, ">=3 and <4":0, ">=4":0})
	mse = 0

	true_values = ml_latest_small_data.filter(lambda x:(x[0], x[1]) in to_be_predicted).map(lambda x:(x[0], x[1], x[2])).collect()

	calc_absolute_differences_and_mse(predictions=predictions, true_values=true_values)

	end_time = time.time()

	with open("Vishal_Seshagiri_UserBasedCF.txt", "w") as file:
		for tup in predictions:
			file.write("{}, {}, {}\n".format(tup[0], tup[1], tup[2]))

	for key in sorted(absolute_difference.keys()):
		print("{}:{}".format(key, absolute_difference[key]))

	print("RMSE: {}".format(mse**.5))

	print("Time: {}".format(end_time - start_time))

# >=0 and <1:14959
# >=1 and <2:4426
# >=2 and <3:721
# >=3 and <4:139
# >=4:11
# RMSE: 0.954399154922
# Time: 209.217955828
