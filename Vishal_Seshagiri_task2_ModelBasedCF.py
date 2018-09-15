from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark import Accumulator
import time
import sys


def calc_absolute_differences_and_mse(iterator):
	global mse

	iterator = list(iterator)
	true_rating = iterator[0]
	prediction = iterator[1]
	diff = abs(true_rating - prediction)
	mse += diff ** 2
	
	if diff < 1:
		return (">=0 and <1", 1)
	elif diff < 2:
		return (">=1 and <2", 1)
	elif diff < 3:
		return (">=2 and <3", 1)
	elif diff < 4:
		return (">=3 and <4", 1)
	else:
		return (">=4", 1)

def line_in_csv(iterator):
	iterator = list(iterator)
	return ",".join(iterator[0][0], iterator[0][1], iterator[1])

if __name__=="__main__":

	start_time = time.time()
	ratings_file_path = sys.argv[1]
	testing_file_path = sys.argv[2]

	sc = SparkContext(appName="Task 2 Model Based CF")

	test_data = sc.textFile(testing_file_path)
	test_head = test_data.first()
	test_data = test_data.filter(lambda x: x != test_head)
	test_data = test_data.map(lambda l: l.split(','))\
	    .map(lambda l: (int(l[0]), int(l[1])))
	len_test = test_data.count()

	data = sc.textFile(ratings_file_path)
	head = data.first()
	ratings = data.filter(lambda x : x != head)
	ratings = ratings.map(lambda l: l.split(',')).map(lambda l: (int(l[0]), int(l[1]), float(l[2])))

	join_test_data = test_data.map(lambda l : ((l[0], l[1]), 0))

	modified_ratings = ratings.map(lambda l : ((l[0], l[1]), l[2]))

	true_ratings  = modified_ratings.join(join_test_data).map(lambda x : (x[0], x[1][0]))

	train_data = modified_ratings.subtract(true_ratings).map(lambda l : Rating(l[0][0], l[0][1], l[1]))

	# Build the recommendation model using Alternating Least Squares
	if len_test > 20256:
		rank = 5
		numIterations = 20
		save_path = "Vishal_Seshagiri_ModelBasedCF_Big.txt"
	else:
		rank = 5
		numIterations = 12
		save_path = "Vishal_Seshagiri_ModelBasedCF_Small.txt"

	l = 0.01
	mse = sc.accumulator(0)
	model = ALS.train(train_data, rank, numIterations, lambda_=l)
	
	# Evaluate the model on training data
	predictions = model.predictAll(test_data).map(lambda r: ((r[0], r[1]), r[2]))

	ratesAndPreds = true_ratings.join(predictions)

	base_line_predictions = dict(ratesAndPreds.map(lambda r: (r[1][0], r[1][1])).map(calc_absolute_differences_and_mse).groupByKey().mapValues(lambda values : sum(values)).collect())

	rmse = (mse.value / float(len_test))**0.5

	end_time = time.time()

	with open(save_path, "w") as file:
		for tup in predictions.sortByKey().collect():
			file.write("{}, {}, {}\n".format(tup[0][0], tup[0][1], tup[1]))

	for key in sorted(base_line_predictions.keys()):
		print("{}:{}".format(key, base_line_predictions[key]))

	print("RMSE: {}".format(rmse))

	print("Time: {}".format(end_time - start_time))

# Small data
# >=0 and <1:12989
# >=1 and <2:4286
# >=2 and <3:1074
# >=3 and <4:292
# >=4:92
# RMSE: 1.09556905027
# Time: 8.97816205025

# 20m data
# >=0 and <1:3241770
# >=1 and <2:700781
# >=2 and <3:91318
# >=3 and <4:11446
# >=4:1016
# RMSE: 0.826982854717
# Time: 696.621098042
