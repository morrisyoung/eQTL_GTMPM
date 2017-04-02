import numpy as np
import math
import timeit







##
## non-linear neural network modeling
## simulate sparsely
## will assume the cis- effects have been removed and only simulate the trans- factor effects
## will simulate multiple tissues, and incomplete gene tensor
## will simulate 10th scale
##






##==== scale of the input data (real data)
I = 244519
J = 1942
D = 40
#N = 450
N = 4500

K = 28










if __name__ == "__main__":







	# X
	X = np.random.random_sample((N, I)) * 2

	# beta1
	beta1 = np.random.normal(size=(I+1, D))
	beta1_indicator = np.random.binomial(1, 0.1, (I, D))						## TODO: sparsity
	m_ones = np.ones((1, D))
	beta1_indicator_ext = np.concatenate((beta1_indicator, m_ones), axis=0)		# (I+1) x D
	beta1 = np.multiply(beta1, beta1_indicator_ext)
	print "beta1 shape:", beta1.shape



	################################################
	################################################
	## to make the activation not saturated
	beta1 = beta1 / 10
	################################################
	################################################



	# beta2
	beta2 = np.random.normal(size=(K, D+1, J))
	beta2_indicator = np.random.binomial(1, 0.5, (D, J))						## TODO: sparsity
	m_ones = np.ones((1, J))
	beta2_indicator_ext = np.concatenate((beta2_indicator, m_ones), axis=0)
	for k in range(len(beta2)):
		beta2[k] = np.multiply(beta2[k], beta2_indicator_ext)
	print "beta2 shape:", beta2.shape

	# Y
	# first layer
	m_ones = np.ones((N, 1))
	X_ext = np.concatenate((X, m_ones), axis=1)								# N x (I+1)
	m_factor = np.dot(X_ext, beta1)											# N x D


	################################################
	################################################
	# activation function
	m_factor = 1.0 / (1.0 + np.exp( -m_factor ))
	################################################
	################################################


	# second layer
	m_ones = np.ones((N, 1))
	m_factor_ext = np.concatenate((m_factor, m_ones), axis=1)				# N x (D+1)
	# (indiv, factor+1) x (tissue, factor+1, gene) = (indiv, tissue, gene)
	Y = np.tensordot(m_factor_ext, beta2, axes=([1],[1]))
	Y = np.transpose(Y, (1, 0, 2))
	Y_spread = np.reshape(Y, -1)







	##
	## make the tensor incomplete, for both train set and test set
	##
	shape = Y.shape
	K = shape[0]
	N = shape[1]
	J = shape[2]
	threshold = int(0.5 * N)						## to make these amount Nan
	for k in range(K):
		list_index = np.arange(N)
		np.random.shuffle(list_index)
		for index in list_index[:threshold]:
			Y[k][index] = np.zeros(J) + float("Nan")

	## train/test split
	threshold = int(N * 0.75)
	print "train indiv amount:", threshold
	Y_train = Y[:, :threshold, :]
	print "Y_train shape:", Y_train.shape
	Y_test = Y[:, threshold:, :]
	print "Y_test shape:", Y_test.shape
	##
	Y_train_spread = np.reshape(Y_train, -1)
	Y_test_spread = np.reshape(Y_test, -1)

	## coverage check (only needed for train set)
	count = 0
	repo_tissue = {}
	repo_indiv = {}
	for k in range(len(Y_train)):
		for n in range(len(Y_train[k])):
			if not math.isnan(Y_train[k][n][0]):
				repo_tissue[k] = 1
				repo_indiv[n] = 1
				count += 1
	print "train tissue coverage: ", len(repo_tissue),
	print "train indiv coverage: ", len(repo_indiv)

	## also split X
	X_train = X[:threshold]
	X_test = X[threshold:]
	print "X_train and X_test shapes:", X_train.shape, X_test.shape

	## get the index of spreads, train and test
	# train
	K = len(Y_train)
	N = len(Y_train[0])
	J = len(Y_train[0][0])
	list_index = []
	for k in range(K):
		for n in range(N):
			if not math.isnan(Y_train[k][n][0]):
				list_temp = np.arange(J) + k * N * J + n * J
				list_index += list_temp.tolist()
	print "Y_train_index shape:", len(list_index)
	np.save("./data_simu_nn/Y_train_index", list_index)
	# test
	K = len(Y_test)
	N = len(Y_test[0])
	J = len(Y_test[0][0])
	list_index = []
	for k in range(K):
		for n in range(N):
			if not math.isnan(Y_test[k][n][0]):
				list_temp = np.arange(J) + k * N * J + n * J
				list_index += list_temp.tolist()
	print "Y_test_index shape:", len(list_index)
	np.save("./data_simu_nn/Y_test_index", list_index)










	## extract the VE of train samples and test samples
	####
	Y_train_matrix = []
	for k in range(len(Y_train)):
		for n in range(len(Y_train[k])):
			if not math.isnan(Y_train[k][n][0]):
				Y_train_matrix.append(Y_train[k][n])
	Y_train_matrix = np.array(Y_train_matrix)
	print "train matrix shape:", Y_train_matrix.shape
	##
	Y_train_mean = np.mean(Y_train_matrix, axis=0)
	ve_train = np.sum(np.square(Y_train_matrix - Y_train_mean))
	print "ve_train:", ve_train


	####
	Y_test_matrix = []
	for k in range(len(Y_test)):
		for n in range(len(Y_test[k])):
			if not math.isnan(Y_test[k][n][0]):
				Y_test_matrix.append(Y_test[k][n])
	Y_test_matrix = np.array(Y_test_matrix)
	print "test matrix shape:", Y_test_matrix.shape
	##
	Y_test_mean = np.mean(Y_test_matrix, axis=0)
	ve_test = np.sum(np.square(Y_test_matrix - Y_test_mean))
	print "ve_test:", ve_test











	##==== save data
	np.save("./data_simu_nn/X", X)
	np.save("./data_simu_nn/X_train", X_train)
	np.save("./data_simu_nn/X_test", X_test)
	np.save("./data_simu_nn/Y", Y)
	np.save("./data_simu_nn/Y_spread", Y_spread)
	np.save("./data_simu_nn/Y_train", Y_train)
	np.save("./data_simu_nn/Y_train_spread", Y_train_spread)
	np.save("./data_simu_nn/Y_test", Y_test)
	np.save("./data_simu_nn/Y_test_spread", Y_test_spread)
	np.save("./data_simu_nn/beta1_real", beta1)
	np.save("./data_simu_nn/beta2_real", beta2)








	##====================================================
	## simu another copy as the init (randomly use another copy to init -- we can of course init more wisely)
	##====================================================
	# beta1
	beta1 = np.random.normal(size=(I+1, D))
	beta1_indicator = np.random.binomial(1, 0.1, (I, D))						## TODO: sparsity
	m_ones = np.ones((1, D))
	beta1_indicator_ext = np.concatenate((beta1_indicator, m_ones), axis=0)		# (I+1) x D
	beta1 = np.multiply(beta1, beta1_indicator_ext)
	print "beta1 shape:", beta1.shape



	################################################
	################################################
	## to make the activation not saturated
	beta1 = beta1 / 10
	################################################
	################################################



	# beta2
	beta2 = np.random.normal(size=(K, D+1, J))
	beta2_indicator = np.random.binomial(1, 0.5, (D, J))						## TODO: sparsity
	m_ones = np.ones((1, J))
	beta2_indicator_ext = np.concatenate((beta2_indicator, m_ones), axis=0)
	for k in range(len(beta2)):
		beta2[k] = np.multiply(beta2[k], beta2_indicator_ext)
	print "beta2 shape:", beta2.shape
	##
	np.save("./data_simu_nn/beta1_init", beta1)
	np.save("./data_simu_nn/beta2_init", beta2)













