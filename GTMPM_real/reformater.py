import numpy as np
import math






## (Mar.28, 2017) program checked to be correct






## reformat the data (train/test) into complete tensor with Nan (also its spread version and index file)
K = 28








if __name__ == "__main__":





	"""
	##========================
	##==== data train
	##========================
	header = "/ifs/scratch/c2b2/ip_lab/sy2515/GTEx_gtmnn/preprocess/data_train/"

	#
	X = np.load(header + "X.npy")
	# Y and Y_pos
	Y = []
	Y_pos = []
	for k in range(K):
		data = np.load(header + "Tensor_tissue_" + str(k) + ".npy")
		list_pos = np.load(header + "Tensor_tissue_" + str(k) + "_pos.npy")
		Y.append(data)
		Y_pos.append(list_pos)
	Y = np.array(Y)
	Y_pos = np.array(Y_pos)

	##
	K = len(Y)
	N = len(X)
	J = len(Y[0][0])
	print "shape (train):"
	print "K:", K
	print "N:", N
	print "J:", J

	# reformat train, and save
	Y_train = np.zeros((K, N, J)) + float("Nan")
	for k in range(K):
		for i in range(len(Y_pos[k])):
			indiv = Y_pos[k][i]
			exp = Y[k][i]
			Y_train[k][indiv] = exp
	print "Y_train shape:", Y_train.shape
	np.save("./data_tf_train_test/Y_train", Y_train)
	#
	Y_train_spread = np.reshape(Y_train, -1)
	np.save("./data_tf_train_test/Y_train_spread", Y_train_spread)
	#
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
	np.save("./data_tf_train_test/Y_train_index", list_index)






	##=======================
	##==== data test
	##=======================
	header = "/ifs/scratch/c2b2/ip_lab/sy2515/GTEx_gtmnn/preprocess/data_test/"

	#
	X = np.load(header + "X.npy")
	# Y and Y_pos
	K = 28										## TODO: specify the number of tissues
	Y = []
	Y_pos = []
	for k in range(K):
		data = np.load(header + "Tensor_tissue_" + str(k) + ".npy")
		list_pos = np.load(header + "Tensor_tissue_" + str(k) + "_pos.npy")
		Y.append(data)
		Y_pos.append(list_pos)
	Y = np.array(Y)
	Y_pos = np.array(Y_pos)

	##
	K = len(Y)
	N = len(X)
	J = len(Y[0][0])
	print "shape (train):"
	print "K:", K
	print "N:", N
	print "J:", J

	# reformat train, and save
	Y_test = np.zeros((K, N, J)) + float("Nan")
	for k in range(K):
		for i in range(len(Y_pos[k])):
			indiv = Y_pos[k][i]
			exp = Y[k][i]
			Y_test[k][indiv] = exp
	print "Y_test shape:", Y_test.shape
	np.save("./data_tf_train_test/Y_test", Y_test)
	#
	Y_test_spread = np.reshape(Y_test, -1)
	np.save("./data_tf_train_test/Y_test_spread", Y_test_spread)
	#
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
	np.save("./data_tf_train_test/Y_test_index", list_index)
	"""






	####################################################################################
	####################################################################################
	####################################################################################




	##=======================
	##==== cis train
	##=======================
	header = "/ifs/scratch/c2b2/ip_lab/sy2515/GTEx_gtmnn/preprocess/data_train/"

	#
	X = np.load(header + "X.npy")
	# Y and Y_pos
	Y = []
	Y_pos = []
	for k in range(K):
		data = np.load(header + "Tensor_tissue_" + str(k) + ".npy")
		list_pos = np.load(header + "Tensor_tissue_" + str(k) + "_pos.npy")
		Y.append(data)
		Y_pos.append(list_pos)
	Y = np.array(Y)
	Y_pos = np.array(Y_pos)

	##
	K = len(Y)
	N = len(X)
	J = len(Y[0][0])
	print "shape (train):"
	print "K:", K
	print "N:", N
	print "J:", J

	##
	Y_cis_train = np.load("/ifs/scratch/c2b2/ip_lab/sy2515/GTEx_gtmnn/workbench54/data_real_init/Y_cis_train.npy")
	Y = Y_cis_train

	# reformat cis train, and save
	Y_cis_train = np.zeros((K, N, J)) + float("Nan")
	for k in range(K):
		for i in range(len(Y_pos[k])):
			indiv = Y_pos[k][i]
			exp = Y[k][i]
			Y_cis_train[k][indiv] = exp
	print "Y_cis_train shape:", Y_cis_train.shape
	np.save("./data_tf_train_test/Y_cis_train", Y_cis_train)
	#
	Y_cis_train_spread = np.reshape(Y_cis_train, -1)
	np.save("./data_tf_train_test/Y_cis_train_spread", Y_cis_train_spread)
	#
	K = len(Y_cis_train)
	N = len(Y_cis_train[0])
	J = len(Y_cis_train[0][0])
	list_index = []
	for k in range(K):
		for n in range(N):
			if not math.isnan(Y_cis_train[k][n][0]):
				list_temp = np.arange(J) + k * N * J + n * J
				list_index += list_temp.tolist()
	print "Y_cis_train_index shape:", len(list_index)
	np.save("./data_tf_train_test/Y_cis_train_index", list_index)




	##=======================
	##==== cis test
	##=======================
	header = "/ifs/scratch/c2b2/ip_lab/sy2515/GTEx_gtmnn/preprocess/data_test/"

	#
	X = np.load(header + "X.npy")
	# Y and Y_pos
	Y = []
	Y_pos = []
	for k in range(K):
		data = np.load(header + "Tensor_tissue_" + str(k) + ".npy")
		list_pos = np.load(header + "Tensor_tissue_" + str(k) + "_pos.npy")
		Y.append(data)
		Y_pos.append(list_pos)
	Y = np.array(Y)
	Y_pos = np.array(Y_pos)

	##
	K = len(Y)
	N = len(X)
	J = len(Y[0][0])
	print "shape (test):"
	print "K:", K
	print "N:", N
	print "J:", J

	##
	Y_cis_test = np.load("/ifs/scratch/c2b2/ip_lab/sy2515/GTEx_gtmnn/workbench54/data_real_init/Y_cis_test.npy")
	Y = Y_cis_test

	# reformat cis train, and save
	Y_cis_test = np.zeros((K, N, J)) + float("Nan")
	for k in range(K):
		for i in range(len(Y_pos[k])):
			indiv = Y_pos[k][i]
			exp = Y[k][i]
			Y_cis_test[k][indiv] = exp
	print "Y_cis_test shape:", Y_cis_test.shape
	np.save("./data_tf_train_test/Y_cis_test", Y_cis_test)
	#
	Y_cis_test_spread = np.reshape(Y_cis_test, -1)
	np.save("./data_tf_train_test/Y_cis_test_spread", Y_cis_test_spread)
	#
	K = len(Y_cis_test)
	N = len(Y_cis_test[0])
	J = len(Y_cis_test[0][0])
	list_index = []
	for k in range(K):
		for n in range(N):
			if not math.isnan(Y_cis_test[k][n][0]):
				list_temp = np.arange(J) + k * N * J + n * J
				list_index += list_temp.tolist()
	print "Y_cis_test_index shape:", len(list_index)
	np.save("./data_tf_train_test/Y_cis_test_index", list_index)











