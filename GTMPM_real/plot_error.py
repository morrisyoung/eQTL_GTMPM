import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines









def load_array_txt(filename):
	array = []
	file = open(filename, 'r')
	while 1:
		line = (file.readline()).strip()
		if not line:
			break

		value = float(line)
		array.append(value)
	file.close()
	array = np.array(array)
	return array








if __name__=="__main__":




	ls_train = "--"
	ls_test = "-"





	##============================================================================================================
	##==== plot the effects of initialization for convergence
	##============================================================================================================

	ve_train = 82944750.0
	ve_test = 26776502.0985



	threshold = 500



	fig = plt.figure(1)
	plt.subplot(121)



	##======== ML model, R0.5
	list_error_train = np.load("./result/list_error_train_ml_LassoR0.5.npy")
	list_error_test = np.load("./result/list_error_test_ml_LassoR0.5.npy")
	#
	list_error_train = 1 - list_error_train / ve_train
	list_error_test = 1 - list_error_test / ve_test
	#
	plt.plot(list_error_train[:threshold], color='m', ls=ls_train, label="ml, strong penalty, train")
	plt.plot(list_error_test[:threshold], color='m', ls=ls_test, label="ml, strong penalty, test")



	##======== ML model
	list_error_train = np.load("./result/list_error_train_ml.npy")
	list_error_test = np.load("./result/list_error_test_ml.npy")
	#
	list_error_train = 1 - list_error_train / ve_train
	list_error_test = 1 - list_error_test / ve_test
	#
	plt.plot(list_error_train[:threshold], color='r', ls=ls_train, label="ml, mild penalty, train")
	plt.plot(list_error_test[:threshold], color='r', ls=ls_test, label="ml, mild penalty, test")


	##======== ML model, R0.05
	list_error_train = np.load("./result/list_error_train_ml_LassoR0.05.npy")
	list_error_test = np.load("./result/list_error_test_ml_LassoR0.05.npy")
	#
	list_error_train = 1 - list_error_train / ve_train
	list_error_test = 1 - list_error_test / ve_test
	#
	plt.plot(list_error_train[:threshold], color='c', ls=ls_train, label="ml, weak penalty, train")
	plt.plot(list_error_test[:threshold], color='c', ls=ls_test, label="ml, weak penalty, test")




	##============================================================================================================




	##======== NN model
	list_error_train = np.load("./result/list_error_train_nn.npy")
	list_error_test = np.load("./result/list_error_test_nn.npy")
	#
	list_error_train = 1 - list_error_train / ve_train
	list_error_test = 1 - list_error_test / ve_test
	#
	plt.plot(list_error_train[:threshold], color='b', ls=ls_train, label="nn, strong penalty, train")
	plt.plot(list_error_test[:threshold], color='b', ls=ls_test, label="nn, strong penalty, test")



	##======== NN model, small LASSO
	list_error_train = np.load("./result/list_error_train_nn_smallLasso.npy")
	list_error_test = np.load("./result/list_error_test_nn_smallLasso.npy")
	#
	list_error_train = 1 - list_error_train / ve_train
	list_error_test = 1 - list_error_test / ve_test
	#
	plt.plot(list_error_train[:threshold], color='g', ls=ls_train, label="nn, mild penalty, train")
	plt.plot(list_error_test[:threshold], color='g', ls=ls_test, label="nn, mild penalty, test")



	##======== NN model, smallest LASSO
	list_error_train = np.load("./result/list_error_train_nn_smallestLasso.npy")
	list_error_test = np.load("./result/list_error_test_nn_smallestLasso.npy")
	#
	list_error_train = 1 - list_error_train / ve_train
	list_error_test = 1 - list_error_test / ve_test
	#
	plt.plot(list_error_train[:threshold], color='y', ls=ls_train, label="nn, weak penalty, train")
	plt.plot(list_error_test[:threshold], color='y', ls=ls_test, label="nn, weak penalty, test")




	##============================================================================================================




	## cis
	list_error_train = load_array_txt("./result_cis/error_total_online_train.txt")
	list_error_test = load_array_txt("./result_cis/error_total_online_test.txt")
	#
	list_error_train = 1 - list_error_train / ve_train
	list_error_test = 1 - list_error_test / ve_test
	#
	plt.plot(list_error_train[:threshold], color='magenta', ls=ls_train, label="cis, train")
	plt.plot(list_error_test[:threshold], color='magenta', ls=ls_test, label="cis, test")










	##
	##
	plt.xlabel("iterations")
	plt.ylabel("variance explained")
	plt.grid(True)
	plt.axis([0, threshold, 0.35, 0.72])
	#plt.axis([0, threshold, 0.0, 1.0])
	#plt.title("various penalty strength for wide linear model (of genetics to factors) Lasso initialization, for ml model and nn model")
	#plt.title("multiple linear and neural net, different init Lasso (genetics to factors)")
	plt.legend(loc=4)
	#plt.show()
	##
	##








	##============================================================================================================



	plt.subplot(122)


	threshold = 1000



	##======== TM model, LassoR0.005
	list_error_train = np.load("./result_tm_new/list_error_train_tm_LassoR0.005.npy")
	list_error_test = np.load("./result_tm_new/list_error_test_tm_LassoR0.005.npy")
	#
	list_error_train_largeR = 1 - list_error_train / ve_train
	list_error_test_largeR = 1 - list_error_test / ve_test


	##======== TM model, LassoR0.001
	list_error_train = np.load("./result_tm_new/list_error_train_tm_LassoR0.001.npy")
	list_error_test = np.load("./result_tm_new/list_error_test_tm_LassoR0.001.npy")
	#
	list_error_train_mildR = 1 - list_error_train / ve_train
	list_error_test_mildR = 1 - list_error_test / ve_test


	##======== TM model, LassoR0.0005
	list_error_train = np.load("./result_tm_new/list_error_train_tm_LassoR0.0005.npy")
	list_error_test = np.load("./result_tm_new/list_error_test_tm_LassoR0.0005.npy")
	#
	list_error_train_smallR = 1 - list_error_train / ve_train
	list_error_test_smallR = 1 - list_error_test / ve_test


	## normal plot
	#
	#plt.plot(list_error_train_largeR[:], 'r-', label="tm, strong penalty, train")
	#plt.plot(list_error_test_largeR[:], 'r--', label="tm, strong penalty, test")
	#
	#plt.plot(list_error_train_mildR[:], 'b-', label="tm, mild penalty, train")
	#plt.plot(list_error_test_mildR[:], 'b--', label="tm, mild penalty, test")
	#
	#plt.plot(list_error_train_smallR[:], 'g-', label="tm, weak penalty, train")
	#plt.plot(list_error_test_smallR[:], 'g--', label="tm, weak penalty, test")


	# re-order
	plt.plot(list_error_train_largeR[:], color='r', ls=ls_train, label="tm, strong penalty, train")
	#
	plt.plot(list_error_train_mildR[:], color='b', ls=ls_train, label="tm, mild penalty, train")
	#
	plt.plot(list_error_train_smallR[:], color='g', ls=ls_train, label="tm, weak penalty, train")
	##
	plt.plot(list_error_test_largeR[:], color='r', ls=ls_test, label="tm, strong penalty, test")
	plt.plot(list_error_test_mildR[:], color='b', ls=ls_test, label="tm, mild penalty, test")
	plt.plot(list_error_test_smallR[:], color='g', ls=ls_test, label="tm, weak penalty, test")






	# plot delta
	"""
	##
	for i in range(1, len(list_error_test_largeR)):
		list_error_test_largeR[i-1] = list_error_test_largeR[i] - list_error_test_largeR[i-1]
	list_error_test_largeR[-1] = list_error_test_largeR[-2]
	##
	for i in range(1, len(list_error_test_mildR)):
		list_error_test_mildR[i-1] = list_error_test_mildR[i] - list_error_test_mildR[i-1]
	list_error_test_mildR[-1] = list_error_test_mildR[-2]
	##
	for i in range(1, len(list_error_test_smallR)):
		list_error_test_smallR[i-1] = list_error_test_smallR[i] - list_error_test_smallR[i-1]
	list_error_test_smallR[-1] = list_error_test_smallR[-2]

	for i in range(100):
		list_error_test_largeR[i] = 0
		list_error_test_mildR[i] = 0
		list_error_test_smallR[i] = 0


	plt.plot(list_error_test_largeR[:], 'r--', label="tm, strong penalty, test")
	plt.plot(list_error_test_mildR[:], 'b--', label="tm, mild penalty, test")
	plt.plot(list_error_test_smallR[:], 'g--', label="tm, weak penalty, test")
	"""










	plt.xlabel("iterations")
	plt.ylabel("variance explained")
	#plt.ylabel("delta (from #100 iter)")
	plt.grid(True)

	plt.axis([0, 10000, 0.35, 0.72])

	#plt.axis([0, 200, 0.35, 0.72])
	#plt.axis([0, threshold, 0.0, 1.0])
	#plt.title("various penalty strength for wide linear model (of genetics to factors) Lasso initialization, for ml model and nn model")
	#plt.title("tensor predictive, different init Lasso (genetics to factors)")
	plt.legend(loc=1, ncol=2)
	#plt.legend(loc=1)





	plt.suptitle('multiple linear (ml), neural net (nn), tensor predictive modeling (tm): different init Lasso penalty (of genetics to factors)')
	plt.show()













