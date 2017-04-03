import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines











if __name__=="__main__":






	##============================================================================================================
	##==== plot the effects of initialization for convergence
	##============================================================================================================

	ve_train = 82944750.0
	ve_test = 26776502.0985



	threshold = 500




	##======== ML model, R0.5
	list_error_train = np.load("./result/list_error_train_ml_LassoR0.5.npy")
	list_error_test = np.load("./result/list_error_test_ml_LassoR0.5.npy")
	#
	list_error_train = 1 - list_error_train / ve_train
	list_error_test = 1 - list_error_test / ve_test
	#
	plt.plot(list_error_train[:threshold], 'm-', label="ml, strong penalty, train")
	plt.plot(list_error_test[:threshold], 'm--', label="ml, strong penalty, test")



	##======== ML model
	list_error_train = np.load("./result/list_error_train_ml.npy")
	list_error_test = np.load("./result/list_error_test_ml.npy")
	#
	list_error_train = 1 - list_error_train / ve_train
	list_error_test = 1 - list_error_test / ve_test
	#
	plt.plot(list_error_train[:threshold], 'r-', label="ml, mild penalty, train")
	plt.plot(list_error_test[:threshold], 'r--', label="ml, mild penalty, test")


	##======== ML model, R0.05
	list_error_train = np.load("./result/list_error_train_ml_LassoR0.05.npy")
	list_error_test = np.load("./result/list_error_test_ml_LassoR0.05.npy")
	#
	list_error_train = 1 - list_error_train / ve_train
	list_error_test = 1 - list_error_test / ve_test
	#
	plt.plot(list_error_train[:threshold], 'c-', label="ml, weak penalty, train")
	plt.plot(list_error_test[:threshold], 'c--', label="ml, weak penalty, test")








	##======== NN model
	list_error_train = np.load("./result/list_error_train_nn.npy")
	list_error_test = np.load("./result/list_error_test_nn.npy")
	#
	list_error_train = 1 - list_error_train / ve_train
	list_error_test = 1 - list_error_test / ve_test
	#
	plt.plot(list_error_train[:threshold], 'b-', label="nn, strong penalty, train")
	plt.plot(list_error_test[:threshold], 'b--', label="nn, strong penalty, test")



	##======== NN model, small LASSO
	list_error_train = np.load("./result/list_error_train_nn_smallLasso.npy")
	list_error_test = np.load("./result/list_error_test_nn_smallLasso.npy")
	#
	list_error_train = 1 - list_error_train / ve_train
	list_error_test = 1 - list_error_test / ve_test
	#
	plt.plot(list_error_train[:threshold], 'g-', label="nn, mild penalty, train")
	plt.plot(list_error_test[:threshold], 'g--', label="nn, mild penalty, test")



	##======== NN model, smallest LASSO
	list_error_train = np.load("./result/list_error_train_nn_smallestLasso.npy")
	list_error_test = np.load("./result/list_error_test_nn_smallestLasso.npy")
	#
	list_error_train = 1 - list_error_train / ve_train
	list_error_test = 1 - list_error_test / ve_test
	#
	plt.plot(list_error_train[:threshold], 'y-', label="nn, weak penalty, train")
	plt.plot(list_error_test[:threshold], 'y--', label="nn, weak penalty, test")









	plt.xlabel("iterations")
	plt.ylabel("variance explained")
	plt.grid(True)
	#plt.axis([0, threshold, 0.3, 0.9])
	#plt.axis([0, threshold, 0.0, 1.0])
	plt.title("various penalty strength for wide linear model (of genetics to factors) Lasso initialization, for ml model and nn model")
	plt.legend()
	plt.show()







