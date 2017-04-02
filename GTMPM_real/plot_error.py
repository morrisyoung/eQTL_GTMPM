import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines









if __name__=="__main__":




	ve_train = 82944750.0
	ve_test = 26776502.0985



	threshold = 500


	##======== ML model
	list_error_train = np.load("./result/list_error_train_ml.npy")
	list_error_test = np.load("./result/list_error_test_ml.npy")
	#
	list_error_train = 1 - list_error_train / ve_train
	list_error_test = 1 - list_error_test / ve_test
	#
	plt.plot(list_error_train[:threshold], 'r-', label="ml train")
	plt.plot(list_error_test[:threshold], 'r--', label="ml test")



	##======== NN model
	list_error_train = np.load("./result/list_error_train_nn.npy")
	list_error_test = np.load("./result/list_error_test_nn.npy")
	#
	list_error_train = 1 - list_error_train / ve_train
	list_error_test = 1 - list_error_test / ve_test
	#
	plt.plot(list_error_train[:threshold], 'b-', label="nn train")
	plt.plot(list_error_test[:threshold], 'b--', label="nn test")





	##======== NN model, small LASSO
	list_error_train = np.load("./result/list_error_train_nn_smallLasso.npy")
	list_error_test = np.load("./result/list_error_test_nn_smallLasso.npy")
	#
	list_error_train = 1 - list_error_train / ve_train
	list_error_test = 1 - list_error_test / ve_test
	#
	plt.plot(list_error_train[:threshold], 'g-', label="nn train, small LASSO")
	plt.plot(list_error_test[:threshold], 'g--', label="nn test, small LASSO")





	plt.xlabel("iterations")
	plt.ylabel("variance explained")
	plt.grid(True)
	plt.axis([0, threshold, 0.3, 0.9])
	plt.legend()
	plt.show()







