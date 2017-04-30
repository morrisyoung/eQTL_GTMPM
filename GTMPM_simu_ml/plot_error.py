import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines











if __name__=="__main__":





	## plot various rates for ml simu, real N, ml init
	"""
	##
	## real N
	ve_train = 6.25379060239e+12
	ve_test = 2.03421596757e+12

	# ## large N
	# ve_train = 8.36802024454e+13
	# ve_test = 2.7943939401e+13


	##
	list_error_train = np.load("./result_ml_realN_new/list_error_train_mlinit_largeR.npy")
	list_error_train = 1 - list_error_train / ve_train
	plt.plot(list_error_train, 'r-', label="large R train")
	##
	list_error_test = np.load("./result_ml_realN_new/list_error_test_mlinit_largeR.npy")
	list_error_test = 1 - list_error_test / ve_test
	plt.plot(list_error_test, 'r--', label="large R test")


	##
	list_error_train = np.load("./result_ml_realN_new/list_error_train_mlinit_midR.npy")
	list_error_train = 1 - list_error_train / ve_train
	plt.plot(list_error_train, 'g-', label="middle R train")
	##
	list_error_test = np.load("./result_ml_realN_new/list_error_test_mlinit_midR.npy")
	list_error_test = 1 - list_error_test / ve_test
	plt.plot(list_error_test, 'g--', label="middle R test")


	##
	list_error_train = np.load("./result_ml_realN_new/list_error_train_mlinit_smallR.npy")
	list_error_train = 1 - list_error_train / ve_train
	plt.plot(list_error_train, 'b-', label="small R train")
	##
	list_error_test = np.load("./result_ml_realN_new/list_error_test_mlinit_smallR.npy")
	list_error_test = 1 - list_error_test / ve_test
	plt.plot(list_error_test, 'b--', label="small R test")



	#plt.axis([0, len(list_error_test), 0.3, 1.0])
	plt.xlabel("iterations")
	plt.ylabel("variance explained")
	plt.title("ml model: ml init, and learn with different rates")
	plt.legend(loc=4)
	plt.grid(True)
	plt.show()
	"""








	####======================================================================================================
	####======================================================================================================
	####======================================================================================================
	####======================================================================================================






	plt.subplot(121)



	#### ml simu, real N, ml + nn inits
	##
	## real N
	ve_train = 6.25379060239e+12
	ve_test = 2.03421596757e+12


	##
	list_error_train = np.load("./result_ml_realN_new/list_error_train_mlinit_largeR.npy")
	list_error_train = 1 - list_error_train / ve_train
	plt.plot(list_error_train, 'r-', label="ml init train")
	##
	list_error_test = np.load("./result_ml_realN_new/list_error_test_mlinit_largeR.npy")
	list_error_test = 1 - list_error_test / ve_test
	plt.plot(list_error_test, 'r--', label="ml init test")


	##
	list_error_train = np.load("./result_ml_realN_new/list_error_train_nninit.npy")
	list_error_train = 1 - list_error_train / ve_train
	plt.plot(list_error_train, 'b-', label="nn init train")
	##
	list_error_test = np.load("./result_ml_realN_new/list_error_test_nninit.npy")
	list_error_test = 1 - list_error_test / ve_test
	plt.plot(list_error_test, 'b--', label="nn init test")





	#plt.axis([0, len(list_error_test), 0.3, 1.0])
	plt.xlabel("iterations")
	plt.ylabel("variance explained")
	plt.title("ml model: real sample size, 10th scale of features")
	plt.legend(loc=4)
	plt.grid(True)
	#plt.show()





	plt.subplot(122)




	#### ml simu, large N, ml + nn inits
	## large N
	ve_train = 8.36802024454e+13
	ve_test = 2.7943939401e+13


	threshold = 3000

	##
	list_error_train = np.load("./result_ml_largeN_new/list_error_train_mlinit.npy")
	list_error_train = 1 - list_error_train / ve_train
	plt.plot(list_error_train[:threshold], 'r-', label="ml init train")
	##
	list_error_test = np.load("./result_ml_largeN_new/list_error_test_mlinit.npy")
	list_error_test = 1 - list_error_test / ve_test
	plt.plot(list_error_test[:threshold], 'r--', label="ml init test")





	##
	list_error_train = np.load("./result_ml_largeN_new/list_error_train_nninit.npy")
	list_error_train = 1 - list_error_train / ve_train
	plt.plot(list_error_train[:threshold], 'b-', label="nn init train")
	##
	list_error_test = np.load("./result_ml_largeN_new/list_error_test_nninit.npy")
	list_error_test = 1 - list_error_test / ve_test
	plt.plot(list_error_test[:threshold], 'b--', label="nn init test")





	#plt.axis([0, threshold, 0.4, 0.9])				## fit for this scenario alone
	plt.axis([0, threshold, 0.3, 1.0])
	plt.xlabel("iterations")
	plt.ylabel("variance explained")
	plt.title("ml model: 10x sample size, 10th scale of features")
	plt.legend(loc=4)
	plt.grid(True)






	plt.show()












