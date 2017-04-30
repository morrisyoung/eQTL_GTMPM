import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines











if __name__=="__main__":






	plt.subplot(121)

	##====================================
	##============ the real N
	##====================================
	#simu_nn real N:
	#train matrix shape: (4747, 1942)
	#ve_train: 81902627.3759
	#test matrix shape: (1553, 1942)
	#ve_test: 27098182.6352
	ve_train = 81902627.3759
	ve_test = 27098182.6352


	threshold = 500


	####
	list_error_train = np.load("./result_nn/list_error_train_realN_nninit.npy")
	list_error_test = np.load("./result_nn/list_error_test_realN_nninit.npy")

	#
	list_error_train = 1 - list_error_train / ve_train
	list_error_test = 1 - list_error_test / ve_test
	#
	plt.plot(list_error_train[:threshold], 'r-', label="nn init train")
	plt.plot(list_error_test[:threshold], 'r--', label="nn init test")
	##
	#plt.semilogx(list_error_train[:threshold], 'r-', label="nn init train", basex=2)
	#plt.semilogx(list_error_test[:threshold], 'r--', label="nn init test", basex=2)






	####
	list_error_train = np.load("./result_nn/list_error_train_realN_nnrand.npy")
	list_error_test = np.load("./result_nn/list_error_test_realN_nnrand.npy")

	#
	list_error_train = 1 - list_error_train / ve_train
	list_error_test = 1 - list_error_test / ve_test
	#
	plt.plot(list_error_train[:threshold], 'g-', label="nn rand train")
	plt.plot(list_error_test[:threshold], 'g--', label="nn rand test")
	##
	#plt.semilogx(list_error_train[:threshold], 'g-', label="nn rand train", basex=2)
	#plt.semilogx(list_error_test[:threshold], 'g--', label="nn rand test", basex=2)






	####
	list_error_train = np.load("./result_nn/list_error_train_realN_mlinit.npy")
	list_error_test = np.load("./result_nn/list_error_test_realN_mlinit.npy")

	#
	list_error_train = 1 - list_error_train / ve_train
	list_error_test = 1 - list_error_test / ve_test
	#
	plt.plot(list_error_train[:threshold], 'b-', label="ml init train")
	plt.plot(list_error_test[:threshold], 'b--', label="ml init test")
	##
	#plt.semilogx(list_error_train[:threshold], 'b-', label="ml init train", basex=2)
	#plt.semilogx(list_error_test[:threshold], 'b--', label="ml init test", basex=2)










	plt.xlabel("iterations")
	plt.ylabel("variance explained")
	plt.title("nn model: real sample size, 10th scale of features")
	plt.grid(True)
	plt.legend(loc=4)
	plt.axis([0, threshold, 0.3, 0.9])








	plt.subplot(122)


	##====================================
	##============ the large N
	##====================================
	#large N setting:
	#train matrix shape: (47264, 1942)
	#ve_train: 847752631.846
	#test matrix shape: (15736, 1942)
	#ve_test: 281713301.983
	ve_train = 847752631.846
	ve_test = 281713301.983


	threshold = 500



	####
	list_error_train = np.load("./result_nn/list_error_train_largeN_nninit.npy")
	list_error_test = np.load("./result_nn/list_error_test_largeN_nninit.npy")

	#
	list_error_train = 1 - list_error_train / ve_train
	list_error_test = 1 - list_error_test / ve_test
	#
	plt.plot(list_error_train[:threshold], 'r-', label="nn init train")
	plt.plot(list_error_test[:threshold], 'r--', label="nn init test")
	##
	#plt.semilogx(list_error_train[:threshold], 'r-', label="nn init train", basex=2)
	#plt.semilogx(list_error_test[:threshold], 'r--', label="nn init test", basex=2)





	####
	list_error_train = np.load("./result_nn/list_error_train_largeN_nnrand.npy")
	list_error_test = np.load("./result_nn/list_error_test_largeN_nnrand.npy")

	#
	list_error_train = 1 - list_error_train / ve_train
	list_error_test = 1 - list_error_test / ve_test
	#
	plt.plot(list_error_train[:threshold], 'g-', label="nn rand train")
	plt.plot(list_error_test[:threshold], 'g--', label="nn rand test")
	##
	#plt.semilogx(list_error_train[:threshold], 'g-', label="nn rand train", basex=2)
	#plt.semilogx(list_error_test[:threshold], 'g--', label="nn rand test", basex=2)




	####
	list_error_train = np.load("./result_nn/list_error_train_largeN_mlinit.npy")
	list_error_test = np.load("./result_nn/list_error_test_largeN_mlinit.npy")

	#
	list_error_train = 1 - list_error_train / ve_train
	list_error_test = 1 - list_error_test / ve_test
	#
	plt.plot(list_error_train[:threshold], 'b-', label="ml init train")
	plt.plot(list_error_test[:threshold], 'b--', label="ml init test")
	##
	#plt.semilogx(list_error_train[:threshold], 'b-', label="ml init train", basex=2)
	#plt.semilogx(list_error_test[:threshold], 'b--', label="ml init test", basex=2)














	plt.xlabel("iterations")
	plt.ylabel("variance explained")
	plt.title("nn model: 10x sample size, 10th scale of features")
	plt.grid(True)
	plt.legend(loc=4)
	plt.axis([0, threshold, 0.3, 0.9])








	plt.show()














