import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines







map_color = {"0.6": 'k',
			"0.7": '#988ED5',
			"0.8": 'm',
			"0.9": '#8172B2',
			"1.0": '#348ABD',
			"1.1": '#fa8174',
			"1.2": '#FF9F9A',
			"1.3": '#56B4E9',
			"1.4": 'c',
			"1.5": '#6d904f',
			"1.6": 'cyan',
			"1.7": 'red',
			"1.8": 'darkgoldenrod',
			"1.9": 'yellow',
			"2.0": '#6ACC65',
			"2.1": 'gray',
			"2.2": '#F0E442',
			"2.3": '#017517',
			"2.4": '#B0E0E6',
			"2.5": 'magenta'}




if __name__=="__main__":




	ve_train = 82944750.0
	ve_test = 26776502.0985



	threshold = 200






	fig = plt.figure(1)
	plt.subplot(131)
	##============================================================================================================
	##
	model = "ml"
	R = "0.6"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	color = map_color[R]
	plt.plot(list_error_test[:], color=color, ls='--', label="R=" + R)

	##
	model = "ml"
	R = "0.7"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	color = map_color[R]
	plt.plot(list_error_test[:], color=color, ls='--', label="R=" + R)

	##
	model = "ml"
	R = "0.8"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	color = map_color[R]
	plt.plot(list_error_test[:], color=color, ls='--', label="R=" + R)

	##
	model = "ml"
	R = "0.9"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	color = map_color[R]
	plt.plot(list_error_test[:], color=color, ls='--', label="R=" + R)

	##
	model = "ml"
	R = "1.0"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	color = map_color[R]
	plt.plot(list_error_test[:], color=color, ls='--', label="R=" + R)








	#### more
	##
	model = "ml"
	R = "1.1"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	color = map_color[R]
	plt.plot(list_error_test[:], color=color, ls='--', label="R=" + R)
	##
	model = "ml"
	R = "1.2"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	color = map_color[R]
	plt.plot(list_error_test[:], color=color, ls='--', label="R=" + R)
	##
	model = "ml"
	R = "1.3"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	color = map_color[R]
	plt.plot(list_error_test[:], color=color, ls='--', label="R=" + R)
	##
	model = "ml"
	R = "1.4"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	color = map_color[R]
	plt.plot(list_error_test[:], color=color, ls='--', label="R=" + R)
	##
	model = "ml"
	R = "1.5"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	color = map_color[R]
	plt.plot(list_error_test[:], color=color, ls='--', label="R=" + R)






	#### more
	##
	model = "ml"
	R = "1.6"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	color = map_color[R]
	plt.plot(list_error_test[:], color=color, ls='--', label="R=" + R)
	##
	model = "ml"
	R = "1.7"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	color = map_color[R]
	plt.plot(list_error_test[:], color=color, ls='--', label="R=" + R)
	##
	model = "ml"
	R = "1.8"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	color = map_color[R]
	plt.plot(list_error_test[:], color=color, ls='--', label="R=" + R)
	##
	model = "ml"
	R = "1.9"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	color = map_color[R]
	plt.plot(list_error_test[:], color=color, ls='--', label="R=" + R)
	##
	model = "ml"
	R = "2.0"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	color = map_color[R]
	plt.plot(list_error_test[:], color=color, ls='--', label="R=" + R)







	#### more
	##
	model = "ml"
	R = "2.1"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	color = map_color[R]
	plt.plot(list_error_test[:], color=color, ls='--', label="R=" + R)
	##
	model = "ml"
	R = "2.2"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	color = map_color[R]
	plt.plot(list_error_test[:], color=color, ls='--', label="R=" + R)
	##
	model = "ml"
	R = "2.3"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	color = map_color[R]
	plt.plot(list_error_test[:], color=color, ls='--', label="R=" + R)
	##
	model = "ml"
	R = "2.4"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	color = map_color[R]
	plt.plot(list_error_test[:], color=color, ls='--', label="R=" + R)
	##
	model = "ml"
	R = "2.5"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	color = map_color[R]
	plt.plot(list_error_test[:], color=color, ls='--', label="R=" + R)









	##
	plt.xlabel("iterations")
	plt.ylabel("variance explained")
	plt.grid(True)

	#plt.axis([0, threshold, 0.550, 0.6])

	#plt.axis([0, threshold, 0.585, 0.595])				# ml zoom in
	#plt.axis([0, threshold, 0.35, 0.72])
	#plt.axis([0, threshold, 0.0, 1.0])
	#plt.title("various penalty strength for wide linear model (of genetics to factors) Lasso initialization, for ml model and nn model")
	#plt.title("multiple linear and neural net, different init Lasso (genetics to factors)")
	plt.title("multiple linear modeling (ml)")
	plt.legend(loc=4, ncol=2)
	#plt.show()












	plt.subplot(132)
	##============================================================================================================
	## nn model
	##
	model = "nn"
	R = "0.02"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	plt.plot(list_error_test[:], 'r--', label="R=" + R)
	##
	model = "nn"
	R = "0.03"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	plt.plot(list_error_test[:], 'y--', label="R=" + R)
	##
	model = "nn"
	R = "0.04"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	plt.plot(list_error_test[:], 'g--', label="R=" + R)
	##
	model = "nn"
	R = "0.05"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	plt.plot(list_error_test[:], 'c--', label="R=" + R)
	##
	model = "nn"
	R = "0.06"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	plt.plot(list_error_test[:], 'm--', label="R=" + R)









	##
	plt.xlabel("iterations")
	plt.ylabel("variance explained")
	plt.grid(True)

	#plt.axis([0, threshold, 0.550, 0.6])

	#plt.axis([0, threshold, 0.35, 0.72])
	#plt.axis([0, threshold, 0.0, 1.0])
	#plt.title("various penalty strength for wide linear model (of genetics to factors) Lasso initialization, for ml model and nn model")
	#plt.title("multiple linear and neural net, different init Lasso (genetics to factors)")
	plt.title("neural network modeling (nn)")
	plt.legend(loc=4)
	#plt.show()










	plt.subplot(133)
	##============================================================================================================
	## tm model
	##
	model = "tm"
	R = "0.006"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	plt.plot(list_error_test[:], 'r--', label="R=" + R)
	##
	model = "tm"
	R = "0.007"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	plt.plot(list_error_test[:], 'y--', label="R=" + R)
	##
	model = "tm"
	R = "0.008"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	plt.plot(list_error_test[:], 'g--', label="R=" + R)
	##
	model = "tm"
	R = "0.009"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	plt.plot(list_error_test[:], 'c--', label="R=" + R)
	##
	model = "tm"
	R = "0.01"
	list_error_test = np.load("./result_lasso_finetune/list_error_test_" + model + "_R" + R + ".npy")
	list_error_test = 1 - list_error_test / ve_test
	plt.plot(list_error_test[:], 'm--', label="R=" + R)










	##
	plt.xlabel("iterations")
	plt.ylabel("variance explained")
	plt.grid(True)

	#plt.axis([0, threshold, 0.550, 0.6])

	#plt.axis([0, threshold, 0.585, 0.595])				# ml zoom in
	#plt.axis([0, threshold, 0.35, 0.72])
	#plt.axis([0, threshold, 0.0, 1.0])
	#plt.title("various penalty strength for wide linear model (of genetics to factors) Lasso initialization, for ml model and nn model")
	#plt.title("multiple linear and neural net, different init Lasso (genetics to factors)")
	plt.title("tensor predictive modeling (tm)")
	plt.legend(loc=4)

















	#plt.suptitle("models with different penalty for initialization Lasso (of genetics to factors), test set performance")
	plt.suptitle("models' initialization Lasso (of genetics to factors) with different penalty (till no SNPs picked up), test set performance")
	plt.show()















