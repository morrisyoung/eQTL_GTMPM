import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn import linear_model






##
## nn init: similar to the linear model, only one more logistic twist
##
## model only contains trans- factor effects
## init sparsely
##







init_beta1 = []
init_beta2 = []






if __name__ == "__main__":





	## load data
	X = np.load("./data_simu_ml_largeN/X_train.npy")
	Y = np.load("./data_simu_ml_largeN/Y_train.npy")

	##==== fill dimension
	I = len(X[0])
	K = len(Y)
	N = len(Y[0])
	J = len(Y[0][0])
	D = 40										## TODO: manually set this
	print "shape:"
	print "I:", I
	print "K:", K
	print "N:", N
	print "J:", J
	print "D:", D








	##=====================================================================================================================
	##==== cell factor (PCA + LASSO)
	##=====================================================================================================================
	##
	## init_beta_cellfactor2
	##
	####=============================== Scheme ===============================
	##	1. do PCA on sample matrix
	##	2. averaging the (Individual x Factor) matrix in order to eliminate the tissue effects, thus only individual effects left
	##	3. use these individual effects to retrieve their SNP causality
	##	4. use these individual effects to separately associate tissue effects of these factors
	##==== sample matrix
	Y_train = Y
	Y_train_matrix = []
	Y_train_matrix_pos = []
	for k in range(K):
		for n in range(N):
			if not math.isnan(Y_train[k][n][0]):
				Y_train_matrix.append(Y_train[k][n])
				Y_train_matrix_pos.append((k, n))
	Y_train_matrix = np.array(Y_train_matrix)
	Y_train_matrix_pos = np.array(Y_train_matrix_pos)
	print "sample matrix shape (train):", Y_train_matrix.shape
	print "sample matrix pos shape (train):", Y_train_matrix_pos.shape

	##==== do PCA for Sample x Gene, with number of factors as D
	n_factor = D
	pca = PCA(n_components=n_factor)
	pca.fit(Y_train_matrix)
	Y2 = (pca.components_).T 						# Gene x Factor
	Y1 = pca.transform(Y_train_matrix)				# Sample x Factor
	variance = pca.explained_variance_ratio_

	##==== individual factors
	m_factor = np.zeros((N, D))
	list_count = np.zeros(N)
	for i in range(len(Y1)):
		k, n = Y_train_matrix_pos[i]
		m_factor[n] += Y1[i]
		list_count[n] += 1
	for n in range(N):
		m_factor[n] = m_factor[n] / list_count[n]



	####========================================================================
	## twist factors into appropriate range
	## tune factor matrix into [0.1, 0.9]
	value_max = np.amax(m_factor)
	value_min = np.amin(m_factor)
	m_factor_tune = (m_factor - value_min) * (1 / (value_max - value_min))
	m_factor_tune = 0.5 + 0.8 * (m_factor_tune - 0.5)
	m_factor = m_factor_tune	



	####========================================================================
	## fill in the incomp tensor (with over-all mean) --> for group LASSO
	Y_train_mean = np.mean(Y_train_matrix, axis=0)
	for k in range(K):
		for n in range(N):
			if math.isnan(Y_train[k][n][0]):
				Y_train[k][n] = Y_train_mean


	####========================================================================
	## solve the group LASSO
	beta_tensor = []
	for j in range(J):
		Data = m_factor						# X: (n_samples, n_features)
		Target = Y_train[:, :, j].T 		# Y: (n_samples, n_tasks)


		##############################
		##############################
		alpha=10.0
		##############################
		##############################


		clf = linear_model.MultiTaskLasso(alpha=alpha)						## TODO: tune the strength
		clf.fit(Data, Target)

		#clf.coef_							# (n_tasks, n_features)
		#clf.intercept_						# (n_tasks,)
		intercept = (np.array([clf.intercept_])).T
		beta = np.concatenate((clf.coef_, intercept), axis=1)
		beta_tensor.append(beta)
	beta_tensor = np.array(beta_tensor)
	# beta_tensor: (J, K, (D+1))
	init_beta2 = np.transpose(beta_tensor, (1, 2, 0))
	print "init_beta2 shape (exp: K, D+1, J):",
	print init_beta2.shape


	##
	## test sparsity
	##
	matrix = init_beta2[0]
	matrix = np.square(matrix)
	matrix = np.sign(matrix)
	num = np.sum(matrix)
	print num
	print len(matrix) * len(matrix[0])











	##=====================================================================================================================
	##==== wide linear
	##=====================================================================================================================
	####========================================================================
	## need to first of all tune back the factors
	m_factor = np.log( m_factor / (1 - m_factor) )
	##
	## init_beta1
	##
	##==== linear system
	# the linear system between: X x Y1
	Data = X									# X: (n_samples, n_features)
	Target = m_factor 							# Y: (n_samples, n_tasks)
	clf = linear_model.Lasso(alpha=0.005)									## TODO: tune the strength
	clf.fit(Data, Target)
	#clf.coef_									# (n_tasks, n_features)
	#clf.intercept_								# (n_tasks,)
	intercept = (np.array([clf.intercept_])).T
	init_beta1 = np.concatenate((clf.coef_, intercept), axis=1)
	init_beta1 = init_beta1.T
	print "init_beta1 shape:",
	print init_beta1.shape




	## test sparsity
	matrix = init_beta1
	matrix = np.square(matrix)
	matrix = np.sign(matrix)
	num = np.sum(matrix)
	print num
	print len(matrix) * len(matrix[0])









	##=====================================================================================================================
	##==== save the init
	##=====================================================================================================================
	np.save("./data_simu_ml_largeN/beta1_init_nn", init_beta1)
	np.save("./data_simu_ml_largeN/beta2_init_nn", init_beta2)
















