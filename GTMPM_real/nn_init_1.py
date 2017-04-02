import numpy as np
from sklearn.decomposition import PCA
import math
from sklearn import linear_model











if __name__ == "__main__":



	##=====================================================================================================================
	##==== load data (train)
	##=====================================================================================================================
	header = "/ifs/scratch/c2b2/ip_lab/sy2515/GTEx_gtmnn/preprocess/data_train/"

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
	##
	## NOTE: take the residuals
	Y_cis_train = np.load("/ifs/scratch/c2b2/ip_lab/sy2515/GTEx_gtmnn/workbench54/data_real_init/Y_cis_train.npy")
	Y = Y - Y_cis_train
	##
	##

	##==== fill dimension
	I = len(X[0])
	J = len(Y[0][0])
	K = len(Y)
	N = len(X)
	D = 400
	print "shape (train):"
	print "I:", I
	print "J:", J
	print "K:", K
	print "N:", N
	print "D:", D

	## reformat to complete tensor (with Nan)
	Y_train = np.zeros((K, N, J)) + float("Nan")
	for k in range(K):
		for i in range(len(Y_pos[k])):
			indiv = Y_pos[k][i]
			exp = Y[k][i]
			Y_train[k][indiv] = exp








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
	#Y_train = Y
	# Y_train is ready
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
	print variance

	##==== individual factors
	m_factor = np.zeros((N, D))
	list_count = np.zeros(N)
	for i in range(len(Y1)):
		k, n = Y_train_matrix_pos[i]
		m_factor[n] += Y1[i]
		list_count[n] += 1
	for n in range(N):
		m_factor[n] = m_factor[n] / list_count[n]




	##
	## --> non-linearity for the neural network model
	##
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
		#alpha=0.01							## looks fine, still a little too strong
		alpha=0.005							## fine, but a little weak

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
	print "init_beta2 shape (exp: K, D+1, J):", init_beta2.shape
	np.save("./nn_data_real_init/beta2_init", init_beta2)






	## test sparsity
	matrix = init_beta2[0]
	matrix = np.square(matrix)
	matrix = np.sign(matrix)
	num = np.sum(matrix)
	print num, "v.s.", len(matrix) * len(matrix[0])








	####========================================================================
	##
	## -> twist the m_factor back through the logistic func, for nn model
	##
	## need to first of all tune back the factors
	m_factor = np.log( m_factor / (1 - m_factor) )

	## m_factor to be used
	np.save("./nn_data_temp/m_factor", m_factor)

	## save non-0 genes for each factor
	print "non-zero entries for each factor (all tissues):"
	m_indi = []
	for d in range(D):
		m_factor = init_beta2[:, d, :]
		m_factor = np.square(m_factor)
		a_factor = np.sum(m_factor, axis=0)
		indi_factor = np.sign(a_factor)
		m_indi.append(indi_factor)
	m_indi = np.array(m_indi)
	print "m_indi shape:", m_indi.shape
	np.save("./nn_data_temp/m_indi", m_indi)

	## check and save effective SNPs for each factor
	print "output num of active genes and num of corresponding snps in each factor:"
	mapping_cis = np.load("/ifs/scratch/c2b2/ip_lab/sy2515/GTEx_gtmnn/preprocess/data_train/mapping_cis.npy")
	m_indi_snp = []
	for d in range(D):
		indi_snp = np.zeros(I)

		indi_factor = m_indi[d]
		for j in range(J):
			if indi_factor[j] == 1:				# gene in this factor
				start = mapping_cis[j][0]
				end = mapping_cis[j][1]
				for index in range(start, end+1):
					indi_snp[index] = 1
		print d, np.sum(indi_factor), np.sum(indi_snp)
		m_indi_snp.append(indi_snp)
	m_indi_snp = np.array(m_indi_snp)
	print "m_indi_snp shape:", m_indi_snp.shape
	np.save("./nn_data_temp/m_indi_snp", m_indi_snp)



















