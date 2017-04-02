import numpy as np
from sklearn.decomposition import PCA
import math
from sklearn import linear_model












if __name__ == "__main__":







	##=====================================================================================================================
	##==== load data
	##=====================================================================================================================
	header = "/ifs/scratch/c2b2/ip_lab/sy2515/GTEx_gtmnn/preprocess/data_train/"

	#
	X = np.load(header + "X.npy")
	#
	F = np.load("/ifs/scratch/c2b2/ip_lab/sy2515/GTEx_gtmpm/preprocess/ml_data_temp/m_factor.npy")


	##==== fill dimension
	I = len(X[0])
	N = len(X)
	D = len(F[0])
	print "shape:"
	print "I:", I
	print "N:", N
	print "D:", D


	##
	## for each factor: pick up candidate SNPs, do the LASSO
	##
	####========================================================================
	m_indi_snp = np.load("/ifs/scratch/c2b2/ip_lab/sy2515/GTEx_gtmpm/preprocess/ml_data_temp/m_indi_snp.npy")
	beta = []
	for d in range(D):
		Data = np.multiply(X, m_indi_snp[d])			# X: (n_samples, n_features)
		Target = F[:, d] 								# Y: (n_samples, n_tasks)


		##############################
		##############################
		#alpha=0.1										## this is used before, and this is good

		#alpha=0.5
		alpha=0.05


		##############################
		##############################


		clf = linear_model.Lasso(alpha=alpha)
		clf.fit(Data, Target)

		#clf.coef_							# (n_features,)
		#clf.intercept_						# value
		temp = (clf.coef_).tolist()
		temp.append(clf.intercept_)

		beta.append(temp)
	init_beta1 = np.array(beta)
	print "init_beta1 shape:",
	print init_beta1.shape

	## test non-0 elements
	print "check num of non-0 snps each factor has:"
	for d in range(D):
		print d, np.sum(m_indi_snp[d]), np.sum(np.sign(np.square(init_beta1[d])))

	#
	init_beta1 = init_beta1.T
	np.save("./ml_data_real_init/beta1_init", init_beta1)















