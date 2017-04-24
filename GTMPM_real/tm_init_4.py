import numpy as np
from sklearn.decomposition import PCA
import math
from sklearn import linear_model





## output:
##	Beta (from genetics to factors)

## for each factor, we use cis- SNPs of expressed genes in this factor as the input candidate SNPs, and we use LASSO

## I assume we have a subfolder under ~/GTEx_gtmnn/, and all the code and data will be cached there










if __name__ == "__main__":




	##=====================================================================================================================
	##==== load data
	##=====================================================================================================================
	#
	X = np.load("/ifs/scratch/c2b2/ip_lab/sy2515/GTEx_gtmnn/preprocess/data_train/X.npy")
	#
	F = np.load("/ifs/scratch/c2b2/ip_lab/sy2515/GTEx_gtmpm/preprocess/tm_data_temp/fm_indiv.npy")
	#
	print "X and F shapes:", X.shape, F.shape


	##==== fill dimension
	I = len(X[0])
	N = len(X)
	D = len(F[0])
	print "shape:"
	print "I:", I
	print "N:", N
	print "D:", D






	####========================================================================
	## for each factor: pick up candidate SNPs, do the LASSO
	####========================================================================
	m_indi_snp = np.load("/ifs/scratch/c2b2/ip_lab/sy2515/GTEx_gtmpm/preprocess/tm_data_temp/m_indi_snp.npy")
	beta = []
	for d in range(D):
		Data = np.multiply(X, m_indi_snp[d])			# X: (n_samples, n_features)
		Target = F[:, d] 								# Y: (n_samples, n_tasks)

		###################################
		#alpha = 0.0001

		#alpha = 0.0005

		#alpha = 0.001

		#alpha = 0.005



		## lasso fine tune
		#alpha = 0.006
		#alpha = 0.007
		#alpha = 0.008
		#alpha = 0.009
		alpha = 0.01










		###################################

		clf = linear_model.Lasso(alpha=alpha)			# TODO: parameter tunable
		clf.fit(Data, Target)

		#clf.coef_							# (n_features,)
		#clf.intercept_						# value
		temp = (clf.coef_).tolist()
		temp.append(clf.intercept_)

		beta.append(temp)
	init_beta = np.array(beta)				# (D, S+1)
	print "init_beta shape:",
	print init_beta.shape

	## test non-0 elements
	print "check num of non-0 snps each factor has (# of candidates, # of learned ones):"
	for d in range(D):
		print d, np.sum(m_indi_snp[d]), np.sum(np.sign(np.square(init_beta[d])))


	init_beta = init_beta.T 				# (S+1, D)
	np.save("./tm_data_real_init/Beta", init_beta)
















