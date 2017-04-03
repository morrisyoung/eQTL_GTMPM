## this is the tensor init series, which involves three parts (three scripts):
##	1. PCA on (sample x gene) matrix, to get gene factor matrix and the smaller tensor of (indiv x tissue x factor)
##	2. do the incomplete PCA with R, and get individual factor matrix and tissue factor matrix
##	3. re-solve the linear system with LASSO of (sample x gene) and (sample x factor) [from indiv and tissue fm] to get sparse gene fm


## this script takes the preprocessed and normalized tensor as input
## this script will get accompany with incomplete PCA R script (for indiv and tissue fm), and LASSO script (for gene fm)
## this script will be followed by Beta init (wide linear with Lasso)


## output:
##	T, U, V






##===============
##==== libraries
##===============
import numpy as np
import math
import timeit
from sklearn.decomposition import PCA












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
	## NOTE: take the residuals (regressing out the cis- effects)
	Y_cis_train = np.load("/ifs/scratch/c2b2/ip_lab/sy2515/GTEx_gtmnn/workbench54/data_real_init/Y_cis_train.npy")
	Y = Y - Y_cis_train
	##
	##

	##==== fill dimension
	J = len(Y[0][0])
	K = len(Y)
	N = len(X)
	D = 400
	print "shape (train):"
	print "J:", J
	print "K:", K
	print "N:", N
	print "D:", D



	##==== reshape to sample matrix
	#Y_train = Y
	# Y_train is ready
	Data = []
	Data_index = []
	for k in range(K):
		for n in range(len(Y[k])):
			#
			Data.append(Y[k][n])
			#
			pos = Y_pos[k][n]
			Data_index.append((k, pos))
	Data = np.array(Data)
	Data_index = np.array(Data_index)
	#
	print "sample matrix shape (train):", Data.shape
	print "sample matrix pos shape (train):", Data_index.shape











	##=============
	##==== do PCA for Sample x Gene matrix
	##=============
	print "performing PCA..."
	n_factor = D
	pca = PCA(n_components=n_factor)
	pca.fit(Data)
	Y2 = (pca.components_).T
	Y1 = pca.transform(Data)
	variance = pca.explained_variance_ratio_

	print variance
	print "and the cumulative variance are:"
	for i in range(len(variance)):
		print i,
		print np.sum(variance[:i+1]),
	print ""

	print "sample factor matrix:", Y1.shape
	print "gene factor matrix:", Y2.shape
	np.save("./tm_data_temp/fm_gene_initial", Y2)				## not useful actually











	##=============
	##==== save the Individual x Tissue matrix (with Nan in) under "tm_data_temp"
	##=============
	##
	## TO use: Y1, Data_index
	##
	Data = np.zeros((K, N, D)) + float("Nan")
	for i in range(len(Data_index)):
		(tissue, pos) = Data_index[i]
		Data[tissue][pos] = Y1[i]
	print "the Tissue x Individual x Factor tensor has the dimension:",
	print Data.shape


	for d in range(D):
		m_factor = Data[:, :, d]
		np.save("./tm_data_temp/f" + str(d) + "_tissue_indiv", m_factor)
	print "per-factor saving done..."


	##== need to save the results in tsv file (including Nan), in order to load in R
	for d in range(D):
		m_factor = np.load("./tm_data_temp/f" + str(d) + "_tissue_indiv.npy")
		file = open("./tm_data_temp/f" + str(d) + "_tissue_indiv.txt", 'w')
		for i in range(len(m_factor)):
			for j in range(len(m_factor[i])):
				value = m_factor[i][j]
				file.write(str(value))
				if j != len(m_factor[i])-1:
					file.write('\t')
			file.write('\n')
		file.close()















