import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from numpy.linalg import inv
import os
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import matplotlib.lines as mlines
from sklearn import manifold, datasets









## FM, hierarchical clustering and MDS of tissue fm







list_chr_color = ['k', '#988ED5', 'm', '#8172B2', '#348ABD', '#fa8174', '#FF9F9A', '#56B4E9', 'w', '#6d904f', 'cyan', 'red', 'darkgoldenrod', 'yellow', '#6ACC65', 'gray', '#F0E442', '#017517', '#B0E0E6', 'magenta', '#b3de69', '0.70', 'c', '#C4AD66', '#EAEAF2', '#A60628', '#CC79A7', '#7600A1']
list_tissues = ["Thyroid", "Testis", "Skin - Not Sun Exposed (Suprapubic)", "Esophagus - Muscularis", "Heart - Atrial Appendage", "Breast - Mammary Tissue", "Brain - Cerebellum", "Esophagus - Mucosa", "Artery - Coronary", "Esophagus - Gastroesophageal Junction", "Artery - Aorta", "Pancreas", "Adipose - Subcutaneous", "Skin - Sun Exposed (Lower leg)", "Whole Blood", "Muscle - Skeletal", "Brain - Caudate (basal ganglia)", "Heart - Left Ventricle", "Colon - Transverse", "Stomach", "Adipose - Visceral (Omentum)", "Adrenal Gland", "Lung", "Cells - Transformed fibroblasts", "Artery - Tibial", "Colon - Sigmoid", "Nerve - Tibial", "Cells - EBV-transformed lymphocytes"]











if __name__ == "__main__":








	##====================================================================================================================
	## factor matrix
	##====================================================================================================================
	"""
	import seaborn as sns
	fm_loading = np.load("./result_model/fm_tissue.npy")


	threshold_factor = 50											## TODO
	fm_loading = fm_loading[:, :threshold_factor]


	## the following list is learned from code below
	list_index = [27, 1, 6, 16, 21, 14, 11, 18, 5, 23, 15, 4, 17, 19, 7, 2, 13, 0, 22, 26, 25, 3, 9, 24, 8, 10, 12, 20]
	fm_loading = fm_loading[list_index]
	list_tissues = np.array(list_tissues)
	list_tissues = list_tissues[list_index]


	## TEST range
	## seems [-1500, 1500] is good enough
	#print np.sort(np.reshape(fm_loading, -1))[:10]
	#print np.sort(np.reshape(fm_loading, -1))[-10:-1]
	## seems [-1200, 1200] is good enough
	sns.set(context="paper", font="monospace")
	f, ax = plt.subplots(figsize=(22, 19))							# TODO
	#sns_plot = sns.heatmap(fm_loading, xticklabels=np.arange(threshold_factor), yticklabels=list_tissues, vmin=-1500, vmax=1500)
	sns_plot = sns.heatmap(fm_loading, xticklabels=np.arange(threshold_factor), yticklabels=list_tissues, vmin=-1200, vmax=1200)
	##
	#sns_plot = sns.heatmap(fm_loading, xticklabels=np.arange(threshold_factor), vmin=-1200, vmax=1200)
	#plt.yticks(np.arange(0, 10)+0.5, list_tissues[:10], rotation=0)
	#plt.yticks(np.arange(10, len(list_tissues))+0.5, list_tissues[10:], rotation=0)



	########
	pos1, pos2, pos3, pos4, pos5, pos6, pos7 = 1, 10, 13, 17, 20, 23, 26
	color1, color2, color3, color4, color5, color6, color7, color8 = 'b', 'g', 'r', 'y', 'm', 'cyan', '#348ABD', '#6ACC65'
	########



	count = len(ax.get_yticklabels())
	for j in range(len(ax.get_yticklabels())):
		index = count - j - 1
		ytick = ax.get_yticklabels()[index]
		if j < pos1:
			ytick.set_color(color1)
		elif j < pos2:
			ytick.set_color(color2)
		elif j < pos3:
			ytick.set_color(color3)
		elif j < pos4:
			ytick.set_color(color4)
		elif j < pos5:
			ytick.set_color(color5)
		elif j < pos6:
			ytick.set_color(color6)
		elif j < pos7:
			ytick.set_color(color7)
		else:
			ytick.set_color(color8)



	#sns_plot = sns.heatmap(fm_loading)
	#sns_plot = sns.heatmap(fm_loading, yticklabels=y_label)
	ax.set_xlabel('Factors')
	ax.set_ylabel('Tissues')
	#plt.yticks(rotation=0)
	plt.show()
	#fig = sns_plot.get_figure()
	#fig.savefig("plot/quantile_c22_gene.jpg")
	#fig.savefig("/Users/shuoyang/Desktop/fm_gene.jpg")
	#fig.savefig("/Users/shuoyang/Desktop/fm_heatmap.jpg")
	"""














	##====================================================================================================================
	## hierarchical clustering
	##====================================================================================================================
	fm_loading = np.load("./result_model/fm_tissue.npy")
	#threshold_factor = 50
	#fm_loading = fm_loading[:, :threshold_factor]
	X = fm_loading
	print X.shape


	## NOTE: the clustering method
	# generate the linkage matrix
	Z = linkage(X, 'weighted')
	print Z.shape


	# calculate full dendrogram
	#plt.figure()
	fig = plt.figure(figsize=(15, 15))
	ax = plt.subplot()

	plt.title('hierarchical clustering of tissue activations for different factors')
	plt.xlabel('tissues')
	plt.ylabel('distance (Euclidean)')
	d = dendrogram(
	    Z,
	    leaf_rotation=90.,  # rotates the x axis labels
	    leaf_font_size=12.,  # font size for the x axis labels
	    labels = list_tissues,
		color_threshold=5000,
	)
	print d['leaves']
	print d['color_list']


	########
	pos1, pos2, pos3, pos4, pos5, pos6, pos7 = 1, 10, 13, 17, 20, 23, 26
	color1, color2, color3, color4, color5, color6, color7, color8 = 'b', 'g', 'r', 'y', 'm', 'cyan', '#348ABD', '#6ACC65'
	########


	for i in range(len(ax.get_xticklabels())):
		xtick = ax.get_xticklabels()[i]
		if i < pos1:
			xtick.set_color(color1)
		elif i < pos2:
			xtick.set_color(color2)
		elif i < pos3:
			xtick.set_color(color3)
		elif i < pos4:
			xtick.set_color(color4)
		elif i < pos5:
			xtick.set_color(color5)
		elif i < pos6:
			xtick.set_color(color6)
		elif i < pos7:
			xtick.set_color(color7)
		else:
			xtick.set_color(color8)


	plt.show()
	#plt.savefig("/Users/shuoyang/Desktop/d" + str(d) + ".png")
	#plt.close(fig)














	##====================================================================================================================
	## MDS
	##====================================================================================================================
	"""
	# X = np.load("./result_model/fm_tissue.npy")
	# print X.shape
	# n_components = 2
	# mds = manifold.MDS(n_components)
	# Y = mds.fit_transform(X)
	# print Y.shape
	# np.save("./result_model/Y_MDS_28k", Y)



	##
	Y = np.load("./result_model/Y_MDS_28k.npy")


	########
	## the following list is learned from code above
	list_index = [27, 1, 6, 16, 21, 14, 11, 18, 5, 23, 15, 4, 17, 19, 7, 2, 13, 0, 22, 26, 25, 3, 9, 24, 8, 10, 12, 20]
	Y = Y[list_index]
	list_tissues = np.array(list_tissues)
	list_tissues = list_tissues[list_index]
	########


	########
	list_marker = ['o', 'v', '^', 's', '*', '+', 'x', 'D', 'p']
	pos1, pos2, pos3, pos4, pos5, pos6, pos7 = 1, 10, 13, 17, 20, 23, 26
	color1, color2, color3, color4, color5, color6, color7, color8 = 'b', 'g', 'r', 'y', 'm', 'cyan', '#348ABD', '#6ACC65'
	########


	list_handles = []
	for k in range(len(Y)):
		##
		#color = list_chr_color[k]
		## new color scheme
		if k < pos1:
			color = color1
			marker = list_marker[(k - 0) % len(list_marker)]
		elif k < pos2:
			color = color2
			marker = list_marker[(k - pos1) % len(list_marker)]
		elif k < pos3:
			color = color3
			marker = list_marker[(k - pos2) % len(list_marker)]
		elif k < pos4:
			color = color4
			marker = list_marker[(k - pos3) % len(list_marker)]
		elif k < pos5:
			color = color5
			marker = list_marker[(k - pos4) % len(list_marker)]
		elif k < pos6:
			color = color6
			marker = list_marker[(k - pos5) % len(list_marker)]
		elif k < pos7:
			color = color7
			marker = list_marker[(k - pos6) % len(list_marker)]
		else:
			color = color8
			marker = list_marker[(k - pos7) % len(list_marker)]



		plt.plot(Y[k, 0], Y[k, 1], marker = marker, color = color, markersize = 10)

		##
		tissue = list_tissues[k]
		line = mlines.Line2D([], [], marker=marker, markersize = 10, color=color, label=tissue, linestyle = 'None')
		list_handles.append(line)

	plt.legend(handles=list_handles, ncol=1, loc = 2, fancybox=True, numpoints=1)
	plt.xlabel('coordinate 1')
	plt.ylabel('coordinate 2')
	#plt.axis([-1000, 4000, -1000, 800])
	#plt.axis([-600, 1400, -800, 600])
	plt.axis([-9000, 3000, -2000, 5000])

	plt.title('MDS for 28 tissue parameters')
	plt.show()
	"""

















