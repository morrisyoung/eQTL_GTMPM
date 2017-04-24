import numpy as np
import tensorflow as tf
import timeit








## full GD for tensor predictive modeling

## iteratively optimizing

## shapes of parameters:
##	1. beta_snp: has intercept parameters, to count for the average individual effects
##	2. beta_gene: has intercept parameters, to explain effects beyond factors
##	3. beta_tissue: has no intercept parameters, so add the intercept (1)
## the individual factors should also get the intercept (1); we only allow one degree of freedom for these non-genetic factors to genes







##==================================================================================================================
## load ddata
header = "./data_real/"
X_train = np.load(header + "X_train.npy")
# add the intercept to X:
m_ones = np.ones((len(X_train), 1))
X_train = np.concatenate((X_train, m_ones), axis=1)									# N x (I+1)

Y_train = np.load(header + "Y_train.npy")
Y_train_spread = np.load(header + "Y_train_spread.npy")
Y_train_index = np.load(header + "Y_train_index.npy")


X_test = np.load(header + "X_test.npy")
# add the intercept to X:
m_ones = np.ones((len(X_test), 1))
X_test = np.concatenate((X_test, m_ones), axis=1)									# N x (I+1)

Y_test = np.load(header + "Y_test.npy")
Y_test_spread = np.load(header + "Y_test_spread.npy")
Y_test_index = np.load(header + "Y_test_index.npy")






##==================================================================================================================
## extract the cis- effects first of all
#
Y_cis_train = np.load(header + "Y_cis_train.npy")
Y_cis_train_spread = np.load(header + "Y_cis_train_spread.npy")
Y_train = Y_train - Y_cis_train
Y_train_spread = Y_train_spread - Y_cis_train_spread
#
Y_cis_test = np.load(header + "Y_cis_test.npy")
Y_cis_test_spread = np.load(header + "Y_cis_test_spread.npy")
Y_test = Y_test - Y_cis_test
Y_test_spread = Y_test_spread - Y_cis_test_spread







##==================================================================================================================
## load model
#header = "./tm_data_real_init/"
header = "../preprocess/tm_data_real_init/"
beta_snp_init = np.load(header + "Beta.npy")						## (num_of_snp+1) x (num_of_factor)

beta_gene_init = np.load(header + "fm_gene.npy")					## (num_of_gene) x (num_of_factor+1)
beta_tissue_init = np.load(header + "fm_tissue.npy")				## (num_of_tissue) x (num_of_factor)

















with tf.device("/cpu:0"):




	##==================================================================================================================
	## data and model
	##
	x = tf.placeholder(tf.float32)

	## beta_snp
	place_beta_snp = tf.placeholder(tf.float32, shape=beta_snp_init.shape)
	beta_snp = tf.Variable(place_beta_snp)

	##
	f = tf.matmul(x, beta_snp)
	## expand f
	f_intercept_shape = tf.placeholder(tf.int32)
	tensor_constant = tf.ones(dtype=tf.float32, shape=f_intercept_shape)
	f_ext = tf.concat([f, tensor_constant], 1)



	## beta_gene
	place_beta_gene = tf.placeholder(tf.float32, shape=beta_gene_init.shape)
	beta_gene = tf.Variable(place_beta_gene)



	## beta_tissue
	place_beta_tissue = tf.placeholder(tf.float32, shape=beta_tissue_init.shape)
	beta_tissue = tf.Variable(place_beta_tissue)
	## expand tissue factors (with constant --> only allowing one free intercept parameters, in gene fm)
	beta_tissue_intercept = tf.ones(dtype=tf.float32, shape=(len(beta_tissue_init), 1))
	beta_tissue_ext = tf.concat([beta_tissue, beta_tissue_intercept], 1)
	## for T, we have the extra dimension for broadcasting the multiply op following up
	beta_tissue_ext = tf.expand_dims(beta_tissue_ext, 1)			## now this is K x 1 x (factor+1)





	####################################
	#### the decomposed tensor product
	####################################
	## element-wise multiply for tissue and indiv factors
	# K x 1 x (feature_len+1), N x (feature_len+1) --> K x N x (feature_len+1)
	TUP = tf.multiply(beta_tissue_ext, f_ext)
	# K x N x (feature_len+1), J x (feature_len+1) --> K x N x J
	result = tf.einsum('knd,jd->knj', TUP, beta_gene)
	# or use tensordot for this

	##
	result_flatten = tf.reshape(result, [-1])
	y_index = tf.placeholder(tf.int32)
	y_ = tf.gather(result_flatten, y_index)

	# real Y
	y = tf.placeholder(tf.float32)








	##==================================================================================================================
	## cost function
	cost_base = tf.reduce_sum(tf.square(tf.subtract(y, y_)))


	## sparsity
	coeff_regularizer = tf.constant(.001)
	norm_sums = tf.add(tf.reduce_sum(tf.abs(beta_snp)),
						tf.reduce_sum(tf.abs(beta_gene)))
	norm_sums = tf.add(norm_sums,
						tf.reduce_sum(tf.abs(beta_tissue)))
	cost_regularizer = tf.multiply(coeff_regularizer, norm_sums)


	## total train cost
	cost_train = tf.add(cost_base, cost_regularizer)




	## real data:
	#lr = tf.constant(0.00000000001, name='learning_rate')					## works
	#lr = tf.constant(0.00000000005, name='learning_rate')					## good, but the testing set is not strictly decreasing
	
#	lr = tf.constant(0.00000000003, name='learning_rate')




	## tm:
	lr = tf.constant(0.00000000000000005, name='learning_rate')					## good for: LassoR0.005, LassoR0.001









	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.96, staircase=True)


	## learn!!!
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	#training_step1 = optimizer.minimize(cost_train, global_step=global_step)
	training_step1 = optimizer.minimize(cost_train, global_step=global_step, var_list=[beta_snp])
	training_step2 = optimizer.minimize(cost_train, global_step=global_step, var_list=[beta_gene])
	training_step3 = optimizer.minimize(cost_train, global_step=global_step, var_list=[beta_tissue])








	##==================================================================================================================
	# execute
	init = tf.global_variables_initializer()
	#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	sess = tf.Session()
	#sess.run(init)
	sess.run(init, feed_dict={place_beta_snp: beta_snp_init, place_beta_gene: beta_gene_init, place_beta_tissue: beta_tissue_init})







	list_error_train = []
	list_error_test = []
	#for i in xrange(1000):
	#for i in xrange(67):				## the rounds needed to converge
	for i in xrange(200):


		print "iter#", i



		##==== timer
		start_time = timeit.default_timer()


		if i == 0:
			## training error
			error = sess.run(cost_base, feed_dict={x: X_train, y_index: Y_train_index, y: Y_train_spread[Y_train_index], f_intercept_shape: [len(X_train), 1]})
			print "initial train error:", error
			error = sess.run(cost_base, feed_dict={x: X_test, y_index: Y_test_index, y: Y_test_spread[Y_test_index], f_intercept_shape: [len(X_test), 1]})
			print "initial test error:", error




		sess.run(training_step1, feed_dict={x: X_train, y_index: Y_train_index, y: Y_train_spread[Y_train_index], f_intercept_shape: [len(X_train), 1]})
		sess.run(training_step2, feed_dict={x: X_train, y_index: Y_train_index, y: Y_train_spread[Y_train_index], f_intercept_shape: [len(X_train), 1]})
		sess.run(training_step3, feed_dict={x: X_train, y_index: Y_train_index, y: Y_train_spread[Y_train_index], f_intercept_shape: [len(X_train), 1]})




		header = "./tm_result/"
		## training error
		error = sess.run(cost_base, feed_dict={x: X_train, y_index: Y_train_index, y: Y_train_spread[Y_train_index], f_intercept_shape: [len(X_train), 1]})
		print "train error:", error,
		list_error_train.append(error)
		np.save(header + "list_error_train", list_error_train)


		# testing error
		error = sess.run(cost_base, feed_dict={x: X_test, y_index: Y_test_index, y: Y_test_spread[Y_test_index], f_intercept_shape: [len(X_test), 1]})
		print "test error:", error
		list_error_test.append(error)
		np.save(header + "list_error_test", list_error_test)








		# ## save the learned para, for tissue and gene fm
		# ########################################################
		# ########################################################
		# fm_tissue = beta_tissue.eval(session=sess)
		# fm_gene = beta_gene.eval(session=sess)
		# np.save("./result_model/fm_tissue", fm_tissue)
		# np.save("./result_model/fm_gene", fm_gene)
		# ########################################################
		# ########################################################









		##==== timer
		elapsed = timeit.default_timer() - start_time
		print "time spent this iter:", elapsed


















