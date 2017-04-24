import numpy as np
import tensorflow as tf
import timeit











##==================================================================================================================
## load ddata
header = "./data_real/"
#
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
#header = "./nn_data_real_init/"
header = "../preprocess/nn_data_real_init/"
beta1_init = np.load(header + "beta1_init.npy")
beta2_init = np.load(header + "beta2_init.npy")

















with tf.device("/cpu:0"):




	##==================================================================================================================
	## data and model
	x = tf.placeholder(tf.float32)

	# beta1
	place_beta1 = tf.placeholder(tf.float32, shape=beta1_init.shape)
	beta1 = tf.Variable(place_beta1)
	## beta2
	place_beta2 = tf.placeholder(tf.float32, shape=beta2_init.shape)
	beta2 = tf.Variable(place_beta2)



	##
	f = tf.matmul(x, beta1)
	## neural network (with sigmoid func as the activation)
	f = tf.sigmoid(f)

	## expand f
	f_intercept_shape = tf.placeholder(tf.int32)
	tensor_constant = tf.ones(dtype=tf.float32, shape=f_intercept_shape)
	f_ext = tf.concat([f, tensor_constant], 1)


	## tensordot: (indiv, factor+1) x (tissue, factor+1, gene) = (indiv, tissue, gene)
	result_exp = tf.tensordot(f_ext, beta2, [[1], [1]])
	result_exp_reshape = tf.transpose(result_exp, perm=[1, 0, 2])
	result_exp_flatten = tf.reshape(result_exp_reshape, [-1])
	y_index = tf.placeholder(tf.int32)
	y_ = tf.gather(result_exp_flatten, y_index)

	# real Y
	y = tf.placeholder(tf.float32)








	##==================================================================================================================
	## cost function
	cost_base = tf.reduce_sum(tf.square(tf.subtract(y, y_)))


	## sparsity
	coeff_regularizer = tf.constant(.001)
	norm_sums = tf.add(tf.reduce_sum(tf.abs(beta1)),
						tf.reduce_sum(tf.abs(beta2)))
	cost_regularizer = tf.multiply(coeff_regularizer, norm_sums)


	## total train cost
	cost_train = tf.add(cost_base, cost_regularizer)


	##
	lr = tf.constant(0.00000000000002, name='learning_rate')						## good






	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.96, staircase=True)


	## learn!!!
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	#training_step = optimizer.minimize(cost_train, global_step=global_step)
	training_step1 = optimizer.minimize(cost_train, global_step=global_step, var_list=[beta1])
	training_step2 = optimizer.minimize(cost_train, global_step=global_step, var_list=[beta2])








	##==================================================================================================================
	# execute
	#init = tf.initialize_all_variables()
	init = tf.global_variables_initializer()
	#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	sess = tf.Session()
	#sess.run(init)
	sess.run(init, feed_dict={place_beta1: beta1_init, place_beta2: beta2_init})




	list_error_train = []
	list_error_test = []
	for i in xrange(200):
		print "iter#", i


		## initial training error
		if i == 0:
			error = sess.run(cost_base, feed_dict={x: X_train, y_index: Y_train_index, y: Y_train_spread[Y_train_index], f_intercept_shape: [len(X_train), 1]})
			print "initial train error:", error
			error = sess.run(cost_base, feed_dict={x: X_test, y_index: Y_test_index, y: Y_test_spread[Y_test_index], f_intercept_shape: [len(X_test), 1]})
			print "initial test error:", error



		#sess.run(training_step, feed_dict={x: X_train, y_index: Y_train_index, y: Y_train_spread[Y_train_index], f_intercept_shape: [len(X_train), 1]})
		sess.run(training_step1, feed_dict={x: X_train, y_index: Y_train_index, y: Y_train_spread[Y_train_index], f_intercept_shape: [len(X_train), 1]})
		sess.run(training_step2, feed_dict={x: X_train, y_index: Y_train_index, y: Y_train_spread[Y_train_index], f_intercept_shape: [len(X_train), 1]})






		header = "./nn_result/"
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







	########################################################
	result_beta1 = beta1.eval(session=sess)
	result_beta2 = beta2.eval(session=sess)
	np.save("./nn_result/result_beta1", result_beta1)
	np.save("./nn_result/result_beta2", result_beta2)
	print "beta1 and beta2 saving done..."
	########################################################















