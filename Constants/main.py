import tensorflow as tf

# scalar constant
t_1 = tf.constant(4)

# vector of shape [1, 3]
t_2 = tf.constant([4, 3, 2])

# tensor with all elements as zero of shape [1, 3] and type int32
tf.zeros([1, 3], tf.int32)

# tensor with all elements set to one
tf.ones([1, 3], tf.int32)

# create tensor of the same shape as an existing vector
tf.zeros_like(t_2)  # Create a zero matrix of same shape as t_2
tf.ones_like(t_2)  # Create as one matrix of shame shape as t_2
