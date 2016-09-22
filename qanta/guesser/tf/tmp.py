import numpy as np
import tensorflow as tf

with tf.Graph().as_default(), tf.Session() as session:
    v1 = tf.Variable(np.array([1, 2, 3, 4, 5]))
    v2 = tf.Variable(np.array([1, 0, 3, 0, 5]))
    eq = tf.equal(v1, v2)
    session.run(tf.initialize_all_variables())
    print(session.run(tf.reduce_sum(eq)))
