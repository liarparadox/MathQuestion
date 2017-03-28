import tensorflow as tf

x = tf.get_variable('x', [], tf.int32, initializer=tf.constant_initializer(1))
r1 = x*x
g1 = tf.gradients(r1,x)
i, t = tf.constant(0), tf.constant(0)
c = lambda i, t: tf.less(i, x)
b = lambda i, t: [tf.add(i, 1), tf.add(t,x)]
_, r2 = tf.while_loop(c, b, [i,t])
g2 = tf.gradients(r2,x)

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  print sess.run([x,r1,r2,g1,g2])
