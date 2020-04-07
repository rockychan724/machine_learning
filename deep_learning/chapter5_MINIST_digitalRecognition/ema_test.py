import tensorflow as tf

# v = tf.Variable(0, dtype=tf.float32)
# for variable in tf.global_variables():
#     print(variable.name)
#
# ema = tf.train.ExponentialMovingAverage(0.99)
# moving_average_update_op = ema.apply(tf.global_variables())
#
# for variable in tf.global_variables():
#     print(variable.name)
#
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#
#     sess.run(tf.assign(v, 10))
#     sess.run(moving_average_update_op)
#     print(sess.run([v, ema.average(v)]))
#     saver.save(sess, './model_ema/model.ckpt')

v1 = tf.Variable(0, dtype=tf.float32)
v2 = tf.Variable(0, dtype=tf.float32)
ema = tf.train.ExponentialMovingAverage(0.99)
# saver = tf.train.Saver()
saver = tf.train.Saver({'Variable': v1, 'Variable/ExponentialMovingAverage': v2})  # v/ExponentialMovingAverage
print(ema.variables_to_restore())
# saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess, './model_ema/model.ckpt')
    print(v1, sess.run(v1))
    print(v2, sess.run(v2))
