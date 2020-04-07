# coding: utf-8

"""
# 方式一：保存计算图中所有节点
import tensorflow as tf

# 保存模型
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
result = v1 + v2
saver = tf.train.Saver()
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess, './model/model.ckpt')
    
# 加载模型
saver = tf.train.import_meta_graph('./model/model.ckpt.meta')
with tf.Session() as sess:
    saver.restore(sess, './model/model.ckpt')
    print(sess.run(tf.get_default_graph().get_tensor_by_name('add:0')))
"""



# 方式二：保存计算图中部分节点
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

# 保存模型
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
result = v1 + v2
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    # 导出当前计算图的GraphDef部分
    graph_def = tf.get_default_graph().as_graph_def()
    # 导出指定节点
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
    # 将导出的模型存入文件
    with tf.gfile.GFile('./model_save_part_data/test.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    writer = tf.summary.FileWriter('./log/', tf.get_default_graph())
    writer.close()

# 加载模型
with tf.Session() as sess:
    with gfile.FastGFile('./model_save_part_data/combined_model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    result = tf.import_graph_def(graph_def, return_elements=['add:0'])
    print(sess.run(result))


# # import tensorflow as tf
# #
# # reader = tf.train.NewCheckpointReader('./model/model.ckpt')
# # global_variables = reader.get_variable_to_shape_map()
# # for variable_name in global_variables:
# #     print(variable_name, global_variables[variable_name])
# # print('Value of v2 is ', reader.get_tensor('v2'))
#
# import tensorflow as tf
#
# v3 = tf.Variable(tf.constant(1.0, shape=[1]), name='v3')
# v4 = tf.Variable(tf.constant(2.0, shape=[1]), name='v4')
# result1 = v3 + v4
# saver = tf.train.Saver({'v1': v3, 'v2': v4})
# with tf.Session() as sess:
#     saver.restore(sess, './model/model.ckpt')
#     # print(sess.run(tf.get_default_graph().get_tensor_by_name('v3:0')))
#     # print(sess.run(tf.get_default_graph().get_tensor_by_name('v4:0')))
#     # print(sess.run(tf.get_default_graph().get_tensor_by_name('add:0')))
#     print(sess.run(tf.global_variables()))
#     print(sess.run(v3))
#     print(sess.run(v4))
#     print(sess.run(result1))
