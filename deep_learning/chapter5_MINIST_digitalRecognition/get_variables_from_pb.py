# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import tensorflow as tf
import os

model_dir = './model_save_part_data/'
model_name = 'test.pb'


# 读取并创建一个图graph来存放训练好的 Inception_v3模型（函数）
def create_graph():
    with tf.gfile.FastGFile(os.path.join(
            model_dir, model_name), 'rb') as f:
        # 使用tf.GraphDef()定义一个空的Graph
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # Imports the graph from graph_def into the current default Graph.
        tf.import_graph_def(graph_def, name='')
        writer = tf.summary.FileWriter('./log1/', tf.get_default_graph())
        writer.close()


# 创建graph
create_graph()

tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
for tensor_name in tensor_name_list:
    print(tensor_name)
