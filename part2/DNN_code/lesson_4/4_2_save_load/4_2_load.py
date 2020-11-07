import numpy as np
import tensorflow as tf

tf.reset_default_graph()

n = 100
x = np.random.rand(n)
d = x *  3 + 2

noise = 0.3
d = d + noise * np.random.randn(n) 

W = tf.get_variable("W", shape=[1])
b = tf.get_variable("b", shape=[1])

y = W * x + b

# Function
loss = tf.reduce_mean(tf.square(y - d))

"""
# Setting
SaverとSessionのインスタンスを生成し、モデルを復元する。
"""
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, 'model/test_model')
print('Restored a model')

print(sess.run(y))
loss_val = sess.run(loss)

print('loss_val', loss_val)

print('W:', sess.run(W))
print('b:', sess.run(b))
