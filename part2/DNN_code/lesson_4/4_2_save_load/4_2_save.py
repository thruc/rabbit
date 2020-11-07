import numpy as np
import tensorflow as tf

iters_num = 100
plot_interval = 10

# データを生成
n = 100
x = np.random.rand(n)
d = x *  3 + 2

# ノイズを加える
noise = 0.3
d = d + noise * np.random.randn(n) 


# 最適化の対象の変数を初期化
W = tf.Variable(tf.zeros([1]), name='W')
b = tf.Variable(tf.zeros([1]), name='b')

# モデル
y = W * x + b

# 誤差関数
loss = tf.reduce_mean(tf.square(y - d))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初期化
saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# トレーニング
for i in range(iters_num):
    sess.run(train)
    if (i+1) % plot_interval == 0:
        loss_val = sess.run(loss) 
        print('Generation: ' + str(i+1) + '. 誤差 = ' + str(loss_val))

# saver.save(sess, 'model.ckpt')        
saver.save(sess, 'model/test_model')

print('W:', sess.run(W))
print('b:', sess.run(b))

sess.close()