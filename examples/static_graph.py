# coding: utf-8
import tensorflow as tf 
import numpy as np 

# In TensorFlow, we define the computational graph once and then execute the same graph over and over again,
# In PyTorch, each forward pass defines a new computational graph.
# Static graphs are nice because you can optimize the graph up front(预先).

# First we set up the computational graph:
# ---------------------------------------------------------------------
N, D_in, H, D_out = 64, 1000, 100, 10 

# create placeholders for the input and target data
x = tf.placeholder(tf.float32, shape=(None, D_in))
y = tf.placeholder(tf.float32, shape=(None, D_out))

# create variables for the weights and initialize them with random data
w1 = tf.Variable(tf.random_normal((D_in, H)))
w2 = tf.Variable(tf.random_normal((H, D_out)))

# forward pass
h = tf.matmul(x, w1)
h_relu = tf.maximum(h, tf.zeros(1))
y_pred = tf.matmul(h_relu, w2)

# compute the loss
loss = tf.reduce_sum((y - y_pred) ** 2.0)

# compute gradient of the loss with respect to w1 and w2.
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

# update the weights using gradient descent. To actually update the weights
# we need to evaluate new_w1 and new_w2 when executing the graph. Note that
# in TensorFlow the act of updating the value of the weights is part of
# the computational graph; in PyTorch this happens outside the computational
# graph. (在Tensorflow中权值更新是计算图的一部分, 而pytorch中不是)
learning_rate = 1e-6
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)

# Now we have built our computational graph, so we enter a TensorFlow session to
# actually execute the graph.
# ---------------------------------------------------------------------
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # get data
    # create Numpy array holding the actual data for the inputs x and targets y
    x_value = np.random.randn(N, D_in)
    y_value = np.random.randn(N, D_out)

    # epoch = 500
    for t in range(500):
        loss_value, _, _ = sess.run([loss, new_w1, new_w2], feed_dict={x: x_value, y: y_value})
        print(t, loss_value)
