from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 获取数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# print(mnist.train.images.shape, mnist.train.labels.shape)
# print(mnist.test.images.shape, mnist.test.labels.shape)
# print(mnist.validation.images.shape, mnist.validation.labels.shape)

# 建立session，定义算法公式，也就是神经网络forward时的计算
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义loss
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# 选定优化器，并指定优化器优化loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 初始化参数
tf.global_variables_initializer().run()
# 迭代训练
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

# 在测试集或验证集上对准确率进行评测
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
