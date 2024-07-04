#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[16]:


#生成200个随机点
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise = np.random.normal(0,0.2,x_data.shape)
y_data = np.square(x_data) + noise

#设置两个占位符用于储存数据
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

#定义中间层
Weight_L1 = tf.Variable(tf.random.normal([1,10]))
bias_L1 = tf.Variable(tf.zeros([1,10]))
W_b_L1 = tf.matmul(x,Weight_L1) + bias_L1

#激活函数
L1 = tf.nn.relu(W_b_L1)

#定义输出层
Weight_L2 = tf.Variable(tf.random.normal([10,1]))
bias_L2 = tf.Variable(tf.zeros([1,1]))
W_b_L2 = tf.matmul(L1,Weight_L2) + bias_L2

#激活函数
output = tf.nn.tanh(W_b_L2)

#损失函数
loss = tf.reduce_mean(tf.square(y_data - output))

#优化器
optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#创建会话
with tf.Session() as sess:
    sess.run(init)
    for step in range(2000):
        sess.run(optimizer,feed_dict={x:x_data,y:y_data})
    output_value = sess.run(output,feed_dict={x:x_data})
    
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data,output_value,'r-', lw=5)
    plt.show()


# In[17]:





# In[5]:





# In[ ]:




