import board
import numpy as np
import tensorflow as tf
import tflearn

grid_size = 2


def build_policy_network(grid_size):
    num_inputs = np.power(2, 2*(grid_size+1))
    inputs = tf.placeholder(tf.float32,[None,num_inputs])
    net = tflearn.fully_connected(net,2*num_inputs,activation='sigmoid')
    policy_network = tflearn.fully_connected(net,num_inputs,activation='Softmax')
    return inputs, policy_network


def build_graph(grid_size):
    state, policy_net = build_policy_network(grid_size)
    num_inputs = np.power(2, 2*(grid_size+1))

    network_params = tf.trainable_variables()

    reward = tf.placeholder('float',[None])
    logprob = tf.log(policy_net)
    mean = tf.reduce_sum(logprob, reduction_indices=1)
    loss = tf.mul(mean, reward)
    loss = tf.mul(loss, -1)
    optimizer = tf.train.AdadeltaOptimizer()
    grad_update = optimizer.minimize(loss, var_list=network_params)

if __name__ == '__main__':

    print('Running Program!')