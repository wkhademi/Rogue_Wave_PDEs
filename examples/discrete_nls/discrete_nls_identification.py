"""
Data-driven identification of the Ablowitz-Ladik (AL) model.

Code adapted from Physics Informed Neural Networks.

Code:
    https://github.com/maziarraissi/PINNs

Papers:
    1) https://arxiv.org/pdf/1711.10561.pdf
    2) https://arxiv.org/pdf/1711.10566.pdf
"""

import sys
import time
import scipy.io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pyDOE import lhs
from scipy.io import loadmat
from scipy.interpolate import griddata


np.random.seed(1234)
tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X, u, v, layers, lb, ub, N, m):
        self.idx = 0

        self.N = N
        self.m = m

        self.lb = lb
        self.ub = ub

        self.x = X[:,0:1]
        self.t = X[:,1:2]

        self.u = u
        self.v = v

        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # Initialize parameters
        self.lambda_1 = tf.Variable([1.0], dtype=tf.float32)

        # tf Placeholders
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])

        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]])

        self.N_tf = tf.placeholder(tf.int32, shape=())
        self.m_tf = tf.placeholder(tf.int32, shape=())

        # tf Graphs
        self.u_pred, self.v_pred, _ , _ = self.net_uv(self.x_tf, self.t_tf, self.N_tf, self.m_tf)
        self.f_u_pred, self.f_v_pred = self.net_f_uv(self.x_tf, self.t_tf, self.N_tf, self.m_tf)

        # Loss
        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.v_tf - self.v_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_v_pred))

        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method = 'L-BFGS-B',
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50})
                                                                           #'ftol' : 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session()

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2./(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_uv(self, x, t, N, m):
        X = tf.concat([x, t], 1)

        uv = self.neural_net(X, self.weights, self.biases)
        u = uv[:, 0:1]
        v = uv[:, 1:2]

        # reshape so lattice nodes can be shifted at each time step
        reshaped_u = tf.reshape(u, shape=(N, m))
        reshaped_v = tf.reshape(v, shape=(N, m))

        u_x = tf.roll(reshaped_u, -1, axis=-1) + tf.roll(reshaped_u, 1, axis=-1)
        v_x = tf.roll(reshaped_v, -1, axis=-1) + tf.roll(reshaped_v, 1, axis=-1)

        u_x = tf.reshape(u_x, shape=(-1, 1))
        v_x = tf.reshape(v_x, shape=(-1, 1))

        return u, v, u_x, v_x

    def net_f_uv(self, x, t, N, m):
        lambda1 = tf.exp(self.lambda_1)

        u, v, u_x, v_x = self.net_uv(x, t, N, m)

        # compute partials with respect to time
        u_t = tf.gradients(u, t)[0]
        v_t = tf.gradients(v, t)[0]

        # reshape so lattice nodes can be shifted at each time step
        reshaped_u = tf.reshape(u, shape=(N, m))
        reshaped_v = tf.reshape(v, shape=(N, m))

        # compute discrete spatial second derivative
        u_xx = tf.roll(reshaped_u, -1, axis=-1) - 2*reshaped_u + tf.roll(reshaped_u, 1, axis=-1)
        v_xx = tf.roll(reshaped_v, -1, axis=-1) - 2*reshaped_v + tf.roll(reshaped_v, 1, axis=-1)

        u_xx = tf.reshape(u_xx, shape=(-1, 1))
        v_xx = tf.reshape(v_xx, shape=(-1, 1))

        lambda1 = 1.

        f_u = u_t + v_xx + lambda1*v_x*(u**2 + v**2) + 2.*(1.-lambda1)*(u**2 + v**2)*v - v
        f_v = v_t - u_xx - lambda1*u_x*(u**2 + v**2) - 2.*(1.-lambda1)*(u**2 + v**2)*u + u

        return f_u, f_v

    def callback(self, loss, lambda_1):
        print('Iteration {} - Loss: {}, l1: {}'.format(str(self.idx), str(loss), str(lambda_1)))
        self.idx += 1

    def train(self, nIter):
        tf_dict = {self.N_tf: self.N, self.m_tf: self.m,
                   self.x_tf: self.x, self.t_tf: self.t,
                   self.u_tf: self.u, self.v_tf: self.v}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value, lambda_1 = self.sess.run([self.loss, tf.exp(self.lambda_1)], tf_dict)
                print('It: %d, Loss: %.3e, l1: %.5f, Time: %.2f' %
                      (it, loss_value, lambda_1, elapsed))
                start_time = time.time()

        #self.optimizer.minimize(self.sess,
        #                        feed_dict = tf_dict,
        #                        fetches = [self.loss, self.lambda_1],
        #                        loss_callback = self.callback)

    def predict(self, X_star, N, m):
        tf_dict = {self.N_tf: N, self.m_tf: m, self.x_tf: X_star[:,0:1], self.t_tf: X_star[:,1:2]}

        u_star, v_star, f_u_star, f_v_star = self.sess.run([self.u_pred, self.v_pred,
                                                            self.f_u_pred, self.f_v_pred], tf_dict)

        return u_star, v_star, f_u_star, f_v_star


if __name__ == "__main__":
    lb = np.array([-50, 0])
    ub = np.array([50, 2*np.pi])

    m = 101  # number of lattice nodes
    x = np.linspace(-50, 50, m)

    lambda_1 = 1.0

    N_total = 5001
    N = 1000  # number of time steps to sample for training
    layers = [2, 100, 100, 100, 100, 2]

    with open('data_AL.mat', 'rb') as solution_file:
        data = loadmat(solution_file)
        t = data['tout']
        Exact = data['yout']

    Exact_u = Exact[:,:m]
    Exact_v = Exact[:,m:]
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)

    X, T = np.meshgrid(x,t)

    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact_u.flatten()[:,None]
    v_star = Exact_v.flatten()[:,None]
    h_star = Exact_h.flatten()[:,None]

    ########################## Train neural network ############################

    idx = np.random.choice(t.shape[0], N, replace=False)
    X_train = np.hstack((X[idx,:].flatten()[:,None], T[idx,:].flatten()[:,None]))
    u_train = Exact_u[idx,:].flatten()[:,None]
    v_train = Exact_v[idx,:].flatten()[:,None]

    model = PhysicsInformedNN(X_train, u_train, v_train, layers, lb, ub, N, m)

    start_time = time.time()
    model.train(10000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    ################### Predict solution and compute error #####################

    lambda_1_value = model.sess.run(model.lambda_1)
    u_pred, v_pred, f_u_pred, f_v_pred = model.predict(X_star, N_total, m)
    h_pred = np.sqrt(u_pred**2 + v_pred**2)
    H_pred = griddata(X_star, h_pred.flatten(), (X, T), method='cubic')

    error_lambda_1 = np.abs(np.exp(lambda_1_value) - lambda_1)/lambda_1 * 100
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_h = np.linalg.norm(h_star-h_pred,2)/np.linalg.norm(h_star,2)
    error_H = np.linalg.norm(Exact_h[0,:]-H_pred[0,:],2)/np.linalg.norm(Exact_h[0,:],2)

    print('Error l1: %.5f%%' % (error_lambda_1))
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error h: %e' % (error_h))
    print('Error at time step = 0: %e' % (error_H))

    ############################# Plot solution ################################

    plt.imshow(H_pred, interpolation='nearest', cmap='rainbow',
               extent=[lb[0], ub[0], lb[1], ub[1]], origin='lower', aspect='auto')
    plt.title('|h(x,t)|')
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.colorbar()
    plt.savefig("predicted_peregrine_solition_identification.png")
    plt.show()

    plt.plot(x, Exact_h[:,50], 'b-', linewidth = 2, label = 'Exact')
    plt.plot(x, H_pred[:,50], 'r--', linewidth = 2, label = 'Prediction')
    plt.title('$x = 0$', fontsize=10)
    plt.xlabel('$t$')
    plt.ylabel('$|h(x,t)|$')
    plt.legend(frameon=False)
    plt.savefig("predicited_vs_exact_peregrine_time_step_0_identification.png")
    plt.show()
