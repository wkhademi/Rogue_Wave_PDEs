"""
Data-driven inference of the Ablowitz-Ladik (AL) model.

Code adapted from Physics Informed Neural Networks.

Code:
    https://github.com/maziarraissi/PINNs

Papers:
    1) https://arxiv.org/pdf/1711.10561.pdf
    2) https://arxiv.org/pdf/1711.10566.pdf
"""

import sys
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.interpolate import griddata

np.random.seed(1234)
tf.set_random_seed(1234)

g = 1.


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x0, u0, v0, X_f, layers, lb, ub, N_init, N, m):
        self.idx = 0

        t0 = np.repeat(np.linspace(lb[1], ub[1], N_init), x0.shape[0])[:, None]
        x0 = np.tile(x0, (N_init, 1))
        X0 = np.concatenate((x0, t0), 1)

        self.N_init = N_init
        self.N = N
        self.m = m

        self.lb = lb
        self.ub = ub

        self.x0 = X0[:, 0:1]
        self.t0 = X0[:, 1:2]

        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]

        self.u0 = u0
        self.v0 = v0

        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # tf Placeholders
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])

        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])
        self.v0_tf = tf.placeholder(tf.float32, shape=[None, self.v0.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.N_init_tf = tf.placeholder(tf.int32, shape=())
        self.N_tf = tf.placeholder(tf.int32, shape=())
        self.m_tf = tf.placeholder(tf.int32, shape=())

        # tf Graphs
        self.u0_pred, self.v0_pred, _, _, _, _ = self.net_uv(self.x0_tf, self.t0_tf, self.N_init_tf, self.m_tf)
        self.u_pred, self.v_pred, self.u_x_pred, self.v_x_pred, self.f_u_pred, self.f_v_pred = self.net_f_uv(self.x_f_tf, self.t_f_tf, self.N_tf, self.m_tf)

        # reshape data back to spatio-temporal domain layout
        self.u_pred = tf.reshape(self.u_pred, shape=(self.N_tf, self.m_tf))
        self.v_pred = tf.reshape(self.v_pred, shape=(self.N_tf, self.m_tf))
        self.u_x_pred = tf.reshape(self.u_x_pred, shape=(self.N_tf, self.m_tf))
        self.v_x_pred = tf.reshape(self.v_x_pred, shape=(self.N_tf, self.m_tf))

        # get boundary values to enforce boundary condition
        self.u_lb_pred = self.u_pred[:, 0]
        self.u_ub_pred = self.u_pred[:, -1]
        self.v_lb_pred = self.v_pred[:, 0]
        self.v_ub_pred = self.v_pred[:, -1]
        self.u_x_lb_pred = self.u_x_pred[:, 0]
        self.u_x_ub_pred = self.u_x_pred[:, -1]
        self.v_x_lb_pred = self.v_x_pred[:, 0]
        self.v_x_ub_pred = self.v_x_pred[:, -1]

        # Loss
        self.loss = tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
            tf.reduce_mean(tf.square(self.v0_tf - self.v0_pred)) + \
            tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred)) + \
            tf.reduce_mean(tf.square(self.v_lb_pred - self.v_ub_pred)) + \
            tf.reduce_mean(tf.square(self.f_u_pred)) + \
            tf.reduce_mean(tf.square(self.f_v_pred))

            #tf.reduce_mean(tf.square(self.u_x_lb_pred - self.u_x_ub_pred)) + \
            #tf.reduce_mean(tf.square(self.v_x_lb_pred - self.v_x_ub_pred)) + \

        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method = 'L-BFGS-B',
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 100,
                                                                           'maxls': 50,
                                                                           'gtol': 1e-8,
                                                                           'eps': 1,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

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
        for l in range(0, num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32), dtype=tf.float32)
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
        for l in range(0, num_layers-2):
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

        test_u_x = 0.5 * (tf.roll(reshaped_u, -1, axis=-1) - tf.roll(reshaped_u, 1, axis=-1))
        test_v_x = 0.5 * (tf.roll(reshaped_v, -1, axis=-1) - tf.roll(reshaped_v, 1, axis=-1))

        u_x = tf.reshape(u_x, shape=(-1, 1))
        v_x = tf.reshape(v_x, shape=(-1, 1))

        return u, v, u_x, v_x, test_u_x, test_v_x

    def net_f_uv(self, x, t, N, m):
        u, v, u_x, v_x, test_u_x, test_v_x = self.net_uv(x, t, N, m)

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

        f_u = u_t + v_xx + g*v_x*(u**2 + v**2) + 2.*(1.-g)*(u**2 + v**2)*v - v
        f_v = v_t - u_xx - g*u_x*(u**2 + v**2) - 2.*(1.-g)*(u**2 + v**2)*u + u

        return u, v, test_u_x, test_v_x, f_u, f_v

    def callback(self, loss):
        print('Iteration {} - Loss: {}'.format(str(self.idx), str(loss)))
        self.idx += 1

    def train(self, nIter):
        tf_dict = {self.N_init_tf: self.N_init, self.N_tf: self.N, self.m_tf: self.m,
                   self.x0_tf: self.x0, self.t0_tf: self.t0,
                   self.u0_tf: self.u0, self.v0_tf: self.v0,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

        #self.optimizer.minimize(self.sess,
        #                        feed_dict = tf_dict,
        #                        fetches = [self.loss],
        #                        loss_callback = self.callback)

    def predict(self, X_star, N_init, N, m):
        tf_dict = {self.N_init_tf: N_init, self.m_tf: m,
                   self.x0_tf: X_star[:, 0:1], self.t0_tf: X_star[:, 1:2]}
        u_star = self.sess.run(self.u0_pred, tf_dict)
        v_star = self.sess.run(self.v0_pred, tf_dict)

        tf_dict = {self.N_tf: N, self.m_tf: m,
                   self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]}
        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        f_v_star = self.sess.run(self.f_v_pred, tf_dict)

        return u_star, v_star, f_u_star, f_v_star


if __name__ == "__main__":
    lb = np.array([-50., 0.])
    ub = np.array([50., 6*np.pi / 5.])

    omega = 2.*np.pi / 5.  # period
    dt = omega / 2.  # time step for sampling solution

    m = 101  # number of lattice nodes
    N_init = int(np.rint(ub[1] / dt) + 1)  # number of time steps with solution
    N = 600  # number of time steps to use as collocation points for training
    N_total = 3001  # total number of time steps
    layers = [2, 100, 100, 100, 100, 2]

    x = np.linspace(lb[0], ub[0], m)

    with open('discrete_exact_nls_data.npz', 'rb') as solution_file:
        data = np.load(solution_file)
        t = data['t']
        Exact = data['U']

    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact_u.flatten()[:, None]
    v_star = Exact_v.flatten()[:, None]
    h_star = Exact_h.flatten()[:, None]

    ########################## Train neural network ############################

    # training points with solution
    x0 = x[:, None]
    peak_idx = np.linspace(0., N_total-1, N_init, dtype=np.int32)
    u0 = Exact_u[peak_idx, :].flatten()[:, None]
    v0 = Exact_v[peak_idx, :].flatten()[:, None]

    # collocation points
    idx = np.random.choice(t.shape[0], N, replace=False)
    X_f, T_f = np.meshgrid(x, t[idx])
    X_f = np.hstack((X_f.flatten()[:, None], T_f.flatten()[:, None]))

    model = PhysicsInformedNN(x0, u0, v0, X_f, layers, lb, ub, N_init, N, m)

    start_time = time.time()
    model.train(150000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    ################### Predict solution and compute error #####################

    u_pred, v_pred, f_u_pred, f_v_pred = model.predict(X_star, N_total, N_total, m)
    h_pred = np.sqrt(u_pred**2 + v_pred**2)
    H_pred = griddata(X_star, h_pred.flatten(), (X, T), method='cubic')

    error_u = np.linalg.norm(u_star-u_pred, 2)/np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star-v_pred, 2)/np.linalg.norm(v_star, 2)
    error_h = np.linalg.norm(h_star-h_pred, 2)/np.linalg.norm(h_star, 2)
    error_H = np.linalg.norm(Exact_h[:, 50]-H_pred[:, 50], 2)/np.linalg.norm(Exact_h[:, 50], 2)

    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error h: %e' % (error_h))
    print('Error at time step = 2: %e' % (error_H))

    ############################# Plot solution ################################

    plt.imshow(H_pred, interpolation='nearest', cmap='rainbow',
               extent=[lb[0], ub[0], lb[1], ub[1]], origin='lower', aspect='auto')
    plt.title('|h(x,t)|')
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.colorbar()
    plt.savefig("predicted_exact_DNLS.png")
    plt.show()

    plt.plot(t, Exact_h[:, 50], 'b-', linewidth = 2, label = 'Exact')
    plt.plot(t, H_pred[:, 50], 'r--', linewidth = 2, label = 'Prediction')
    plt.title('$x = 0$', fontsize=10)
    plt.xlabel('$t$')
    plt.ylabel('$|h(x,t)|$')
    plt.legend(frameon=False)
    plt.savefig("predicited_vs_exact_DNLS_lattice_node_0.png")
    plt.show()
