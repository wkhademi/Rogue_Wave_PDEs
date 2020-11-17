"""
Data-driven inference of the Peregrine Soliton using Neural Networks.

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
import tensorflow.keras.backend as K

from pyDOE import lhs
from tensorflow.keras import Model
from scipy.interpolate import griddata
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import LSTM, TimeDistributed, Dense, Input, Layer


np.random.seed(1234)
tf.set_random_seed(1234)


class PhysicsInformedLayer(Layer):
    def __init__(self, **kwargs):
        super(PhysicsInformedLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.lambda1 = self.add_weight(name='lambda1',shape=(),
                                       initializer=Constant(1.0),
                                       trainable=True)

        self.lambda2 = self.add_weight(name='lambda2',shape=(),
                                       initializer=Constant(-3.0),
                                       trainable=True)

        super(PhysicsInformedLayer, self).build(input_shape)

    def call(self, input):
        lambda1 = K.print_tensor(K.exp(self.lambda1), message='lambda1: ')
        lambda2 = K.print_tensor(self.lambda2 + 0., message='lambda2: ')

        x = input[0]
        t = input[1]
        uv = input[2]

        u = K.reshape(uv[:,:,0:1], shape=(-1, 1))
        v = K.reshape(uv[:,:,1:2], shape=(-1, 1))

        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]

        v_t = tf.gradients(v, t)[0]
        v_x = tf.gradients(v, x)[0]
        v_xx = tf.gradients(v_x, x)[0]

        f_u = u_t + lambda1*v_xx + lambda2*(u**2 + v**2)*v
        f_v = v_t - lambda1*u_xx - lambda2*(u**2 + v**2)*u

        f_uv = K.concatenate([f_u, f_v], axis=-1)
        f_uv = K.reshape(f_uv, shape=(-1,)+input[3])

        return f_uv

    def compute_output_shape(self, input_shape):
        return input_shape[2].get_shape()


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, input_shape):
        self.model = self.build_rnn(input_shape)

    def build_rnn(self, input_shape):
        xt = Input(shape=input_shape)

        # hack for computing gradients in PhysicsInformedLayer
        x = K.reshape(xt[:,:,0:1], shape=(-1, 1))
        t = K.reshape(xt[:,:,1:2], shape=(-1, 1))
        input = K.concatenate([x, t], axis=-1)
        input = K.reshape(input, shape=(-1,)+input_shape)

        recurrent1 = LSTM(128, return_sequences=True, unroll=True)(input)
        #recurrent2 = LSTM(128, return_sequences=True, unroll=True)(recurrent1)
        uv = TimeDistributed(Dense(2))(recurrent1)

        f_uv = PhysicsInformedLayer()([x, t, uv, input_shape])

        model = Model(inputs=xt, outputs=[uv, f_uv])
        model.compile(loss=MSE, optimizer=Adam(learning_rate=1e-4))

        return model

    def physics_informed_loss(self, f_uv):
        def loss(uv_true, uv_pred):
            total_loss = MSE(uv_true, uv_pred) + MSE(f_uv, 0.)

            return total_loss

        return loss

    def train(self, xt, uv, iterations):
        zero_const = np.zeros(uv.shape)

        for idx in range(iterations):
            loss = self.model.train_on_batch(xt, y=[uv, zero_const])
            print('Iteration %s: %s'%(idx, loss))

    def predict(self):
        pass


if __name__ == "__main__":
    lb = np.array([-7.5, -1.0])
    ub = np.array([7.5, 6.0])

    N0 = 10
    N_b = 10

    with open('time_periodic_peregrine_soliton_data1.npz', 'rb') as solution_file:
        data = np.load(solution_file)
        t = data['t'][::14,None]  # t in [-1,6]
        x = data['x'][2850:3151,None]  # x in [-7.5, 7.5]
        Exact = data['U'][::14,2850:3151]

    Exact_u = Exact[:,:,0]
    Exact_v = Exact[:,:,1]
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)

    X, T = np.meshgrid(x,t)

    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact_u.flatten()[:,None]
    v_star = Exact_v.flatten()[:,None]
    h_star = Exact_h.flatten()[:,None]

    ########################## Train neural network ############################

    x_step = len(x) // N0
    sample_x = x[::x_step,:]

    t_step = len(t) // N_b
    sample_t = t[::t_step,:]

    sample_u = Exact_u[::t_step,::x_step].T
    sample_v = Exact_v[::t_step,::x_step].T

    sample_X, sample_T = np.meshgrid(sample_x, sample_t)
    sample_X = sample_X.T
    sample_T = sample_T.T

    xt = np.concatenate([sample_X[:,:,None], sample_T[:,:,None]], axis=-1)
    uv = np.concatenate([sample_u[:,:,None], sample_v[:,:,None]], axis=-1)

    model = PhysicsInformedNN(xt.shape[1:])

    start_time = time.time()
    model.train(xt, uv, 20000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    sys.exit()

    ################### Predict solution and compute error #####################

    u_pred, v_pred, f_u_pred, f_v_pred = model.predict(X_star)
    h_pred = np.sqrt(u_pred**2 + v_pred**2)
    H_pred = griddata(X_star, h_pred.flatten(), (X, T), method='cubic')

    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_h = np.linalg.norm(h_star-h_pred,2)/np.linalg.norm(h_star,2)
    error_H = np.linalg.norm(Exact_h[0,:]-H_pred[0,:],2)/np.linalg.norm(Exact_h[0,:],2)

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
    plt.savefig("predicted_time_periodic_lstm_identification.png")
    plt.show()

    plt.plot(x, Exact_h[29,:], 'b-', linewidth = 2, label = 'Exact')
    plt.plot(x, H_pred[29,:], 'r--', linewidth = 2, label = 'Prediction')
    plt.title('$t = 0$', fontsize=10)
    plt.xlabel('$x$')
    plt.ylabel('$|h(x,t)|$')
    plt.legend(frameon=False)
    plt.savefig("predicited_vs_exact_time_periodic_time_step_0_lstm_identification.png")
    plt.show()
