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
import tf.keras.backend as K
import matplotlib.pyplot as plt

from pyDOE import lhs
from tf.keras import Model
from tf.keras.losses import MSE
from tf.keras.optimizers import Adam
from scipy.interpolate import griddata
from tf.keras.initializers import Constant
from tf.keras.layers import LSTM, TimeDistributed, Dense, Input, Layer


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
                                       initializer=Constant(3.0),
                                       trainable=True)

        super(PhysicsInformedLayer, self).build(input_shape)

    def call(self, input):
        lambda_1 = K.exp(self.lambda_1)
        lambda_2 = self.lambda_2

        xt = input[0]
        uv = input[1]

        x = xt[:,:,0:1]
        t = xt[:,:,1:2]
        u = uv[:,:,0:1]
        v = uv[:,:,1:2]

        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]

        v_t = tf.gradients(v, t)[0]
        v_x = tf.gradients(v, x)[0]
        v_xx = tf.gradients(v_x, x)[0]

        f_u = u_t + lambda_1*v_xx + lambda_2*(u**2 + v**2)*v
        f_v = v_t - lambda_1*u_xx - lambda_2*(u**2 + v**2)*u

        f_uv = K.concatenate([f_u, f_v], axis=-1)
        f_uv = K.reshape(f_uv, shape=input[0].get_shape().as_tuple())

        return f_uv

    def compute_output_shape(self, input_shape):
        return input_shape


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X, u, v, lb, ub):
        pass

    def build_rnn(self, input_shape):
        xt = Input(shape=input_shape)

        recurrent1 = LSTM(128, return_sequences=True)(xt)
        recurrent2 = LSTM(128, return_sequences=True)(recurrent1)
        uv = TimeDistributed(Dense(2))(recurrent2)

        f_uv = PhysicsInformedLayer()([xt, uv])
        loss = self.physics_informed_loss(f_uv)

        model = Model(inputs=xt, outputs=uv)
        model.compile(loss=loss, optimizer=Adam(learning_rate=1e-4))

        return model

    def physics_informed_loss(self, f_uv):
        def loss(uv_true, uv_pred):
            total_loss = MSE(uv_true, uv_pred) + MSE(f_uv, 0.)

            return total_loss

        return loss

    def predict(self, X_star):
        tf_dict = {self.x0_tf: X_star[:,0:1], self.t0_tf: X_star[:,1:2]}
        u_star = self.sess.run(self.u0_pred, tf_dict)
        v_star = self.sess.run(self.v0_pred, tf_dict)

        tf_dict = {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]}
        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        f_v_star = self.sess.run(self.f_v_pred, tf_dict)

        return u_star, v_star, f_u_star, f_v_star


if __name__ == "__main__":
    lb = np.array([-7.5, -1.0])
    ub = np.array([7.5, 6.0])

    N0 = 50
    N_b = 50
    N_f = 20000

    with open('time_periodic_peregrine_soliton_data1.npz', 'rb') as solution_file:
        data = np.load(solution_file)
        t = data['t'][::12,None]  # t in [0,6]
        x = data['x'][2850:3151,None]  # x in [-7.5, 7.5]
        Exact = data['U'][::12,2850:3151]

    Exact_u = Exact[:,:,0]
    Exact_v = Exact[:,:,1]
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)

    X, T = np.meshgrid(x,t)

    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact_u.flatten()[:,None]
    v_star = Exact_v.flatten()[:,None]
    h_star = Exact_h.flatten()[:,None]

    ########################## Train neural network ############################

    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x,:]
    u0 = np.concatenate((Exact_u[0:1,idx_x].T, Exact_u[74:75,idx_x].T), axis=0)
    v0 = np.concatenate((Exact_v[0:1,idx_x].T, Exact_v[74:75,idx_x].T), axis=0)

    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t,:]

    X_f = lb + (ub-lb)*lhs(2, N_f)

    model = PhysicsInformedNN(x0, u0, v0, lb, ub)

    start_time = time.time()
    model.train(0)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

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
