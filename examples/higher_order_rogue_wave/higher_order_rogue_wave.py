""" Solving Equations describing Higher Order Rogue Waves using a 4th order Runge-Kutta method """

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rk4(f, u, t, dx, h):
    """
    Fourth-order Runge-Kutta method for computing u at the next time step.
    """
    k1 = f(u, t, dx)
    k2 = f(u + 0.5*h*k1, t + 0.5*h, dx)
    k3 = f(u + 0.5*h*k2, t + 0.5*h, dx)
    k4 = f(u + h*k3, t + h, dx)

    return u + (h/6)*(k1 + 2*k2 + 2*k3 + k4)


def dudx(u, dx):
    """
    Approximate the first derivative using the centered finite difference
    formula.
    """
    first_deriv = np.zeros_like(u)

    # wrap to compute derivative at endpoints
    first_deriv[0] = (u[1] - u[-1]) / (2*dx)
    first_deriv[-1] = (u[0] - u[-2]) / (2*dx)

    # compute du/dx for all the other points
    first_deriv[1:-1] = (u[2:] - u[0:-2]) / (2*dx)

    return first_deriv


def d2udx2(u, dx):
    """
    Approximate the second derivative using the centered finite difference
    formula.
    """
    second_deriv = np.zeros_like(u)

    # wrap to compute second derivative at endpoints
    second_deriv[0] = (u[1] - 2*u[0] + u[-1]) / (dx**2)
    second_deriv[-1] = (u[0] - 2*u[-1] + u[-2]) / (dx**2)

    # compute d2u/dx2 for all the other points
    second_deriv[1:-1] = (u[2:] - 2*u[1:-1] + u[0:-2]) / (dx**2)

    return second_deriv


def f(u, t, dx):
    real = -1*d2udx2(u[:, 1], dx) - 2*(u[:, 0]**2 + u[:, 1]**2)*u[:, 1]
    imaginary = d2udx2(u[:, 0], dx) + 2*(u[:, 0]**2 + u[:, 1]**2)*u[:, 0]

    return np.hstack((real[:, None], imaginary[:, None]))


def chabchoub_solution(X, T):
    a0 = 1e-3
    k0 = 30.
    #psi0 = a0*(2**0.5)*k0 + 0.j
    #psi0 = 1. + 0.j
    psi0 = (2**0.5)

    psi2 = np.absolute(psi0)**2
    psi4 = np.absolute(psi0)**4
    X2 = X**2
    T2 = T**2

    # polynomials used in higher-order rational solution
    G = (psi2*X2 + 4*psi4*T2 + 0.75)*(psi2*X2 + 20*psi4*T2 + 0.75) - 0.75
    H = 2*psi2*T*(4*psi4*T2 - 3*psi2*X2) + 2*psi2*T*(2*((psi2*X2 + 4*psi4*T2)**2) - 15./8.)
    D = (1./3.)*((psi2*X2 + 4*psi4*T2)**3) + (1./4.)*((psi2*X2 - 12*psi4*T2)**2) + \
        (3./64.)*(12*psi2*X2 + 176*psi4*T2 + 1)

    solution = psi0*(1. - (G + 1.j*H)/D)*np.exp(2.j*psi2*T)

    return solution


def schrodinger(x0, xN, N, t0, tK, K):
    x = np.linspace(x0, xN, N)  # evenly spaced spatial points
    t = np.linspace(t0, tK, K+1)  # evenly spaced temporal points
    dx = (xN - x0) / float(N - 1)  # space between each spatial point
    dt = (tK - t0) / float(K)  # space between each temporal point
    h = 1e-5  # time step for runge-kutta method

    U = np.zeros(shape=(K+1, N, 2))
    X, T = np.meshgrid(x, t)

    # compute exact solution to the higher-order rogue wave
    U_exact = chabchoub_solution(X, T)
    H_exact = np.absolute(U_exact)

    lb = np.array([-5.0, -1.5])
    ub = np.array([5.0, 1.5])

    #with open('extended_high_order_rogue_wave_data.npz', 'wb') as solution_file:
    #    np.savez(solution_file, U=U_exact, x=x, t=t)

    # initial condition for real part of numerical solution to higher-order rogue wave
    U[0, :, 0] = np.real(U_exact[0,:])

    # initial condition for imaginary part of numerical solution to higher-order rogue wave
    U[0, :, 1] = np.imag(U_exact[0,:])

    for idx in range(K):  # for each temporal point perform runge-kutta method
        ti = t0 + dt*idx
        u = U[idx, :, :]

        for step in range(1000):
            t = ti + h*step
            u = rk4(f, u, t, dx, h)

        U[idx+1, :, :] = u

    H = np.sqrt(U[:, :, 0]**2 + U[:, :, 1]**2)

    U_exact = U_exact[150:451, 1900:2101]
    U = U[150:451, 1900:2101, :]
    H_exact = H_exact[150:451, 1900:2101]
    H = H[150:451, 1900:2101]
    x = x[1900:2101]

    rel_error = np.linalg.norm(H_exact - H, 2) / np.linalg.norm(H_exact, 2)
    print('Relative error h: {}'.format(rel_error))

    rel_error = np.linalg.norm(np.real(U_exact) - U[:,:,0], 2) / np.linalg.norm(np.real(U_exact), 2)
    print('Relative error u: {}'.format(rel_error))

    rel_error = np.linalg.norm(np.imag(U_exact) - U[:,:,1], 2) / np.linalg.norm(np.imag(U_exact), 2)
    print('Relative error v: {}'.format(rel_error))

    error = np.linalg.norm(H_exact[150,:] - H[150,:], 2) / np.linalg.norm(H_exact[150,:], 2)
    print('Error at t=0: {}'.format(error))

    plt.plot(x, H_exact[150,:], 'b-', linewidth=2, label='Exact')
    plt.plot(x, H[150,:], 'r--', linewidth=2, label='Prediction')
    plt.title('$t = 0$', fontsize=10)
    plt.xlabel('$x$')
    plt.ylabel('$|h(x,t)|$')
    plt.legend(frameon=False)
    plt.savefig('psi_exact_vs_rk4_high_order_rogue_wave.png')
    plt.show()

    plt.imshow(H, interpolation='nearest', cmap='rainbow',
               extent=[lb[0], ub[0], lb[1], ub[1]], origin='lower', aspect='auto')
    plt.title('|h(x,t)|')
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.colorbar()
    plt.savefig("psi2_rk4_higher_order_rogue_wave.png")
    plt.show()

    plt.imshow(H_exact, interpolation='nearest', cmap='rainbow',
               extent=[lb[0], ub[0], lb[1], ub[1]], origin='lower', aspect='auto')
    plt.title('$|h(x,t)|$')
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.colorbar()
    plt.savefig("psi2_exact_higher_order_rogue_wave.png")
    plt.show()


if __name__ == '__main__':
    schrodinger(-100, 100, 4001, -3, 3, 600)
