""" Solving the Burgers' Equation using a 4th order Runge-Kutta method """

import numpy as np
import matplotlib.pyplot as plt


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


def f(u, t, dx, nu=0.01/np.pi):
    return -u*dudx(u, dx) + nu*d2udx2(u, dx)


def make_square_axis(ax):
    ax.set_aspect(1 / ax.get_data_ratio())


def burgers(x0, xN, N, t0, tK, K):
    x = np.linspace(x0, xN, N)  # evenly spaced spatial points
    dx = (xN - x0) / float(N - 1)  # space between each spatial point
    dt = (tK - t0) / float(K)  # space between each temporal point
    h = 2e-6  # time step for runge-kutta method

    u = np.zeros(shape=(K, N))
    # u[0, :] = 1 + 0.5*np.exp(-(x**2))  # compute u at initial time step
    u[0, :] = -np.sin(np.pi*x)

    for idx in range(K-1):  # for each temporal point perform runge-kutta method
        ti = t0 + dt*idx
        U = u[idx, :]

        for step in range(1000):
            t = ti + h*step
            U = rk4(f, U, t, dx, h)

        u[idx+1, :] = U

    # plt.imshow(u, extent=[x0, xN, t0, tK])
    plt.imshow(u.T, interpolation='nearest', cmap='rainbow',
               extent=[t0, tK, x0, xN], origin='lower', aspect='auto')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    # burgers(-10, 10, 1024, 0, 50, 500)
    burgers(-1, 1, 512, 0, 1, 500)
