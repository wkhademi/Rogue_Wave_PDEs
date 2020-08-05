""" Solving the Time Periodic Peregrine Soliton Equation using a 4th order Runge-Kutta method """

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


def f(u, t, dx):
    real = -0.5*d2udx2(u[:, 1], dx) - (u[:, 0]**2 + u[:, 1]**2)*u[:, 1]
    imaginary = 0.5*d2udx2(u[:, 0], dx) + (u[:, 0]**2 + u[:, 1]**2)*u[:, 0]

    return np.hstack((real[:, None], imaginary[:, None]))


def schrodinger(x0, xN, N, t0, tK, K):
    x = np.linspace(x0, xN, N)  # evenly spaced spatial points
    t = np.linspace(t0, tK, K+1)  # evenly spaced temporal points
    dx = (xN - x0) / float(N - 1)  # space between each spatial point
    dt = (tK - t0) / float(K)  # space between each temporal point
    h = 2.5e-6  # time step for runge-kutta method

    U = np.zeros(shape=(K+1, N, 2))
    U_exact = np.zeros(shape=(K+1, N, 2))
    X, T = np.meshgrid(x, t)

    # compute exact solution to the time periodic peregrine soliton
    U_exact[:, :, 0] = (-3*np.cos(T*(8**0.5))*np.cos(T) + (2**0.5)*np.cosh(2*X)*np.cos(T) + (8**0.5)*np.sin(T*(8**0.5))*np.sin(T)) / ((2**0.5)*np.cosh(2*X) - np.cos(T*(8**0.5)))
    U_exact[:, :, 1] = (-3*np.cos(T*(8**0.5))*np.sin(T) + (2**0.5)*np.cosh(2*X)*np.sin(T) - (8**0.5)*np.sin(T*(8**0.5))*np.cos(T)) / ((2**0.5)*np.cosh(2*X) - np.cos(T*(8**0.5)))
    H_exact = np.sqrt(U_exact[:, :, 0]**2 + U_exact[:, :, 1]**2)

    with open('time_periodic_peregrine_soliton_data.npz', 'wb') as solution_file:
        np.savez(solution_file, U=U_exact, x=x, t=t)

    # initial condition for real and imaginary part of numerical solution to time periodic peregrine soliton equation
    U[0, :, :] = U_exact[0, :, :]

    for idx in range(K):  # for each temporal point perform runge-kutta method
        ti = t0 + dt*idx
        u = U[idx, :, :]

        for step in range(1000):
            t = ti + h*step
            u = rk4(f, u, t, dx, h)

        U[idx+1, :, :] = u

    H = np.sqrt(U[:, :, 0]**2 + U[:, :, 1]**2)

    rel_error = np.linalg.norm(H_exact - H, 2) / np.linalg.norm(H_exact, 2)
    print('Relative error: {}'.format(rel_error))

    error = np.linalg.norm(H_exact[0,:] - H[0,:], 2) / np.linalg.norm(H_exact[0,:], 2)
    print('Error at t=0: {}'.format(error))

    plt.plot(x[2750:3251], H_exact[0,2750:3251], 'b-', linewidth=2, label='Exact')
    plt.plot(x[2750:3251], H[0,2750:3251], 'r--', linewidth=2, label='Prediction')
    plt.title('$t = 0$', fontsize=10)
    plt.xlabel('$x$')
    plt.ylabel('$|h(x,t)|$')
    plt.legend(frameon=False)
    plt.savefig('exact_vs_rk4_time_periodic_peregrine_time_0_new.png')
    plt.show()

    plt.imshow(H_exact, interpolation='nearest', cmap='rainbow',
               extent=[x0, xN, t0, tK], origin='lower', aspect='auto')
    plt.title('$|h(x,t)|$')
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.colorbar()
    plt.show()

    plt.contourf(X, T, H, cmap='rainbow', origin='lower')
    plt.title('$|h(x,t)|$')
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.colorbar()
    plt.show()

    plt.imshow(H, interpolation='nearest', cmap='rainbow',
               extent=[x0, xN, t0, tK], origin='lower', aspect='auto')
    plt.title('$|h(x,t)|$')
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    schrodinger(-150, 150, 6001, 0, 6, 2400)
