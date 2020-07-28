""" Solving the Peregrine Soliton Equation using a 4th order Runge-Kutta method """

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
    h = 1e-5  # time step for runge-kutta method

    U = np.zeros(shape=(K+1, N, 2))
    U_exact = np.zeros(shape=(K+1, N, 2))
    X, T = np.meshgrid(x, t)

    # compute exact solution to the peregrine soliton
    U_exact[:, :, 0] = np.cos(T) - ((4*np.cos(T) - 8*T*np.sin(T)) / (1 + 4*(T**2) + 4*(X**2)))
    U_exact[:, :, 1] = np.sin(T) - ((4*np.sin(T) + 8*T*np.cos(T)) / (1 + 4*(T**2) + 4*(X**2)))
    H_exact = np.sqrt(U_exact[:, :, 0]**2 + U_exact[:, :, 1]**2)

    with open('peregrine_soliton_data.npz', 'wb') as solution_file:
        np.savez(solution_file, U=U_exact, x=x, t=t)

    # initial condition for real part of numerical solution to peregrine soliton equation
    U[0, :, 0] = np.cos(t0) - ((4*np.cos(t0) - 8*t0*np.sin(t0)) / (1 + 4*(t0**2) + 4*(x**2)))

    # initial condition for imaginary part of numerical solution to peregrine soliton equation
    U[0, :, 1] = np.sin(t0) - ((4*np.sin(t0) + 8*t0*np.cos(t0)) / (1 + 4*(t0**2) + 4*(x**2)))

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

    error = np.linalg.norm(H_exact[600,:] - H[600,:], 2) / np.linalg.norm(H_exact[600,:], 2)
    print('Error at t=0: {}'.format(error))

    plt.plot(x[1750:2251], H_exact[600,1750:2251], 'b-', linewidth=2, label='Exact')
    plt.plot(x[1750:2251], H[600,1750:2251], 'r--', linewidth=2, label='Prediction')
    plt.title('$t = 0$', fontsize=10)
    plt.xlabel('$x$')
    plt.ylabel('$|h(x,t)|$')
    plt.legend(frameon=False)
    plt.savefig('exact_vs_rk4_peregrine_time_0.png')
    plt.show()

    plt.contourf(X, T, H_exact, cmap='rainbow', origin='lower')
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
    schrodinger(-100, 100, 4001, -6, 6, 1200)
