import numpy as np
import matplotlib.pyplot as plt

num_periods = 3
omega = 5.  # frequency
T = 2*np.pi / omega  # period

# define bounds
lb = np.array([-50, 0])
ub = np.array([50, num_periods*T])

n = np.reshape(np.linspace(lb[0], ub[0], 101), (1, 101))  # lattice nodes
t = np.reshape(np.linspace(lb[1], ub[1], 3001), (3001, 1))  # time steps

# compute exact solution
theta = -1. * np.arcsinh(omega)
r = np.arccosh((2 + np.cosh(theta)) / 3.)
G = -1. * omega / (np.sqrt(3) * np.sinh(r))
q = 1. / np.sqrt(2)
psi = q * (np.cos(omega*t + 1j*theta) + G*np.cosh(r*n)) / (np.cos(omega*t) + G*np.cosh(r*n))
psi_mag = np.abs(psi)

with open('discrete_exact_nls_data.npz', 'wb') as solution_file:
    np.savez(solution_file, U=psi, x=n, t=t)

# plot solution over entire spatiotemporal domain
plt.imshow(psi_mag, interpolation='nearest', cmap='rainbow',
           extent=[lb[0], ub[0], lb[1], ub[1]], origin='lower', aspect='auto')
plt.title('|h(x,t)|')
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.colorbar()
plt.savefig("exact_DNLS.png")
plt.show()

# plot solution over time at lattice node n=0
plt.plot(t, psi_mag[:,50], 'b-', linewidth = 2, label = 'Exact')
plt.title('$x = 0$', fontsize=10)
plt.xlabel('$t$')
plt.ylabel('$|h(x,t)|$')
plt.legend(frameon=False)
plt.savefig("exact_DNLS_lattice_node_0.png")
plt.show()
