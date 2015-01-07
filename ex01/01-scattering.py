# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy import sqrt, exp, sinh

plt.rcParams['figure.figsize'] = 16, 9


# Out[1]:

#     Populating the interactive namespace from numpy and matplotlib
#

## Numerov's Method

##### Single Step

# In[2]:

def numerov_step(k0, k1, k2, psi0, psi1, dx):
    """compute psi2 via a single numerov step"""
    dx_sqr = dx**2
    c0 = (1 + 1/12. * dx_sqr * k0)
    c1 = 2 * (1 - 5/12. * dx_sqr * k1)
    c2 = (1 + 1/12. * dx_sqr * k2)
    return (c1 * psi1 - c0 * psi0) / c2


# As a basic test, let's check the "symmetry property" of a single Numerov step:

# In[3]:

assert np.isclose(numerov_step(2, 4, 7, numerov_step(7, 4, 2, 2, 4, .25), 4, .25), 2)


##### Low-Level Numerov

# In[4]:

def numerov_iter(ks, psi0, psi1, dx):
    """compute psis = [psi0, psi1, psi2, ...] for ks = [k0, k1, ...] via iterated numerov steps"""
    n = len(ks)
    psis = np.zeros(n, dtype=np.complex128)
    psis[0] = psi0
    psis[1] = psi1
    for i in range(2, n):
        psis[i] = numerov_step(ks[i-2], ks[i-1], ks[i], psis[i-2], psis[i-1], dx)
    return psis


##### High-Level Numerov

# In[5]:

def numerov(k, psi0, psi1, x_min, x_max, n):
    """compute psis = [psi0, psi1, ...] for k = k(x) according to given discretization"""
    xs, dx = np.linspace(x_min, x_max, num=n, retstep=True)
    ks = np.vectorize(k)(xs)
    return xs, numerov_iter(ks, psi0, psi1, dx)

def numerov_right_to_left(k, psi_2ndlast, psi_last, x_min, x_max, n):
    """compute psis = [..., psi_2ndlast, psi_last] for k = k(x) according to given discretization"""
    xs, dx = np.linspace(x_min, x_max, num=n, retstep=True)
    ks = np.vectorize(k)(xs)
    return xs, numerov_iter(ks[::-1], psi_last, psi_2ndlast, dx)[::-1]


##### Scattering Coefficients

# In[6]:

def scatter(V, E, x_min, x_max, n):
    """compute transmission and reflection coefficients for given potential and energy (and m = hbar = 1)"""
    # compute step size
    dx = (x_max - x_min) / float(n)

    # start with right-travling wave at energy E (w/ phase fixed to 1 at x_max)
    q_free = sqrt(2. * E)
    psi_2nd_last = 1
    psi_last = exp(1j * q_free * dx)

    # compute resulting solution using Numerov
    def k(x):
        return 2. * (E - V(x))
    _, psis = numerov_right_to_left(k, psi_2nd_last, psi_last, x_min - dx, x_max + dx, n + 2)

    # fit psis[] to A * exp(iqx) + B * exp(-iqx)
    A, B = np.linalg.solve(
        [[exp(1j*q_free*(-dx)), exp(-1j*q_free*(-dx))],
         [1, 1]],
        [psis[0], psis[1]])

    # extract transmission and reflection coefficients
    T = 1 / abs(A)**2
    R = abs(B)**2 / abs(A)**2
    return T, R


# "Test" with a free particle:

# In[7]:

assert np.allclose(scatter(V=lambda _: 0, E=1, x_min=0, x_max=1, n=1e5), [1, 0])


## Rectangular Potential

# In[8]:

def rect_potential(a):
    def V(x):
        return float(0 < x <= a)
    return V

xs = np.linspace(-0.5, 1.5, num=100)
plt.figure(figsize=(4, 3))
plt.plot(xs, [rect_potential(a=1)(x) for x in xs])
plt.show()


# Out[8]:

# image file:

# Analytical expression for the transmission probability:

# In[9]:

def transmission_exact(a, E):
    assert 0 < E < 1
    kappa = sqrt(2 * (1-E))
    return 1. / (1 + sinh(kappa*a)**2 / (4. * E * (1-E)))


##### Convergence to the Analytical Value

# In[10]:

a = 2.5
V = rect_potential(a=a)
Es = [0.71, 0.91]
Ts_exact = [transmission_exact(a=a, E=E) for E in Es]
step_sizes = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]

plt.title('Convergence of Transmission Coefficient for a = %f and E = %s' % (a, ', '.join(map(str, Es))))
plt.xlabel('$\Delta x$')
plt.ylabel('$T$')
plt.ylim((0, 0.2))
plt.yticks(Ts_exact)
for E, T_exact in zip(Es, Ts_exact):
    # show theoretical value as horizontal line
    plt.axhline(T_exact, color='gray')

    # plot numerical data points
    Ts_approx = [scatter(V=V, E=E, x_min=0, x_max=a, n=int(np.ceil(a / dx)))[0] for dx in step_sizes]
    plt.semilogx(step_sizes, Ts_approx, 'o')
plt.show()


# Out[10]:

# image file:

##### Dependence on Energy

# In[11]:

as_ = np.linspace(0.5, 5, 10)
colors = cm.rainbow(1 - as_ / 5)
Es = np.linspace(.01, .99)
Es_coarse = np.arange(.01, 1, .05)
n = 1e3

plt.title('Transmission Coefficient as a Function of Energy')
plt.xlabel('$E$')
plt.ylabel('$T$')
plt.xlim(0, 1.3)
for a, color in zip(as_, colors):
    # plot theoretical value
    plt.plot(Es, [transmission_exact(a=a, E=E) for E in Es], color=color, label='a = %.1f' % a)

    # plot numerical data points
    Ts_approx = [scatter(V=rect_potential(a), E=E, x_min=0, x_max=a, n=n)[0] for E in Es_coarse]
    l = plt.plot(Es_coarse, Ts_approx, 'o', color=color)
plt.legend(loc='upper right')
plt.show()


# Out[11]:

# image file:

##### Dependence on Barrier Width

# In[12]:

E = .96
as_ = np.linspace(.01, 5)
as_coarse = np.linspace(0.5, 5, 10)
n = 1e3

plt.title('Transmission Coefficient as a Function of the Barrier Width for Fixed Energy $E = %s$' % E)
plt.xlabel('$a$')
plt.ylabel('$T$')
plt.ylim(0, 1)
plt.plot(as_, [transmission_exact(a=a, E=E) for a in as_])
plt.plot(as_coarse, [scatter(V=rect_potential(a=a), E=E, x_min=0, x_max=a, n=n)[0] for a in as_coarse], 'o')
plt.show()


# Out[12]:

# image file:

### Parabolic Potential

# In[13]:

def parabolic_potential(a):
    a = float(a)
    def V(x):
        if 0 < x <= a:
            return 4. * (x / a  - x**2 / a**2)
        return 0.
    return V

xs = np.linspace(-0.5, 5.5, num=100)
plt.figure(figsize=(4, 3))
plt.ylim(0, 1.1)
plt.plot(xs, [rect_potential(a=5)(x) for x in xs], '--')
plt.plot(xs, [parabolic_potential(a=5)(x) for x in xs])
plt.show()


# Out[13]:

# image file:

##### Compare Dependence on Energy

# In[14]:

as_ = np.linspace(0.5, 5, 10)
colors = cm.rainbow(1 - as_ / 5)
Es = np.linspace(.01, .99)
Es_coarse = np.arange(.01, 1, .05)
n = 1e3

plt.title('Transmission Coefficient as a Function of Energy')
plt.xlabel('$E$')
plt.ylabel('$T$')
plt.xlim(0, 1.3)
for a, color in zip(as_, colors):
    # plot theoretical value
    plt.plot(Es, [transmission_exact(a=a, E=E) for E in Es], '--', color=color)

    # plot numerical data points
    Ts_parabolic = [scatter(V=parabolic_potential(a), E=E, x_min=0, x_max=a, n=n)[0] for E in Es_coarse]
    l = plt.plot(Es_coarse, Ts_parabolic, 'o-', color=color, label='a = %.1f' % a)
plt.legend()
plt.show()


# Out[14]:

# image file:

##### Compare Dependence on Barrier Width

# In[15]:

Es = [.51, .71]
colors = ['blue', 'green']
as_ = np.linspace(.01, 5)
as_coarse = np.linspace(0.5, 5, 10)
n = 1e3

plt.title('Transmission Coefficient as a Function of the Barrier Width for Energies E = %s' % ', '.join(map(str, Es)))
plt.xlabel('$a$')
plt.ylabel('$T$')
plt.ylim(0, 1)
for E, color in zip(Es, colors):
    plt.plot(as_, [transmission_exact(a=a, E=E) for a in as_], '--', color=color, label='E = %s (rect)' % E)
    plt.plot(as_coarse, [scatter(V=parabolic_potential(a=a), E=E, x_min=0, x_max=a, n=n)[0] for a in as_coarse], 'o-', label='E = %s (parabolic)' % E)
plt.legend()
plt.show()


# Out[15]:

# image file:
