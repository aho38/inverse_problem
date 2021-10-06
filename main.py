from kcoef_utils import *
import numpy as np

T = 1.0
L = 1.0
k = 0.005

nx = 20
nt = 100

noise_std_dev = 0.

h = L/float(nx)
dt = T/float(nt)

x = np.linspace(0. + h, L-h, nx-1)
m_true = 0.5 - np.abs(x-0.5)
m = m_true

T = np.zeros((nx-1, nt+1))
T[...,0] = m_true

for i in range(nt):
    T_i = stepFwd(m, k, h, dt, nx-1, 1)
    m = T_i
    T[:,i+1] = T_i


T_final = T[..., -1].copy()

true_intdt, int_dTdx = solveINVFWD(T_final, k, h, nx-1)

out = sp.linalg.spsolve(int_dTdx, true_intdt)

print('done')