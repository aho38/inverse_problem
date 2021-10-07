import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import matplotlib.pyplot as plt

def plot(f, L, nx, style="-b", **kwargs):
    '''
    f - y value of the plot
    L - domain range [0, L]
    nx - number of x value
    style - line type and color of the plot
    kwargs - any key words arguments for plt.plot
    '''

    x = np.linspace(0., L, nx+1)
    f_plot = np.zeros_like(x)
    f_plot[1:-1] = f
    plt.plot(x,f_plot, style, **kwargs)

def assembleMatrix(k, h, dt, n):
    '''

    get matrix (I + dt * K)

    k - diffusion coefficient
    h - spacing of your mesh/grid
    dt - time step
    n - number of interior points
    '''
    diagonals = np.zeros((3, n))   # 3 diagonals
    diagonals[0,:] =  1.0/h**2
    diagonals[1,:] = -2.0/h**2
    diagonals[2,:] =  1.0/h**2
    K = k*sp.spdiags(diagonals, [-1,0,1], n,n)
    M = sp.spdiags(np.ones(n), 0, n,n)
    
    return M - dt*K

class inv_solver:
    
    def __init__(self, T, L, k, nx, nt, m0):
        super(inv_solver, self).__init__()

        self.T = T
        self.L = L
        self.k = k
        self.nx= nx
        self.nt= nt

        self.h = L/float(nx)
        self.dt= T/float(nt)

        self.temps = np.zeros((nx-1, nt+1))
        self.m0 = m0
        self.temps[..., 0] = m0
    
    def stepFWD(self, m, n):
        '''

            dT/dt = k * d^2 T/ dx^2

        m - state condition
        n - number of step 

        central diff for space
        forward euler for time
        '''
        A = assembleMatrix(self.k, self.h, self.dt, self.nx - 1)
        u_old = m.copy()
        for i in np.arange(n):
            u = la.spsolve(A, u_old)
            u_old[:] = u
            
        return u

    def FWDsolve(self):
        m = self.m0
        for i in range(self.nt):
            T_i = self.stepFWD(m, 1)
            m = T_i
            self.temps[...,i+1] = T_i

    def int_dTdt(self, dTdt, i):
        '''
        int_-D^z dT/dt = k dT

        Reimann Summ

        T - temperature data, (a vector)
        i - the index number of the grid which correspond to heigh z we're integrating to
        '''
        if i >= self.nx:
            raise Exception("grid index out of range. Value should be between 0 and {3:}".format(self.nx-1))

        int_dTdx = np.sum(dTdt[0:i]) * self.h
            
        return int_dTdx

    def solveTikhonov(self, d, F, alpha):    
        H = np.dot( F.transpose(), F) + alpha*np.identity(F.shape[1])
        rhs = np.dot( F.transpose(), d)
        return np.linalg.solve(H, rhs)
