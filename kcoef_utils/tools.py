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
    diagonals[0,:] = -1.0/h**2
    diagonals[1,:] =  2.0/h**2
    diagonals[2,:] = -1.0/h**2
    K = k*sp.spdiags(diagonals, [-1,0,1], n,n)
    M = sp.spdiags(np.ones(n), 0, n,n)
    
    return M + dt*K
    

def stepFwd(m, k, h, dt, n, nt):
    '''
    m - initial condition
    k - diffusion coefficient
    h - spacing
    dt - time step
    nt - number of timesteps
    '''
    A = assembleMatrix(k, h, dt, n)
    u_old = m.copy()
    for i in np.arange(nt):
        u = la.spsolve(A, u_old)
        u_old[:] = u
        
    return u    
    
def assembleINVMatrix(T, h, n):
    '''

    get matrix diag(k * dT/dx_i)

    T - temperature
    h - spacing of your mesh/grid
    n  - number of interior points
    '''
    diagonals = np.zeros((2,n))
    diagonals[0, :] = -1.0/h
    diagonals[1, :] =  1.0/h

    dTdx = sp.spdiags(diagonals, [0,1], n, n) * T
    dTdx = sp.spdiags(dTdx, [0], n, n)
    return dTdx
    

def solveINVFWD(T, k, h, n):
    '''
    m - initial condition
    k - diffusion coefficient
    h - spacing
    dt - time step
    nt - number of timesteps
    '''
    dTdx = assembleINVMatrix(T, h, n)

    int_dTdx = dTdx * (k * np.ones(n))
        
    return int_dTdx, dTdx

def computeEigendecomposition(k, h, dt, n, nt):
    ## Compute F as a dense matrix
    F = np.zeros((n,n))
    m_i = np.zeros(n)
    
    for i in np.arange(n):
        m_i[i] = 1.0
        F[:,i] = stepFwd(m_i, k, h, dt, n, nt)
        m_i[i] = 0.0
    
    ## solve the eigenvalue problem
    lmbda, U = np.linalg.eigh(F)
    ## sort eigenpairs in decreasing order
    lmbda[:] = lmbda[::-1]
    lmbda[lmbda < 0.] = 0.0
    U[:] = U[:,::-1]
    
    return lmbda, U 

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

    def solveINVFWD(self, T):
        '''
        T - temperature data, (a vector)
        '''
        n = self.nx - 1
        dTdx = assembleINVMatrix(T, self.h, n)

        int_dTdx = dTdx * (self.k * np.ones(n))
            
        return int_dTdx, dTdx

    def solveTikhonov(self, d, F, alpha):    
        H = np.dot( F.transpose(), F) + alpha*np.identity(F.shape[1])
        rhs = np.dot( F.transpose(), d)
        return np.linalg.solve(H, rhs)
