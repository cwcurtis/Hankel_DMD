import numpy as np


def harmonic(lhs):
    x1, x2 = lhs[0], lhs[1]
    rhs = np.zeros(2, dtype=np.float64)
    rhs[0] = x2
    rhs[1] = -np.sin(x1)
    return rhs


def duffing(lhs):
    x1, x2 = lhs[0], lhs[1]
    rhs = np.zeros(2, dtype=np.float64)
    rhs[0] = x2
    rhs[1] = x1 - x1**3.
    return rhs


def vanderpol(lhs,mu):
    x1, x2 = lhs[0], lhs[1]
    rhs = np.zeros(2, dtype=np.float64)
    rhs[0] = x2
    rhs[1] = mu*(1.-x1**2.)*x2 - x1
    return rhs


def lorentz(lhs,sigma,rval,bval):
    y1, y2, y3 = lhs[0], lhs[1], lhs[2]
    rhs = np.zeros(3, dtype=np.float64)
    rhs[0] = sigma*(y2-y1)
    rhs[1] = rval*y1-y2-y1*y3
    rhs[2] = -bval*y3 + y1*y2
    return rhs


def lorentz_kba(lhs,a,b,F,G):
    y1, y2, y3 = lhs[0], lhs[1], lhs[2]
    rhs = np.zeros(3, dtype=np.float64)
    rhs[0] = -y2**2.-y3**2.-a*(y1-F)
    rhs[1] = y1*y2-b*y1*y3-y2+G
    rhs[2] = b*y1*y2+y1*y3-y3
    return rhs


# 4th order Runge-Kutta timestepper
def rk4(x0, f, dt):
    k1 = dt*f(x0)
    k2 = dt*f(x0 + k1/2.)
    k3 = dt*f(x0 + k2/2.)
    k4 = dt*f(x0 + k3)
    return x0 + (k1 + 2.*k2 + 2.*k3 + k4)/6.


# Time stepping scheme for solving x' = f(x) for t0<=t<=tf with time step dt.
def timestepper(x0,t0,tf,dt,f):
    ndim = np.size(x0)
    nsteps = np.int((tf-t0)/dt)
    solpath = np.zeros((ndim,nsteps+1),dtype=np.float64)
    solpath[:,0] = x0
    for jj in range(nsteps):
        solpath[:, jj+1] = rk4(solpath[:, jj], f, dt)
    return solpath


def lorenz_lyupanov_solver(yt, sigma, rval, bval, dt):
    NT = np.shape(yt)[1]
    gs_rem_terms = np.ones((3, NT), dtype=np.float64)
    jac_mat = np.zeros((3, 3), dtype=np.float64)
    jac_mat[0, 0] = -sigma
    jac_mat[0, 1] = sigma
    jac_mat[1, 1] = -1.
    jac_mat[2, 2] = -bval

    uprior = np.eye(3, dtype=np.float64)

    for jj in range(NT-1):
        jac_mat[1, 0] = rval-yt[2, jj]
        jac_mat[1, 2] = -yt[0, jj]
        jac_mat[2, 0] = yt[1, jj]
        jac_mat[2, 1] = yt[0, jj]

        jupdate = lambda uvec: jac_mat @ uvec
        unext = np.zeros((3, 3), dtype=np.float64)

        for ll in range(3):
            unext[:, ll] = rk4(uprior[:, ll], jupdate, dt)

        uprior, rnext = np.linalg.qr(unext)
        gs_rem_terms[:, jj+1] = np.diag(rnext)

    return np.sum(np.log(np.abs(gs_rem_terms)), 1)/(NT*dt)