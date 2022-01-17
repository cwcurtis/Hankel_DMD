import numpy as np
from scipy.linalg import toeplitz as toep

def dmd_cmp(gtot, thrshhld, window, ndsets):
    nrws, nclmns = gtot.shape
    gm = np.zeros((nrws, ndsets * (window - 1)), dtype=np.float64)
    gp = np.zeros((nrws, ndsets * (window - 1)), dtype=np.float64)
    # Perform DMD method.  Note, we need to be careful about how we break the concantenated Hankel matrix apart.
    for ll in range(ndsets):
        gm[:, ll * (window - 1):(ll + 1) * (window - 1)] = gtot[:, ll * window:(ll + 1) * window - 1]
        gp[:, ll * (window - 1):(ll + 1) * (window - 1)] = gtot[:, 1 + ll * window:(ll + 1) * window]

    u, s, vh = np.linalg.svd(gm, full_matrices=False)
    sm = np.max(s)
    indskp = np.log10(s / sm) > -thrshhld
    sr = s[indskp]
    ur = u[:, indskp]
    v = np.conj(vh.T)
    vr = v[:, indskp]
    kmat = gp @ vr @ np.diag(1. / sr) @ np.conj(ur.T)
    evls, evcs = np.linalg.eig(kmat)
    phim = np.linalg.solve(evcs, gm)
    return evls, phim, evcs


def hankel_matrix(tseries, window):
    NT = np.size(tseries)
    nobserves = NT - (window - 1)
    tcol = tseries[:nobserves]
    trow = tseries[(nobserves - 1):]
    hmat = np.flipud(toep(tcol[::-1], trow))
    hmatt = hmat.T
    sclfac = np.linalg.norm(hmatt[:, -1])
    return hmat, sclfac


def hankel_dmd(rawdata, obs, window, thrshhld):
    NT = np.shape(rawdata[0])[1]
    nclmns = NT - (window - 1)
    nobs = len(obs)
    ndsets = len(rawdata)

    hankel_mats = np.zeros((nclmns * nobs, window * ndsets), dtype=np.float64)
    for ll in range(ndsets):
        for jj in range(nobs):
            tseries = obs[jj](rawdata[ll])
            hmat, sclfac = hankel_matrix(tseries, window)
            if jj == 0:
                usclfac = sclfac
            hankel_mats[jj * nclmns:(jj + 1) * nclmns, ll * window:(ll + 1) * window] = usclfac / sclfac * hmat
    return dmd_cmp(hankel_mats, thrshhld, window, ndsets)


def path_reconstruction(phim, window, initconds):

    phimat = phim[:, ::(window - 1)]
    u, s, vh = np.linalg.svd(phimat.T, full_matrices=False)
    kmat = np.conj(vh.T) @ np.diag(1. / s) @ np.conj(u.T) @ initconds
    recon = np.real(kmat.T @ phim)
    return recon


def path_test(query_pts, initconds, phim, evls, window):
    Nobs = np.shape(phim)[0]
    Nqp = np.shape(query_pts)[0]
    Ns = np.shape(query_pts)[1]
    numiconds = int(np.shape(phim)[1] / (window - 1))
    iconphim = np.zeros((Nobs, numiconds), dtype=np.complex128)

    for ll in range(numiconds):
        iconphim[:, ll] = phim[:, ll * (window - 1)]

    u, s, vh = np.linalg.svd(iconphim, full_matrices=False)
    Kmat = (initconds.T) @ (np.conj(vh)).T @ np.diag(1. / s) @ (np.conj(u)).T
    err = np.linalg.norm(initconds.T - Kmat @ iconphim)
    print("Error in first fit: %1.2e" % err)

    uk, sk, vhk = np.linalg.svd(Kmat, full_matrices=False)
    phi_query = np.conj(vhk).T @ np.diag(1. / sk) @ np.conj(uk).T @ query_pts.T
    err = np.linalg.norm(query_pts.T - Kmat @ phi_query)
    print("Error in second fit: %1.2e" % err)

    test_paths = np.zeros((Nqp, Ns, window-1), dtype=np.float64)
    test_paths[:, :, 0] = query_pts
    eveciter = evls
    for ll in range(1, window-1):
        test_paths[:, :, ll] = np.real((Kmat @ np.diag(eveciter) @ phi_query)).T
        eveciter = eveciter * evls
        print(np.abs(eveciter))
    return test_paths