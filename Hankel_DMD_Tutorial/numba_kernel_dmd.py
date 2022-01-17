import numpy as np
from numba import jit


@jit(nopython=True)
def sinc_kernel(xn, xm, nr, nterms):
    tot = 1.
    tpi = 2.*np.pi
    for jj in range(nr):
        dif = xn[jj] - xm[jj]
        if np.abs(dif) < 1e-15 * np.abs(xm[jj]):
            tot *= (2. * nterms + 1.)
        else:
            tot *= np.sin((nterms + .5) * dif) / np.sin(.5 * dif)
    tot *= 1. / (tpi ** nterms)
    return tot


@jit(nopython=True)
def poly_kernel(xn, xm, nr, nterms):
    tot = 1.
    for ll in range(nr):
        tot += xn[ll]*xm[ll]
    tot = tot**nterms
    return tot


@jit(nopython=True)
def rbf_kernel(xn, xm, nr, sig):
    tot = 0.
    for ll in range(nr):
        tot += (np.abs(xn[ll]-xm[ll]))**2.
    tot = np.exp(-tot/sig**2.)
    return tot


@jit(nopython=True)
def kernel_eval(gm, gp, nr, nc, nterms, amat, gmat):
    for jj in range(nc):
        for kk in range(nc):
            amat[jj, kk] = poly_kernel(gm[:, jj], gp[:, kk], nr, nterms)
            gmat[jj, kk] = poly_kernel(gm[:, jj], gm[:, kk], nr, nterms)
            #amat[jj, kk] = poly_kernel(gm[:, jj], gp[:, kk], nr, nterms) \
            #               + rbf_kernel(gm[:, jj], gp[:, kk], nr, .5)
            #gmat[jj, kk] = poly_kernel(gm[:, jj], gm[:, kk], nr, nterms) \
            #               + rbf_kernel(gm[:, jj], gm[:, kk], nr, .5)
            #amat[jj, kk] = rbf_kernel(gm[:, jj], gp[:, kk], nr, nterms)
            #gmat[jj, kk] = rbf_kernel(gm[:, jj], gm[:, kk], nr, nterms)
            #amat[jj, kk] = sinc_kernel(gm[:, jj], gp[:, kk], nr, nterms)
            #gmat[jj, kk] = sinc_kernel(gm[:, jj], gm[:, kk], nr, nterms)
    return None


def krn_dmd_cmp(gtot, nterms, thrshhld, NT):
    nr, nc = gtot.shape
    nic = int(nc/(NT+1))
    gm = np.zeros((nr, nc-1), dtype=np.float64)
    gp = np.zeros((nr, nc-1), dtype=np.float64)
    for ll in range(nic):
        gm[:, ll*NT:(ll+1)*NT] = gtot[:, ll*(NT+1):(ll+1)*(NT+1)-1]
        gp[:, ll*NT:(ll+1)*NT] = gtot[:, ll*(NT+1)+1:(ll+1)*(NT+1)]
    amat = np.zeros((nc - 1, nc - 1), dtype=np.float64)
    gmat = np.zeros((nc - 1, nc - 1), dtype=np.float64)
    # updates amat and gmat in place
    kernel_eval(gm, gp, nr, nc-1, nterms, amat, gmat)

    q, ssq, qh = np.linalg.svd(gmat, full_matrices=False)
    sr = np.sqrt(ssq)
    srm = np.max(sr)
    indskp = np.log10(sr / srm) > -thrshhld
    srd = sr[indskp]
    qrd = q[:, indskp]
    si = np.diag(1. / srd)
    qs = qrd @ si
    kmat = qs.T @ amat @ qs
    evls, evcs = np.linalg.eig(kmat)

    phim = qrd @ np.diag(srd) @ evcs
    kmodes = (np.linalg.solve(evcs, qs.T @ gm.T)).T

    return evls, phim, kmodes


def dmd_cmp(gtot, thrshhld):
    mrow, ncol = gtot.shape
    # Perform DMD method
    gm = gtot[:, :ncol - 1]
    gp = gtot[:, 1:]

    u, s, vh = np.linalg.svd(gm, full_matrices=False)
    sm = np.max(s)
    indskp = np.log10(s/sm) > -thrshhld
    sr = s[indskp]
    ur = u[:, indskp]
    v = np.conj(vh.T)
    vr = v[:, indskp]
    kmat = gp @ vr @ np.diag(1. / sr) @ np.conj(ur.T)
    evls, evcs = np.linalg.eig(kmat)
    phim = (np.linalg.solve(evcs, gm)).T
    return evls, phim, evcs


def mode_err_cmp(evls, phim):
    nr, nc = phim.shape
    scl_vec = np.sum(np.abs(phim), 0)
    errvec = np.zeros(nc, dtype=np.float64)
    for jj in range(nc):
        errvec[jj] = np.sum(np.abs(phim[1:, jj] - evls[jj]*phim[:(nr-1), jj]))/scl_vec[jj]
    return errvec


def recon_err_cmp(gtot, evls, kmodes, phim):
    nr, nc = phim.shape
    gp = gtot[:, 1:]
    nds, nt = gp.shape
    rcn = np.zeros((nc, nt), dtype=np.complex128)
    for jj in range(nt):
        rcn[:, jj] = evls*phim[jj, :]

    rcn = np.real(kmodes @ rcn)
    err = np.linalg.norm(gp - rcn, ord='fro')/np.linalg.norm(gp, ord='fro')
    print('One-step reconstruction error is: %1.2e' % err)


def dmd_corr_comp(phim, kmodes, shft):
    nt, nmds = np.shape(phim)
    corr_mat = np.zeros((nmds, nmds), dtype=np.complex128)

    kmnrms = np.linalg.norm(kmodes, ord=2, axis=0)
    kmnrlzd = kmodes @ np.diag(1./kmnrms)
    tmat = phim - np.tile(np.mean(phim, 0), (nt, 1))
    vrncs = np.sqrt(np.mean(tmat*np.conj(tmat), 0))
    phimnrmlzd = tmat @ np.diag(1./vrncs)
    for jj in range(nmds):
        for kk in range(nmds):
            kval = np.sum(kmnrlzd[:, jj]*np.conj(kmnrlzd[:, kk]))
            shftvec = np.conj(np.roll(phimnrmlzd[:, kk], shft))

            pval = np.mean(phimnrmlzd[shft:, jj]*shftvec[shft:])
            corr_mat[jj, kk] = kval * pval

    return corr_mat


def kmat_comp(dmd_kmodesred, dmd_evls_red, dmd_totmodes, nds):
    indspos = dmd_evls_red.imag > 0
    md_inds = np.arange(dmd_totmodes)
    indskp = md_inds[indspos]
    dmd_rl_kmodes = dmd_kmodesred[:nds, indskp]
    scfacs = np.linalg.norm(dmd_rl_kmodes, 2, 1)
    dmd_rl_kmodes = np.diag(1./scfacs) @ dmd_rl_kmodes
    kcormat = np.zeros((nds, nds), dtype=np.float64)
    for jj in range(nds):
        for kk in range(jj+1):
            kcormat[jj, kk] = np.abs(np.sum( dmd_rl_kmodes[jj, :] * np.conj(dmd_rl_kmodes[kk, :]) ))
    dk = np.diag(np.diag(kcormat))
    zdk = kcormat - dk
    kcormat = dk + zdk + zdk.T
    return kcormat
