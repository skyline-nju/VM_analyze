import numpy as np
from scipy import fft


def auto_corr(f, zero_pad=True):
    if f.ndim == 1:
        n = f.size
        if zero_pad:
            f_new = np.zeros(n + n // 2)
            f_new[:n] = f
            F = fft.rfft(f_new)
            norm = np.arange(n, n - n // 2, -1)
        else:
            F = fft.rfft(f)
            norm = n
        h = fft.irfft(F * F.conj())[:n // 2] / norm
        return h
    elif f.ndim == 3:
        nt, nx, ny = f.shape
        if zero_pad:
            f_new = np.zeros((nt + nt // 2, nx, ny), np.float32)
            f_new[:nt] = f
            F = fft.rfft(f_new, axis=0)
            norm = np.arange(nt, nt - nt // 2, -1)
        else:
            F = fft.rfft(f, axis=0)
            norm = nt
        h = fft.irfft(F * F.conj(), axis=0)[:nt // 2]
        h_m = np.mean(h, axis=(1, 2)) / norm
        h_last = h[-1].copy() / norm[-1]
        return h_m, h_last


def get_norm(n, m):
    return np.arange(n, n - m, -1)


def cal_EA_OP(vx, vy):
    corr_vx_t, corr_vx_r = auto_corr(vx)
    corr_vy_t, corr_vy_r = auto_corr(vy)
    EA_OP_t = corr_vx_t + corr_vy_t
    EA_OP_r = corr_vx_r + corr_vy_r
    return EA_OP_t, EA_OP_r
