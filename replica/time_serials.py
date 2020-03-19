import glob
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
try:
    from suscepbility.theta import read_phi_theta, untangle
except ImportError:
    print("error when import add_line")


def get_matched_files(L,
                      eps,
                      eta=0.18,
                      disorder_t="RT",
                      seed=30370000,
                      ic="rand"):
    if disorder_t == "RT":
        data_dir = r"D:\data\VM2d\random_torque\time_ave_image"
        if ic == "ordered":
            data_dir += "\\ini_ordered"
        pat = "p%d.%g.%g.%d.*.dat" % (L, eta * 1000, eps * 1000, seed)
        print(pat)
    else:
        data_dir = r"D:\data\VM2d\random_field\time_ave_image"
        if ic == "ordered":
            data_dir += "\\ini_ordered"
        pat = "phi_rf_%d_%g_%g_%d_*.dat" % (L, eta, eps, seed)
    files = glob.glob("%s\\%s" % (data_dir, pat))
    return files


def plot_phi_theta(L, eps, eta=0.18, disorder_t="RT", seed=30370000,
                   ic="rand"):
    files = get_matched_files(L, eps, eta, disorder_t, seed, ic)
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(9, 6), constrained_layout=True, sharex=True)
    beg = 0
    for i, f in enumerate(files):
        phi, theta = read_phi_theta(f, beg)
        theta = untangle(theta)
        if eps == 0.035 and (i == 1 or i == 2):
            theta += 2 * np.pi
        x = (np.arange(phi.size) + beg + 1) * 100
        ax1.plot(x, theta / np.pi)
        m_mean = np.mean(phi[3000:])
        m_var = np.var(phi[3000:])

        label = r"$\langle m\rangle=%.4f," \
            r"\langle m^2\rangle - \langle m\rangle^2=%.6f$"
        ax2.plot(x, phi, lw=0.4, label=label % (m_mean, m_var))
    ax1.set_xlim(0)
    ax2.set_xlim(0)
    if disorder_t == "RT":
        if eps == 0.035:
            ax2.set_ylim(0.6, 0.8)
        elif eps == 0.03:
            ax2.set_ylim(0.7, 0.82)
        if eps == 0.035:
            ax1.set_ylim(-0.4, 1.2)
    else:
        if eps == 0.08:
            ax1.set_ylim(-1, 1)
            ax2.set_ylim(0.7, 0.81)
    ax1.set_ylabel(r"$\theta/\pi$")
    ax2.set_ylabel(r"$m$")
    ax2.set_xlabel(r"$t$")
    ax2.legend(loc="best", ncol=2)
    if disorder_t == "RT":
        title = r"RS: $L=%d, \eta=%g, \epsilon=%g$" % (L, eta, eps)
    else:
        title = r"RF: $L=%d, \eta=%g, \epsilon=%g$" % (L, eta, eps)
    plt.suptitle(title)
    plt.show()
    plt.close()


def plot_theta(L, eps, eta=0.18, disorder_t="RT", seed=30370000, ic="rand"):
    def shift(theta):
        if ic == "rand":
            if seed == 30370000:
                if eps == 0.035 and (i == 1 or i == 2):
                    theta += 2 * np.pi
            elif seed == 30370010:
                if eps == 0.035:
                    if i == 2:
                        theta += 2 * np.pi
                    elif i == 4 or i == 5:
                        theta -= 2 * np.pi
        elif ic == "ordered":
            if seed == 30370000:
                if eps == 0.035 and i == 2:
                    theta += 2 * np.pi

    files = get_matched_files(L, eps, eta, disorder_t, seed, ic)
    fig, ax = plt.subplots(figsize=(9, 3), constrained_layout=True)
    beg = 0
    for i, f in enumerate(files):
        phi, theta = read_phi_theta(f, beg)
        theta = untangle(theta)
        shift(theta)
        x = (np.arange(phi.size) + beg + 1) * 100
        ax.plot(x, theta / np.pi, label="%d" % i)
        print(np.mean(theta[3000:] / np.pi))
    # ax.set_xlim(0)
    # ax.set_ylim(-1, 1.3)
    # ax.set_xscale("log")
    plt.legend()
    plt.show()
    plt.close()


if __name__ == "__main__":
    disorder_t = "RT"
    L = 512
    if disorder_t == "RT":
        eps = 0.035
    else:
        eps = 0.08
    ic = "rand"
    seed = 30370000
    plot_phi_theta(L, eps, disorder_t=disorder_t, seed=seed, ic=ic)
    # plot_theta(L, eps, disorder_t=disorder_t, ic=ic, seed=seed)
