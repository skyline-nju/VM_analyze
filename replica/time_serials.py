import glob
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

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
        new_seeds2 = [
            25750000, 25750001, 12340000, 12340001, 12340002, 12340003,
            12340004, 12340005, 12340006, 12340007, 12340008, 12340009
        ]
        prefix = r"E:\data\random_torque\replica"
        if ic == "rand":
            data_dir = r"%s\ini_rand" % (prefix)
            if seed in new_seeds2:
                data_dir = r"E:\data\random_torque\replica2" +\
                    r"\ini_rand"
        elif ic == "ordered":
            new_seeds = [30370052]
            if seed in new_seeds:
                data_dir = r"%s\ini_ordered_new" % prefix
            elif seed in new_seeds2:
                data_dir = r"E:\data\random_torque\replica2" +\
                    r"\ini_ordered"
            else:
                data_dir = r"%s\ini_ordered" % prefix
        elif ic == "ordered_sym":
            data_dir = r"%s\ini_ordered_sym" % prefix
        elif ic == "ordered_asym":
            data_dir = r"%s\ini_ordered_asym" % prefix
        elif ic == "ordered_diag":
            data_dir = r"%s\ini_ordered_diag" % prefix
        pat = "p%d.%g.%g.%d.*.dat" % (L, eta * 1000, eps * 1000, seed)
    else:
        data_dir = r"E:\data\random_field\normalize_new\time_ave_image"
        if ic == "ordered":
            data_dir += "\\ini_ordered"
        pat = "phi_rf_%d_%g_%g_%d_*.dat" % (L, eta, eps, seed)
    files = glob.glob("%s\\%s" % (data_dir, pat))
    return files


def shift(theta, seed, ic, i):
    if ic == "rand":
        if seed == 30370000:
            if eps == 0.035 and (i == 1 or i == 2):
                theta += 2 * np.pi
            if eps == 0.045:
                if i == 5:
                    theta -= 2 * np.pi
                elif i in [0, 1, 2, 6, 7, 8]:
                    theta += 2 * np.pi
            if eps == 0.02 and (i == 5):
                theta -= 2 * np.pi
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
        elif seed == 35490010:
            if eps == 0.035 and i == 2:
                theta -= 2 * np.pi
        elif seed == 27810010:
            if eps == 0.035 and (i == 4 or i == 5):
                theta += 2 * np.pi
        elif seed == 27810000:
            if i == 10 or i == 11:
                theta += np.pi * 2
        elif seed == 2021324:
            if i == 2 or i == 3:
                theta += np.pi * 2
        elif seed == 20201326:
            if i == 2:
                theta += np.pi * 2
        elif seed == 20203261:
            if i == 2:
                theta += np.pi * 2
        elif seed == 20203262:
            if i == 3:
                theta += np.pi * 2
        elif seed == 20210406 or seed == 20210407:
            if i == 2:
                theta += np.pi * 2
        elif seed == 20200406:
            if i == 3 or i == 4:
                theta += np.pi * 2
        elif seed == 20200407:
            if i == 2:
                theta -= np.pi * 2
    elif ic == "ordered_sym":
        if seed == 25650020:
            if i == 2:
                theta += np.pi * 2
        elif seed == 25650021:
            if i == 3:
                theta += np.pi * 2
        elif seed == 2021324:
            if i == 4:
                theta += np.pi * 2
    elif ic == "ordered_asym":
        if seed == 2021324:
            if i == 3:
                theta += np.pi * 2
        elif seed == 2021334:
            if i == 1:
                theta -= np.pi * 2
    elif ic == "ordered_diag":
        if seed == 25650014:
            if i == 2:
                theta += np.pi * 2


def plot_phi_theta(L,
                   eps,
                   eta=0.18,
                   disorder_t="RT",
                   seed=30370000,
                   ic="rand",
                   sample="A"):
    def set_ylim(ax1, ax2):
        if disorder_t == "RT":
            if L == 512:
                if eps == 0.035:
                    if ic == "rand" and seed == 30370000:
                        ax1.set_ylim(-0.5, 1.0)
                    elif ic == "rand" and seed == 30370010:
                        ax1.set_ylim(-1.0, 0.8)
                        pass
                    ax2.set_ylim(0.6, 0.8)
            if L == 1024:
                if eps == 0.035:
                    if seed == 35490000:
                        ax1.set_ylim(-1.0, 0.6)
                    ax2.set_ylim(0.65, 0.78)
            if L == 256:
                if eps == 0.035:
                    if seed == 27810000:
                        ax1.set_ylim(-0.7, 1)
                    ax2.set_ylim(0.7, 0.82)
            if L == 4096:
                ax1.set_xlim(0, 1.2e6)
        else:
            pass

    files = get_matched_files(L, eps, eta, disorder_t, seed, ic)
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(9, 6), sharex=True)
    beg = 0
    for i, f in enumerate(files):
        print(f)
        phi, theta = read_phi_theta(f, beg)
        # theta = untangle(theta)
        shift(theta, seed, ic, i)
        x = (np.arange(phi.size) + beg + 1) * 100
        ncut = 5000
        theta_m = np.mean(theta[ncut:]) / np.pi
        print("theta_m =", theta_m)
        label = r"replica %d" % i
        ax1.plot(x, theta / np.pi, label=label)
        m_mean = np.mean(phi[ncut:])
        m_var = np.var(phi[ncut:])

        # label = r"$\langle m\rangle=%.4f," \
        #     r"\langle m^2\rangle - \langle m\rangle^2=%.6f$"
        label = r"$%.4f, %.6f$"
        ax2.plot(x, phi, lw=0.5, label=label % (m_mean, m_var))
    ax1.set_xlim(0)
    ax2.set_xlim(0)
    set_ylim(ax1, ax2)
    # ax2.set_ylim(ymax=0.75)
    ax1.set_ylabel(r"$\theta/\pi$")
    ax2.set_ylabel(r"$m$")
    ax2.set_xlabel(r"$t$")
    # ax1.legend()
    # ax2.legend(
    #     loc="lower right",
    #     ncol=4,
    #     title=r"$\langle m\rangle, \langle m^2\rangle - \langle m\rangle^2=$")
    plt.tight_layout(rect=[-0.01, -0.025, 1.01, 0.98])
    if disorder_t == "RT":
        title = r"RS: $L=%d, \eta=%g, \epsilon=%g$" % (L, eta, eps)
    else:
        title = r"RF: $L=%d, \eta=%g, \epsilon=%g$" % (L, eta, eps)
    title += ", Sample %s" % sample
    plt.suptitle(title, fontsize="x-large", y=0.998)
    plt.show()
    plt.close()


def plot_theta(L, eps, eta=0.18, disorder_t="RT", seed=30370000, ic="rand"):
    files = get_matched_files(L, eps, eta, disorder_t, seed, ic)
    # fig, ax = plt.subplots(figsize=(9, 3), constrained_layout=True, projection="polar")
    fig = plt.figure(figsize=(5, 5), constrained_layout=True)
    ax = plt.subplot(111, projection="polar")
    beg = 0
    for i, f in enumerate(files):
        phi, theta = read_phi_theta(f, beg)
        theta0 = int(os.path.basename(f).rstrip(".dat").split(".")[-1])
        theta = untangle(theta)
        shift(theta, seed, ic, i)
        x = (np.arange(phi.size) + beg + 1) * 100
        if ic == "rand":
            label = "%d" % i
        else:
            label = r"$%d\degree$" % theta0
        ax.plot(theta, x, label=label)
        print(np.mean(theta[3000:] / np.pi))
    # ax.set_xlim(0)
    # ax.set_ylim(-1, 1.3)
    # ax.set_xscale("log")
    ax.set_xlabel(r"$t$", fontsize="large")
    ax.set_ylabel(r"$\theta$", fontsize="large")
    if ic == "rand":
        plt.legend(title="replica", loc="lower right", ncol=2)
    else:
        plt.legend(title=r"$\theta_0=$", loc="lower left", ncol=2)
    # if disorder_t == "RT":
    #     title = r"RS: $L=%d, \eta=%g, \epsilon=%g$" % (L, eta, eps)
    # else:
    #     title = r"RF: $L=%d, \eta=%g, \epsilon=%g$" % (L, eta, eps)
    # title += r", Sample A"
    title = r"$\epsilon=%g$" % eps
    plt.title(title, fontsize="x-large", y=0.998, color="r")
    plt.show()
    plt.close()


if __name__ == "__main__":
    disorder_t = "RT"
    L = 512
    if disorder_t == "RT":
        eps = 0.035
    else:
        eps = 0.08
    ic = "ordered"
    eta = 0.18
    # if L == 2048, seed = 2020324, 2021324, 202234
    # if L == 4096, seed = 20201326, 20203261, 20203262
    # seed = 3037000
    seed = 30370010
    # plot_phi_theta(
    #     L, eps, eta=eta, disorder_t=disorder_t, seed=seed, ic=ic, sample="C")
    plot_theta(L, eps, eta=eta, disorder_t=disorder_t, ic=ic, seed=seed)
