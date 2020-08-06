import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def varied_eta(L, eps, seed, show_corr_t=True, show_EA_OP=True):
    """ Plot EA OP vs eta at fixed L and eps.
        seed = 30370000 for L == 512
        seed = 20200725 for L == 256
    """
    def get_EA_OP(L, eta, eps, seed, ncut):
        if L == 512:
            if eps == 0.06:
                if eta == 0.18:
                    theta0 = 60
                else:
                    theta0 = 0
                twin = twin0 = 1000
            folder = "L=%d_seed=%d/field_%d/EA_OP" % (L, seed, twin0)
        elif L == 256:
            twin = twin0 = 1000
            folder = "samples/EA_OP"
            theta0 = 0
        fin = "%s/%d_%.3f_%.3f_%d_%d_%d_%03d.npz" % (folder, L, eta, eps,
                                                     twin0, twin, seed, theta0)
        data = np.load(fin)
        time_corr = data["time_serials"]
        t = np.arange(time_corr.size) * twin0
        EA_OP = np.mean(time_corr[ncut:])
        data.close()
        return EA_OP, t, time_corr

    if L == 512:
        if eps == 0.06:
            eta_arr = np.array([
                0.05, 0.10, 0.15, 0.18, 0.2, 0.22, 0.25, 0.3, 0.35, 0.4, 0.45
            ])
    elif L == 256:
        if eps in [0.06, 0.08, 0.12, 0.14]:
            eta_arr = np.array([
                0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.18, 0.2, 0.225, 0.25,
                0.275, 0.3, 0.35, 0.4, 0.45
            ])
        elif eps in [0.05, 0.055]:
            eta_arr = np.array([
                0.05, 0.075, 0.1, 0.125, 0.15, 0.18, 0.2, 0.225, 0.25, 0.275,
                0.3, 0.35, 0.4, 0.45
            ])
        elif eps == 0.10:
            eta_arr = np.array([
                0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.225, 0.25, 0.275,
                0.3, 0.35, 0.4, 0.45
            ])
    EA_OP = np.zeros_like(eta_arr)
    for i, eta in enumerate(eta_arr):
        EA_OP[i], t, time_corr = get_EA_OP(L, eta, eps, seed, -50)
        if show_corr_t:
            plt.loglog(t, time_corr, label="%g" % eta)
    if show_corr_t:
        plt.title(r"$L=%d, \epsilon=%g, {\rm seed=%d}$" % (L, eps, seed))
        plt.legend()
        plt.xscale("log")
        plt.yscale("log")
        plt.show()
        plt.close()

    if show_EA_OP:
        plt.plot(eta_arr, EA_OP, "-o")
        plt.xlabel(r"$\eta$", fontsize="large")
        plt.ylabel(r"$Q_{\rm EA}$", fontsize="large")
        plt.title(r"$L=%d, \epsilon=%g$" % (L, eps), fontsize="x-large")
        plt.tight_layout()
        plt.show()
        plt.close()
    else:
        return eta_arr, EA_OP


def plot_EA_OP_eta_eps():
    L = 256
    eps_arr = [0.055, 0.06, 0.08, 0.1, 0.12]
    seed = 20200725
    for eps in eps_arr:
        eta_arr, EA_OP = varied_eta(L, eps, seed, False, False)
        plt.plot(eta_arr, EA_OP, "-o", label="%.3f" % eps)
    plt.legend(title=r"$\epsilon=$")
    # plt.yscale("log")
    plt.show()
    plt.close()


def varied_eps(L=512, eta=0.18, seed=30370000):
    if L == 512:
        if eta == 0.45:
            eps_arr = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
            theta0 = 0
        elif eta == 0.18:
            eps_arr = np.array([
                0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.055, 0.06, 0.1,
                0.15, 0.2, 0.25, 0.3
            ])
            theta0 = 60
        twin0 = twin = 1000
        print(eps_arr)
    elif L == 256:
        if seed == 30370000:
            eps_arr = np.array([
                0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.035, 0.04,
                0.045, 0.05, 0.055, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35
            ])
            theta0 = 0
        twin0 = twin = 200
    EA_OP = np.zeros_like(eps_arr)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 5))
    for i, eps in enumerate(eps_arr):
        ncut = -10
        if L == 512:
            if eps in [0.01, 0.02, 0.03, 0.04, 0.01, 0.15, 0.25]:
                twin0 = twin = 200
                theta0 = 0
            else:
                twin0 = twin = 1000
                theta0 = 60
        folder = "L=%d_seed=%d/field_%d/EA_OP" % (L, seed, twin0)
        fin = "%s/%d_%.3f_%.3f_%d_%d_%d_%03d.npz" % (folder, L, eta, eps,
                                                     twin0, twin, seed, theta0)
        data = np.load(fin)
        serials = data["time_serials"]
        EA_OP[i] = np.mean(serials[ncut:])
        print(serials.size)
        t = np.arange(serials.size) * twin0
        if eps in [
                0.3, 0.2, 0.1, 0.06, 0.055, 0.05, 0.25, 0.001, 0.005, 0.03,
                0.04
        ]:
            ax1.loglog(t, serials, "-o", label="%.3f" % eps, ms=2)
    ax1.legend(title=r"$\epsilon=$")

    ax2.plot(eps_arr, EA_OP, "-o")
    ax2.set_yscale("log")
    ax1.set_xlabel(r"$T$", fontsize="large")
    ax1.set_ylabel(r"$Q(T)$", fontsize="large")
    ax2.set_ylabel(r"$Q_{\rm EA}$", fontsize="large")
    ax2.set_xlabel(r"$\epsilon$", fontsize="large")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(r"$L=%d, \eta=%g$" % (L, eta), fontsize="x-large", y=0.995)
    plt.show()


def varied_seed(L=256, eta=0.18, t_win0=200, t_win=200, theta0=0):
    seed_arr = np.array([30370000, 30370001, 30370002])
    eps_arr = np.array([0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3])
    linestyles = ["-", ":", "--"]
    EA_OP = np.zeros((seed_arr.size + 1, eps_arr.size))
    for i, eps in enumerate(eps_arr):
        for j, seed in enumerate(seed_arr):
            folder = "L=%d_seed=%d/field_%d/EA_OP" % (L, seed, t_win0)
            fin = "%s/%d_%.3f_%.3f_%d_%d_%d_%03d.npz" % (
                folder, L, eta, eps, t_win0, t_win, seed, theta0)
            data = np.load(fin)
            serials = data["time_serials"]
            t = np.arange(serials.size) * t_win0
            EA_OP[j, i] = np.mean(serials[-10:])
            plt.loglog(t, serials, linestyles[j], label="%.3f" % eps)

    L = 512
    seed = 30370000
    for i, eps in enumerate(eps_arr):
        folder = "L=%d_seed=%d/field_%d/EA_OP" % (L, seed, t_win0)
        fin = "%s/%d_%.3f_%.3f_%d_%d_%d_%03d.npz" % (
            folder, L, eta, eps, t_win0, t_win, seed, theta0)
        data = np.load(fin)
        serials = data["time_serials"]
        t = np.arange(serials.size) * t_win0
        EA_OP[3, i] = np.mean(serials[-10:])
        plt.loglog(t, serials, ":")
    plt.show()
    plt.close()

    plt.plot(eps_arr, EA_OP[0], "-o")
    plt.plot(eps_arr, EA_OP[1], "-o")
    plt.plot(eps_arr, EA_OP[2], "-o")
    plt.plot(eps_arr, EA_OP[3], "-s")
    plt.show()
    plt.close()


def varied_replica():
    L = 512
    eta = 0.18
    seed = 30370000
    eps_arr = np.array([0.05, 0.055, 0.06, 0.1, 0.2, 0.3])
    EA_OP = np.zeros((2, eps_arr.size))
    theta0 = 0
    t_win0 = t_win = 200
    for i, eps in enumerate(eps_arr):
        folder = "L=%d_seed=%d/field_%d/EA_OP" % (L, seed, t_win0)
        fin = "%s/%d_%.3f_%.3f_%d_%d_%d_%03d.npz" % (
            folder, L, eta, eps, t_win0, t_win, seed, theta0)
        data = np.load(fin)
        serials = data["time_serials"]
        t = np.arange(serials.size) * t_win0
        EA_OP[0, i] = np.mean(serials[-10:])
        plt.loglog(t, serials, "-")

    theta0 = 60
    t_win0 = t_win = 1000
    for i, eps in enumerate(eps_arr):
        folder = "L=%d_seed=%d/field_%d/EA_OP" % (L, seed, t_win0)
        fin = "%s/%d_%.3f_%.3f_%d_%d_%d_%03d.npz" % (
            folder, L, eta, eps, t_win0, t_win, seed, theta0)
        data = np.load(fin)
        serials = data["time_serials"]
        t = np.arange(serials.size) * t_win0
        EA_OP[1, i] = np.mean(serials[-10:])
        plt.loglog(t, serials, "-")
    plt.show()
    plt.close()

    plt.loglog(eps_arr, EA_OP[0], "-o")
    plt.loglog(eps_arr, EA_OP[1], "-s")
    plt.show()
    plt.close()


def finite_size_scaling(eta,
                        eps,
                        ncut=-25,
                        show_corr_t=True,
                        show_EA_OP=True,
                        show_SA_serials=False):
    if eta == 0.18:
        L_arr = np.array([64, 128, 256, 512])
    else:
        L_arr = np.array([32, 64, 128, 256])
    EA_OP_mean = np.zeros(L_arr.size)
    EA_OP_std = np.zeros(L_arr.size)
    EA_OP = {i: [] for i in L_arr}
    if show_SA_serials:
        SA_serials = {i: [] for i in L_arr}
    for i, L in enumerate(L_arr):
        files = glob.glob("samples/EA_OP/%d_%.3f_%.3f_1000_1000_*_000.npz" %
                          (L, eta, eps))
        for j, fin in enumerate(files):
            data = np.load(fin)
            serials = data["time_serials"]
            t_arr = np.arange(serials.size) * 1000
            if show_corr_t:
                plt.plot(t_arr, serials)
            if show_SA_serials:
                if j == 0:
                    SA_serials[L] = serials
                else:
                    SA_serials[L] += serials
            EA_OP[L].append(np.mean(serials[ncut:]))
        if show_corr_t:
            plt.yscale("log")
            plt.xscale("log")
            plt.title(r"$L=%d, \eta=%g, \epsilon=%g$" % (L, eta, eps))
            plt.show()
            plt.close()
        if show_SA_serials:
            SA_serials[L] /= (j + 1)
        EA_OP[L] = np.array(EA_OP[L])
        EA_OP_mean[i] = np.mean(EA_OP[L])
        EA_OP_std[i] = np.std(EA_OP[L])
    if show_SA_serials:
        for L in L_arr:
            t_arr = np.arange(SA_serials[L].size) * 1000
            plt.plot(t_arr, SA_serials[L], "-o", label="%d" % L, ms=1)
        plt.legend(title="$L=$", fontsize="large", loc="upper right")
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(ymax=1)
        plt.xlabel(r"$T$", fontsize="x-large")
        plt.ylabel(r"Sample-averaged $Q(T)$", fontsize="x-large")
        plt.title(r"$\eta=%g, \epsilon=%g$" % (eta, eps), fontsize="x-large")
        plt.tight_layout()
        plt.show()
        plt.close()
    if show_EA_OP:
        for L in L_arr:
            for i in range(EA_OP[L].size):
                plt.plot(L,
                         EA_OP[L][i],
                         "s",
                         fillstyle="none",
                         c="grey",
                         ms=3,
                         alpha=0.6)
        plt.plot(L_arr, EA_OP_mean, "o", ms=5, c="tab:red")
        plt.errorbar(L_arr, EA_OP_mean, EA_OP_std, c="tab:red")

        plt.xscale("log")
        # plt.yscale("log")
        plt.xlabel(r"$L$", fontsize="x-large")
        plt.ylabel(r"$Q_{\rm EA}$", fontsize="x-large")
        plt.title(r"$\eta=%g, \epsilon=%g$" % (eta, eps), fontsize="x-large")
        plt.tight_layout()
        plt.show()
        plt.close()
        for i in range(L_arr.size):
            print(L_arr[i], EA_OP_mean[i])
    else:
        return L_arr, EA_OP_mean, EA_OP_std


def show_finite_size_scaling(eps=None, eta=None, ncut=-40):
    if eta is None:
        if eps == 0.1:
            eta_arr = [0.05, 0.1, 0.18, 0.45]
        elif eps == 0.06:
            eta_arr = [0.05, 0.18, 0.45]
        elif eps == 0.:
            eta_arr = [0.45]
        for eta in eta_arr:
            L_arr, EA_OP_mean, EA_OP_std = finite_size_scaling(
                eta, eps, ncut, False, False)
            line, = plt.plot(L_arr, EA_OP_mean, "o", label="%.3f" % eta)
            plt.errorbar(L_arr, EA_OP_mean, EA_OP_std, c=line.get_c())
        fixed_eta = False
    elif eps is None:
        if eta == 0.45:
            eps_arr = [0.01, 0.06]
        for eps in eps_arr:
            L_arr, EA_OP_mean, EA_OP_std = finite_size_scaling(
                eta, eps, ncut, False, False)
            line, = plt.plot(L_arr, EA_OP_mean, "o", label="%.3f" % eps)
            plt.errorbar(L_arr, EA_OP_mean, EA_OP_std, c=line.get_c())
        fixed_eta = True
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$L$", fontsize="x-large")
    plt.ylabel(r"$Q_{\rm EA}$", fontsize="x-large")
    if not fixed_eta:
        plt.title(r"$\epsilon=%g$" % (eps), fontsize="x-large")
        plt.legend(title=r"$\eta=$", fontsize="large")
    else:
        plt.title(r"$\eta=%g$" % eta, fontsize="x-large")
        plt.legend(title=r"$\epsilon=$", fontsize="large")
    plt.tight_layout()
    plt.show()
    plt.close()


def show_EA_OP_PD(L, seed, twin0=1000):
    if L == 512 and seed == 30370000:
        os.chdir("E:/data/random_torque/defects/EA_OP_PD")
    files = glob.glob("*.npz")
    eta, eps, EA_OP = [], [], []
    for i, fin in enumerate(files):
        s = fin.split("_")
        eta_i, eps_i = float(s[1]), float(s[2])
        data = np.load(fin)
        serials = data["time_serials"]
        EA_OP_i = np.mean(serials[-50:])
        if (EA_OP_i > 0):
            eta.append(eta_i)
            eps.append(eps_i)
            EA_OP.append(EA_OP_i)
    plt.scatter(eta, eps, c=EA_OP, cmap="turbo", marker="s")
    plt.xlim(0)
    plt.ylim(ymin=0, ymax=0.15)
    cb = plt.colorbar()
    cb.set_label(r"$Q_{\rm EA}$", fontsize="x-large")
    plt.xlabel(r"$\eta$", fontsize="x-large")
    plt.ylabel(r"$\epsilon$", fontsize="x-large")
    plt.title(r"$L=%d, {\rm seed}=%d$" % (L, seed), fontsize="x-large")
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    os.chdir("E:/data/random_torque/defects")
    # varied_eps(L=256)
    # varied_seed()
    # varied_replica()
    # varied_eta(256, 0.14, 20200725)
    # finite_size_scaling(eta=0.45, eps=0.01, ncut=-50, show_SA_serials=True)
    # show_finite_size_scaling(eta=0.45, ncut=-50)
    # plot_EA_OP_eta_eps()
    show_EA_OP_PD(512, 30370000)
