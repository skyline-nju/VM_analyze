''' plot time-averaged density profile for epsilon=0.02 cases. '''

import os
import numpy as np
import matplotlib.pyplot as plt
from axes_zoom_effect import zoom_effect03
from read_npz import read_matched_file, eq_Lx_and_nb


def plot_peak(Lx, nb, eta=350, eps=20):
    """ Plot time-averaged peaks for differernt samples with zoom effect."""

    path = r"G:/data/band/Lx/snapshot/uniband"
    os.chdir(path)

    fig = plt.figure(1, figsize=(8, 5))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(224)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(222)

    ax1.set_xlim(60, 210)
    ax3.set_xlim(190, 200)
    ax3.set_ylim(0.45, 0.49)
    zoom_effect03(ax3, ax1, 190, 200, loc="downward")

    dictLSN = read_matched_file({"Lx": Lx, "eta": eta, "eps": eps})
    phi = np.array(
        [i for i in eq_Lx_and_nb(Lx, nb, "mean_phi", dictLSN=dictLSN)])
    std_gap = np.array(
        [i for i in eq_Lx_and_nb(Lx, nb, "std_gap", dictLSN=dictLSN)])
    color_list = plt.cm.jet(
        [(i - phi.min()) / (phi.max() - phi.min()) for i in phi])

    x = np.arange(Lx) + 0.5
    rho_gas = np.zeros_like(phi)

    for i, peak in enumerate(
            eq_Lx_and_nb(Lx, nb, "ave_peak", dictLSN=dictLSN)):
        rho_gas[i] = np.mean(peak[190:200])
        ax1.plot(x, peak, c=color_list[i], lw=0.8)
        ax3.plot(x, peak, c=color_list[i])
    ax2.axis("auto")
    ax4.axis("auto")
    sca = ax2.scatter(rho_gas, phi, c=phi, cmap="jet")
    ax4.scatter(std_gap, phi, c=phi, cmap="jet")
    z = np.polyfit(rho_gas, phi, 1)
    print(z)

    bbox = dict(edgecolor="k", fill=False)
    ax1.text(0.91, 0.91, "(a)", transform=ax1.transAxes, bbox=bbox)
    ax2.text(0.91, 0.91, "(d)", transform=ax2.transAxes, bbox=bbox)
    ax3.text(0.91, 0.91, "(c)", transform=ax3.transAxes, bbox=bbox)
    ax4.text(0.91, 0.91, "(b)", transform=ax4.transAxes, bbox=bbox)

    ylabel = r"$\langle \overline{\rho}_y (x)\rangle_t$"
    xlabel = r"$x$"
    ax1.text(0.01, 0.92, ylabel, transform=ax1.transAxes)
    ax1.text(0.96, 0.07, xlabel, transform=ax1.transAxes)
    ax2.text(0.01, 0.92, r"$\langle \phi \rangle_t$", transform=ax2.transAxes)
    ax2.text(0.90, 0.02, r"$\rho_{\rm gas}$", transform=ax2.transAxes)
    ax3.text(0.01, 0.92, ylabel, transform=ax3.transAxes)
    ax3.text(0.96, 0.02, xlabel, transform=ax3.transAxes)
    ax4.text(0.01, 0.92, r"$\langle \phi \rangle_t$", transform=ax4.transAxes)
    ax4.text(
        0.78,
        0.02,
        r"$\langle \sigma(\Delta x_b)\rangle_t$",
        transform=ax4.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(r"$\eta=%g,\epsilon=%g, \rho_0=1, L_x=%d, L_y=200, n_b=%d$" %
                 (eta / 1000, eps / 1000, Lx, nb))
    fig.subplots_adjust(right=0.86)
    cbar_ax = fig.add_axes([0.88, 0.08, 0.03, 0.8])
    fig.colorbar(sca, cax=cbar_ax)
    fig.text(0.88, 0.9, r"$\langle \phi \rangle_t$", color="b", fontsize=14)

    plt.show()
    # plt.savefig(
    #     r"E:\report\quenched_disorder\report\fig\ave_peak20.pdf", dpi=300)
    plt.close()


def rescale(Lx, nb, eta=350, eps=20, show=True):
    if 400 <= Lx <= 480:
        os.chdir(r"G:/data/band/Lx/snapshot/uniband")
    else:
        os.chdir(r"G:/data/band/Lx/snapshot/eps20")

    dictLSN = read_matched_file({"Lx": Lx, "eta": eta, "eps": eps})
    phi = np.array(
        [i for i in eq_Lx_and_nb(Lx, nb, "mean_phi", dictLSN=dictLSN)])
    color_list = plt.cm.jet(
        [(i - phi.min()) / (phi.max() - phi.min()) for i in phi])

    x = np.arange(Lx) + 0.5
    rho_gas = np.zeros_like(phi)
    peak_ave = np.zeros_like(x)

    if show:
        fig, ax = plt.subplots(
            nrows=1, ncols=3, figsize=(8, 3.5), constrained_layout=True)

        ax[0].set_xlim(60, 210)
        ax[1].set_xlim(60, 210)
        ax[2].set_xlim(60, 210)
        ax[0].set_xlabel(r"$x$", fontsize="large")
        ax[1].set_xlabel(r"$x$", fontsize="large")
        ax[2].set_xlabel(r"$x$", fontsize="large")
        ax[0].set_ylabel(
            r"$\langle \overline{\rho}_y(x)\rangle_t$", fontsize="large")
        ax[1].set_ylabel(
            r"$\langle \overline{\rho}_y(x)\rangle_t/\rho_{\rm gas}$",
            fontsize="large")
        ax[2].set_ylabel(
            r"$\langle \overline{\rho}_y(x)\rangle_t-\rho_{\rm gas}$",
            fontsize="large")

    for i, peak in enumerate(
            eq_Lx_and_nb(Lx, nb, "ave_peak", dictLSN=dictLSN)):
        rho_gas[i] = np.mean(peak[190:200])
        peak_ave += peak
        if show:
            ax[0].plot(x, peak, c=color_list[i], lw=0.5)
            ax[1].plot(x, peak / rho_gas[i], c=color_list[i], lw=0.5)
            ax[2].plot(x, peak - rho_gas[i], c=color_list[i], lw=0.5)
    if show:
        plt.suptitle(
            r"$\eta=0.35, \epsilon=0.02, \rho_0=1, L_x=Lx, L_y=200, n_b=2$",
            fontsize="x-large")
        plt.show()
        plt.close()
    print("sample size =", i)
    return x, peak_ave / i


def sample_ave(Lx, nb, eta=350, eps=20, is_show=False, path=None):
    if path is None:
        path = [
            r"G:/data/band/Lx/snapshot/eps20/",
            r"G:/data/band/Lx/snapshot/uniband/",
            r"G:/data/band/Lx/snapshot/eps20_2019/",
            r"G:/data/band/Lx/snapshot/eps20_mpi/",
            r"G:/data/band/Lx/snapshot/eps20_tanglou/"
        ]

    if nb == 8:
        rate_min = 0.3
    elif nb == 4:
        rate_min = 0.3
    else:
        rate_min = 0.3
    para_dict = {"Lx": Lx, "eta": eta, "eps": eps}
    dictLSN = read_matched_file(para_dict, path, rate_min=rate_min)
    phi = np.array(
        [i for i in eq_Lx_and_nb(Lx, nb, "mean_phi", dictLSN=dictLSN)])
    if (phi.size > 1):
        color_list = plt.cm.jet(
            [(i - phi.min()) / (phi.max() - phi.min()) for i in phi])
    else:
        color_list = plt.cm.jet([0])

    x = np.arange(Lx) + 0.5
    rho_gas = np.zeros_like(phi)
    peak_ave = np.zeros_like(x)

    peak_list = []
    for i, peak in enumerate(
            eq_Lx_and_nb(Lx, nb, "ave_peak", dictLSN=dictLSN)):
        rho_gas[i] = np.mean(peak[190:200])
        # plt.plot(x, peak, c=color_list[i], lw=0.5, alpha=0.5)
        peak_ave += peak
        peak_list.append(peak)
    print(i+1, "valid samples")
    peak_ave /= i + 1
    if is_show:
        for i, peak in enumerate(peak_list):
            plt.plot(x, peak, c=color_list[i])
        plt.show()
        plt.close()
    else:
        return x, peak_ave, peak_list, color_list


def collapse_varied_L(nb_arr, Lx_arr):
    x_list, p_list = [], []
    sample_size = []
    for i in range(len(nb_arr)):
        x, p, ps, c = sample_ave(Lx_arr[i], nb_arr[i])
        x_list.append(x)
        p_list.append(p)
        sample_size.append(len(ps))

    fig = plt.figure(constrained_layout=True, figsize=(5, 5))
    mk = ["-", "--", ":"]
    for i in range(len(x_list)):
        plt.plot(
            x_list[i],
            p_list[i],
            mk[i],
            label=r"$L_x=%d, n_b=%d, n_s=%d$" % (Lx_arr[i], nb_arr[i],
                                                 sample_size[i]),
            lw=1)
    plt.xlim(60, 210)

    # title = r"$\eta=0.35, \epsilon=0.02, \rho_0=1, L_y=200$"
    # plt.suptitle(title, fontsize="xx-large")
    plt.ylabel(
        r"sample-averaged $\langle \overline{\rho}_y(x)\rangle_t$",
        fontsize="x-large")
    plt.xlabel(r"$x$", fontsize="x-large")
    plt.legend(fontsize="large")
    
    ax_in = fig.add_axes([0.25, 0.35, .3, .4])
    for i in range(len(x_list)):
        ax_in.plot(x_list[i], p_list[i], mk[i], lw=1)
    ax_in.set_xlim(125, 135)
    ax_in.set_ylim(0.8, 1.65)
    plt.show()
    plt.close()


def collapse_same_nb(nb=2, L_arr=[400, 420, 440, 460, 480]):
    x_list, p_list = [], []
    sample_size = []
    for Lx in L_arr:
        x, p, ps, c = sample_ave(Lx, nb)
        x_list.append(x)
        p_list.append(p)
        sample_size.append(len(ps))

    plt.figure(constrained_layout=True, figsize=(5, 5))
    for i in range(len(x_list)):
        plt.plot(
            x_list[i],
            p_list[i],
            label=r"$L_x=%d, n_b=%d, n_s=%d$" % (L_arr[i], nb, sample_size[i]))
    plt.xlim(60, 210)
    # title = r"$\eta=0.35, \epsilon=0.02, \rho_0=1, L_y=200$"
    # plt.suptitle(title, fontsize="xx-large")
    plt.ylabel(
        r"sample-averaged $\langle \overline{\rho}_y(x)\rangle_t$",
        fontsize="x-large")
    plt.xlabel(r"$x$", fontsize="x-large")
    plt.legend(fontsize="large")
    plt.show()
    plt.close()


def output_peak_data(eta=350, eps=20):
    for nb in [2, 4, 8]:
        for L0 in [200, 210, 220, 230, 240]:
            Lx = L0 * nb
            x, peak_ave, peak_list, color_list = sample_ave(
                Lx, nb, eta, eps, False)
            outpath = r"C:\Users\duany\Desktop"
            outfile = r"mean_peak_eta%d_eps%d_L%d_nb%d.dat" % (
                eta, eps, Lx, nb)
            with open(outpath + os.path.sep + outfile, "w") as f:
                for i in range(x.size):
                    f.write("%g\t%.8f\n" % (x[i], peak_ave[i]))


if __name__ == "__main__":
    # plot_peak(460, 2)
    # x1, peak1 = rescale(440, 2, show=False)
    # x2, peak2 = rescale(660, 3, show=True)
    # x3, peak3 = rescale(880, 4, show=False)
    # x4, peak4 = rescale(1100, 5, show=False)

    # plt.plot(x1, peak1, x2, peak2, x3, peak3)
    # plt.xlim(60, 210)
    # plt.show()
    # plt.close()
    # sample_ave(660, 3)
    # collapse_same_nb()
    # collapse_varied_L([2, 3, 4, 5], [440, 660, 880, 1100])

    # l = 200
    # collapse_varied_L([2, 4, 8], [2 * l, 4 * l, 8 * l])

    # mypath = r"E:/data/random_torque/bands/Lx/snapshot/eps20_2019/"
    # sample_ave(840, 4, is_show=True, path=mypath)

    output_peak_data()