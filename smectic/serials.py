import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("..")
try:
    from suscepbility.theta import read_phi_theta, untangle
except ImportError:
    print("error when import add_line")


def show_gamma0():
    os.chdir(r"D:\data\smectic\gamma0")
    pat = "s%.2f_b%.2f_r%.2f_L%d_v%.2f.dat"

    L_arr = [48, 48, 96, 144, 192]
    v_arr = [0.2, 0.3, 0.3, 0.3, 0.3]
    for i, L in enumerate(L_arr):
        v0 = v_arr[i]
        f = pat % (0.01, 1.3, 10.0, L, v0)
        phi, theta = read_phi_theta(f, 0)
        theta = untangle(theta)
        t = np.arange(phi.size) * 100
        plt.plot(t, theta, label=r"$L=%d, v_0=%g$" % (L, v0))
    plt.legend()
    plt.xlabel(r"$t$")
    plt.ylabel("orientation of mean velocities")
    plt.title(r"$\eta=0.01, \beta=1.3, \rho_0=10, \gamma=0$")
    plt.tight_layout()
    plt.show()
    plt.close()


def show_gamma90_varied_beta(rho=1.5, v=0.2, eta=0.02, L=128):
    beta_arr = np.array([
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
        1.5
    ])
    # c_list = plt.cm.viridis([(i-0.1)/1.4 for i in beta_arr])
    os.chdir(r"F:\data\smectic\gamma90\small_L")
    for i, beta in enumerate(beta_arr[:8]):
        fname = r"s%.2f_b%.2f_r%.2f_L%d_v%.2f.dat" % (eta, beta, rho, L, v)
        phi, theta = read_phi_theta(fname, ncut=0)
        t = np.arange(phi.size) * 100
        theta = untangle(theta)
        plt.plot(t, theta, label="%.1f" % beta, lw=2)
    plt.legend(title=r"$\beta=$")
    plt.xlabel(r"$t$")
    plt.ylabel("orientation of mean velocities")
    plt.title(r"$L=%d, \eta=%.2f, \rho_0=%g, \gamma=\pi/2$" % (L, eta, rho))
    plt.tight_layout()
    plt.show()
    plt.close()


def show_gamma90_varied_L(rho=3, v=0.2, eta=0.02, beta=1):
    L_arr = np.array([128, 256, 512])
    # c_list = plt.cm.viridis([(i-0.1)/1.4 for i in beta_arr])
    os.chdir(r"D:\data\smectic\gamma90")
    for i, L in enumerate(L_arr):
        fname = r"s%.2f_b%.2f_r%.2f_L%d_v%.2f.dat" % (eta, beta, rho, L, v)
        phi, theta = read_phi_theta(fname, ncut=0)
        t = np.arange(phi.size) * 100
        theta = untangle(theta)
        plt.plot(t, theta, label="%d" % L, lw=2)
    plt.legend(title=r"$\L=$")
    plt.xlabel(r"$t$")
    plt.ylabel("orientation of mean velocities")
    plt.title(r"$\beta=%g, \eta=%.2f, \rho_0=%g, \gamma=\pi/2$" %
              (beta, eta, rho))
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    # beta = 0.5
    # eta = 0.02
    # v = 0.2
    # rho = 5
    # os.chdir(r"D:\data\smectic\gamma90")
    # L_arr = [128, 256, 512]
    # for L in L_arr:
    #     fname = r"s%.2f_b%.2f_r%.2f_L%d_v%.2f.dat" % (eta, beta, rho, L, v)
    #     phi, theta = read_phi_theta(fname, ncut=0)
    #     t = np.arange(phi.size) * 100
    #     theta = untangle(theta)
    #     plt.plot(t, theta, label="%d" % L)
    # plt.show()
    # plt.close()
    # if beta == 0.5:
    #     eta_arr = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12]
    # elif beta == 1.0:
    #     eta_arr = [0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
    # elif beta == 1.5:
    #     eta_arr = [
    #         0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11,
    #         0.12, 0.13, 0.14, 0.15
    #     ]
    # L = 40

    # for eta in eta_arr:
    #     fname = "s%g_b%g_r%g_L%d.dat" % (eta, beta, rho, L)
    #     phi, theta = read_phi_theta(fname, ncut=0)
    #     t = np.arange(phi.size) * 100
    #     theta = untangle(theta)
    #     if beta == 0.5 and eta == 0.06:
    #         theta = -theta
    #     elif beta == 1 and eta == 0.09:
    #         theta = -theta
    #     elif beta == 1.5 and (eta == 0.08 or eta == 0.05 or eta == 0.07
    #                           or eta == 0.02 or eta == 0.03):
    #         theta = -theta
    #     plt.plot(t, theta, label="%.2f" % eta)
    # plt.legend(title=r"$\eta=$")
    # plt.xlabel(r"$t$")
    # plt.ylabel("orientation of mean velocities")
    # plt.title(r"$L=%d, \rho_0=%g, \beta=%g, \gamma=\pi/2$" % (L, rho, beta))
    # plt.tight_layout()
    # plt.show()
    # plt.close()
    # show_gamma0()
    show_gamma90_varied_L()
