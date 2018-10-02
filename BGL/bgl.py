import numpy as np
import matplotlib.pyplot as plt


def cal_P1(eta):
    return np.exp(-eta**2 / 2)


def cal_P2(eta):
    return np.exp(-eta**2 * 4 / 2)


def cal_alpha(rho, rho0, eta):
    P1 = cal_P1(eta)
    return P1 - 1 + 4 / np.pi * (P1 - 2 / 3) * rho0 * rho


def cal_beta(rho0, eta):
    P1 = cal_P1(eta)
    P2 = cal_P2(eta)
    res = 16 / np.pi * (5 * P1 - 2) * (2 * P2 + 1) * rho0 ** 2 \
        / (15 * np.pi * (1 - P2) + 8 * (7 + 5 * P2) * rho0)
    return res


def cal_lambda1(rho0, eta):
    P1 = cal_P1(eta)
    P2 = cal_P2(eta)
    res = 2 * (16 + 30 * P2 - 15 * P1) * rho0 \
        / (15 * np.pi * (1 - P2) + 8 * (7 + 5 * P2) * rho0)
    return res


def cal_v2(rho0, eta):
    beta = cal_beta(rho0, eta)
    P1 = cal_P1(eta)
    alpha = cal_alpha(rho0, rho0, eta)
    if alpha > 0:
        v0 = np.sqrt(alpha / beta)
        return (v0 + rho0 ** 2 * 2 / np.pi * (P1 - 2 / 3) / beta / v0) / v0
    else:
        return np.nan


if __name__ == "__main__":
    eta = np.linspace(0, 1, 500)
    rho0 = np.linspace(0, 10, 400)

    # lambda1 = np.zeros_like(eta)
    # for i in range(eta.size):
    #     lambda1 = cal_lambda1(4, eta)
    # plt.plot(eta, lambda1)
    # plt.show()
    # plt.close()

    v2 = np.zeros((rho0.size, eta.size))
    for i in range(rho0.size):
        for j in range(eta.size):
            v2[i, j] = cal_v2(rho0[i], eta[j])
    # v2[v2 > 5] = 5
    levels = np.linspace(1.5, 5, 15)
    plt.contourf(rho0, eta, v2.T, levels)
    cb = plt.colorbar()
    cb.set_label(r"$v_2 / v_0$", fontsize="x-large")
    plt.xlabel(r"$\rho_0$", fontsize="x-large")
    plt.ylabel(r"$\eta$", fontsize="x-large")
    plt.tight_layout()
    plt.show()
    plt.close()
