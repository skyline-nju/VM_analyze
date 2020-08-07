import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
try:
    from corr2d import add_line
except ModuleNotFoundError:
    print("No mdoule named 'corr2d'")
    exit()


def read_phi_serials(fin, ncut=3000):
    with open(fin, "r") as f:
        lines = f.readlines()[ncut:]
        phi = np.array([float(i.split("\t")[0]) for i in lines])
        # print(lines)
        phi_mean = np.mean(phi)
        phi_var = np.var(phi)
    return phi_mean, phi_var


def cal_phi(L, eta, eps):
    os.chdir("E:/data/random_torque/defects/eta=%g" % eta)
    pat = "serials/p%d.%g.%g.*.dat" % (L, eta * 1000, eps * 1000)
    files = glob.glob(pat)
    phi_mean_arr, phi_var_arr = np.zeros((2, len(files)))
    for i, fin in enumerate(files):
        phi_mean_arr[i], phi_var_arr[i] = read_phi_serials(fin, ncut=3000)

    np.savez("order_para/%d_%.3f_%.3f.npz" % (L, eta, eps),
             mean=phi_mean_arr,
             var=phi_var_arr)


def cal_suscept(L, eta, eps):
    os.chdir("E:/data/random_torque/defects/eta=%g" % eta)
    order_para = np.load("order_para/%d_%.3f_%.3f.npz" % (L, eta, eps))
    phi_mean = np.mean(order_para["mean"])
    chi_dis = L ** 2 * np.var(order_para["mean"])
    chi_con = L ** 2 * np.mean(order_para["var"])
    n = order_para["mean"].size
    return phi_mean, chi_dis, chi_con, n


def plot_suscept(eta, eps, L_arr=None):
    if L_arr is None:
        if eta == 0.45:
            L_arr = np.array([32, 64, 128])
        else:
            L_arr = np.array([16, 32, 64, 128, 256])
    phi, chi_dis, chi_con = np.zeros((3, L_arr.size))
    for i, L in enumerate(L_arr):
        phi[i], chi_dis[i], chi_con[i], n = cal_suscept(L, eta, eps)
        print(L, phi[i], chi_dis[i]/L**2, chi_con[i]/L**2, n)
    plt.plot(L_arr, chi_dis/L_arr**2, "o")
    plt.xscale("log")
    plt.yscale("log")
    add_line.add_line(plt.gca(), 0, 1, 1, -4, scale="log")
    add_line.add_line(plt.gca(), 0, 1, 1, -3, scale="log")
    add_line.add_line(plt.gca(), 0, 1, 1, -2, scale="log")

    plt.show()
    plt.close()


if __name__ == "__main__":
    eta = 0.45
    eps = 0.
    L_arr = np.array([32, 64, 128])
    # for L in L_arr:
    #     cal_phi(L, eta, eps)
    plot_suscept(eta, eps, L_arr)

    folder = "E:/data/random_torque/defects"
    fin = "%s/defects/eta=0.45/order_para/32_0.450_0.000.npz" % folder
    # data = np.load(fin)
    # for key in data.keys():
    #     print(key)
    # print(data["mean"])
