import numpy as np
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt


def read_time_ave(data_dir, eta):
    files = glob.glob(data_dir + os.path.sep + "%.2f_*_*" % eta)
    data_dict = {}
    for filename in files:
        s = os.path.basename(filename).replace(".xlsx", "").split("_")
        eps = float(s[1])
        L = int(s[2])
        df = pd.read_excel(filename, sheet_name="Sheet1", index_col=0)
        m_t_s = df["mean"].mean()
        var_s = df["var"].mean()
        m_t_square_s = (df["mean"] ** 2).mean()
        m_square_t_s = var_s + m_t_square_s
        if L not in data_dict:
            data_dict[L] = [[eps], [m_t_s], [m_square_t_s], [m_t_square_s]]
        else:
            data_dict[L][0].append(eps)
            data_dict[L][1].append(m_t_s)
            data_dict[L][2].append(m_square_t_s)
            data_dict[L][3].append(m_t_square_s)
    for L in data_dict:
        eps_arr = np.array(data_dict[L][0])
        idx = np.argsort(eps_arr)
        data_dict[L][0] = eps_arr[idx]
        data_dict[L][1] = np.array(data_dict[L][1])[idx]
        data_dict[L][2] = np.array(data_dict[L][2])[idx]
        data_dict[L][3] = np.array(data_dict[L][3])[idx]
    if eta == 0.10:
        del data_dict[16]
        del data_dict[22]
        del data_dict[26]
        del data_dict[38]
        del data_dict[54]
        del data_dict[108]
    elif eta == 0.18:
        for L in [12, 14, 16, 19, 22, 23, 26, 27, 38, 45,
                  54, 76, 107, 108, 152, 215, 1024, 1448, 2048]:
            del data_dict[L]
    return data_dict


def plot_cumulant(eta):
    data_dir = r"E:\data\random_torque\susceptibility\time_average"
    if not os.path.exists(data_dir):
        data_dir = data_dir.replace("E", "D")
    data_dir += r"\eta=%.2f" % eta
    data_dict = read_time_ave(data_dir, eta)
    color = plt.cm.gist_rainbow(np.linspace(0, 1, len(data_dict)))
    for i, L in enumerate(sorted(data_dict.keys())):
        # U_con = data_dict[L][2] / data_dict[L][3]
        # U_dis = data_dict[L][3] / data_dict[L][1] ** 2
        U_q = data_dict[L][2] / data_dict[L][1] ** 2
        eps = data_dict[L][0]
        plt.plot(eps, U_q - 1, "-o", c=color[i], label=r"$%d$" % L)
    plt.yscale("log")
    plt.legend()
    plt.show()
    plt.close()


if __name__ == "__main__":
    eta = 0.1
    plot_cumulant(eta)
