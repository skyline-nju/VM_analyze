"""
Cal time- and smaple-averaged order parameters and susceptiliblities from .dat
files.
"""
import numpy as np
import glob
import pandas as pd
import os
import sys


def read_txt(filename, ncut):
    """ Cal mean and variance of the order parameters."""
    with open(filename) as f:
        lines = f.readlines()[ncut:]
        phi = np.array([float(line.split("\t")[0]) for line in lines])
        mean = np.mean(phi)
        var = np.mean(phi**2) - mean * mean
    return mean, var, phi.size


def get_ncut(L, eta, eps=None, double_ncut=False):
    if eta == 0.18:
        if eps < 0.005:
            if L <= 64:
                ncut = 1000
            elif L > 1000:
                ncut = 4000
            elif L > 500:
                ncut = 3000
            else:
                ncut = 2000
        else:
            if L <= 64:
                ncut = 1000
            elif L < 128:
                ncut = 1750
            elif L <= 512:
                ncut = 2500
            elif L == 724:
                ncut = 3000
            else:
                ncut = 3500
            if double_ncut:
                ncut *= 2
    else:
        if L >= 1024:
            ncut = 4000
        elif L >= 724:
            ncut = 3000
        elif L == 521:
            ncut == 2500
        elif L >= 256:
            ncut = 2000
        elif L >= 90:
            ncut = 1750
        else:
            ncut = 1500
    return ncut


def add_new_sample(df,
                   filename,
                   L,
                   eta,
                   seed=None,
                   eps=None,
                   double_ncut=False):
    if seed is None:
        s = os.path.basename(filename).replace(".dat", "").split(".")
        seed = int(s[3])
    if seed not in df.columns:
        ncut = get_ncut(L, eta, eps, double_ncut)
        mean, var, n = read_txt(filename, ncut)
        if double_ncut:
            n *= 50
        else:
            n *= 100
        df[seed] = [mean, var, int(n)]
        print("add", os.path.basename(filename))
    # else:
    #     print(os.path.basename(filename), "already exists!")


def get_filename_pattern(fmt, eta=None, L=None, eps=None):
    if fmt == "dat":
        if eta is not None:
            pat_eta = "%g" % (eta * 1000)
        else:
            pat_eta = "*"
        if L is not None:
            pat_L = "%d" % L
        else:
            pat_L = "*"
        if eps is not None:
            pat_eps = "%g" % (eps * 10000)
        else:
            pat_eps = "*"
        pat = "p%s.%s.%s.*.dat" % (pat_L, pat_eta, pat_eps)
    else:
        if eta is not None:
            pat_eta = "%.2f" % eta
        else:
            pat_eta = "*"
        if L is not None:
            pat_L = "%d" % L
        else:
            pat_L = "*"
        if eps is not None:
            pat_eps = "%.4f" % eps
        else:
            pat_eps = "*"
        pat = "%s_%s_%s.xlsx" % (pat_eta, pat_eps, pat_L)
    return pat


def read_time_average(xlsx_dir,
                      eta=None,
                      L=None,
                      eps=None,
                      transpos=True,
                      key_name="eta"):
    pat = get_filename_pattern("xlsx", eta, L, eps)
    print(pat)
    files = glob.glob(xlsx_dir + os.path.sep + pat)
    data_dict = {}
    for filename in files:
        s = os.path.basename(filename).replace(".xlsx", "").split("_")
        if key_name == "eta" and eta is not None:
            key = float(s[1])
        elif key_name == "eps" and eps is not None:
            key = float(s[0])
        else:
            print("error when reading time-averaged data")
            sys.exit()
        my_L = int(s[2])
        df = pd.read_excel(filename, sheet_name="Sheet1", index_col=0)
        if key not in data_dict:
            if transpos:
                data_dict[key] = {my_L: df.T}
            else:
                data_dict[key] = {my_L: df}
        else:
            if transpos:
                data_dict[key][my_L] = df.T
            else:
                data_dict[key][my_L] = df
    return data_dict


def time_average_const_eta(eta, L=None, eps=None, new_data=True):
    root = r"E:\data\random_torque"
    if not os.path.exists(root):
        root = root.replace("E", "D")
        print("root =", root)
    if not new_data:
        if eta == 0.1:
            data_dir = root + r"\Phi_vs_L\eta=0.10\serials"
    else:
        data_dir = root + r"\susceptibility\phi\eta=%.2f" % eta
    print("data dir", data_dir)

    dest_dir = root + r"\susceptibility\time_average\eta=%.2f" % eta
    # if not os.path.exists(dest_dir):
    #     dest_dir = dest_dir.replace("E", "D")
    pat = get_filename_pattern("dat", eta, L, eps)
    print(pat)
    data_dict = read_time_average(dest_dir, eta, L, eps)

    files = glob.glob(data_dir + os.path.sep + pat)
    print(data_dir + os.path.sep + pat)
    if eta == 0.10 or eta == 0.05 or eta == 0.18:
        data_new_dir = root + r"\susceptibility\phi\eta=%.2f_CSRC" % eta
        files_new = glob.glob(data_new_dir + os.path.sep + pat)
        files += files_new
        data_new_dir = root + r"\susceptibility\phi\eta=%.2f_BM" % eta
        files_new = glob.glob(data_new_dir + os.path.sep + pat)
        files += files_new
    print("total files:", len(files))
    index = ["mean", "var", "n_steps"]
    for filename in files:
        s = os.path.basename(filename)
        s = s.replace(".dat", "").replace("p", "").split(".")
        L = int(s[0])
        eps = int(s[2]) / 10000
        seed = int(s[3])

        if eps not in data_dict:
            data_dict[eps] = {L: pd.DataFrame(index=index)}
        elif L not in data_dict[eps]:
            data_dict[eps][L] = pd.DataFrame(index=index)
        add_new_sample(data_dict[eps][L], filename, L, eta, seed, eps)

    for eps in data_dict:
        for L in data_dict[eps]:
            out_file = dest_dir + os.path.sep + "%.2f_%.4f_%d.xlsx" % (eta,
                                                                       eps, L)
            data_dict[eps][L].T.to_excel(out_file, sheet_name="Sheet1")


def time_average_eta18_old(epsilon=None):
    eta = 0.18
    if epsilon is None:
        epsilon = [
            0, 10, 100, 200, 300, 350, 400, 450, 500, 550, 565, 580, 600, 625,
            650, 700, 725, 800, 850
        ]
    index = ["mean", "var", "n_steps"]
    dest_dir = r"E:\data\random_torque\susceptibility\time_average\eta=0.18"
    data_dir0 = r"E:\data\random_torque\Phi_vs_L\eta=0.18"
    for eps_i in epsilon:
        eps = eps_i / 10000
        data_dict = read_time_average(dest_dir, eta, eps=eps)
        if len(data_dict) > 0:
            data_dict = data_dict[eps]
        if eps_i <= 500:
            data_dir = data_dir0 + os.path.sep + r"%.3f" % (eps_i / 10000)
        else:
            data_dir = data_dir0 + os.path.sep + r"0.055plus"
        files1 = glob.glob(data_dir + os.path.sep +
                           r"p*.180.%d.*.dat" % (eps_i))
        files2 = glob.glob(data_dir + os.path.sep +
                           r"p*.180.%d.*.dat" % (eps_i // 10))
        for i, filename in enumerate(files1 + files2):
            if i < len(files1):
                double_ncut = False
            else:
                double_ncut = True
            s = os.path.basename(filename)
            s = s.replace(".dat", "").replace("p", "").split(".")
            L = int(s[0])
            seed = int(s[3])

            if L not in data_dict:
                data_dict[L] = pd.DataFrame(index=index)
            add_new_sample(data_dict[L], filename, L, eta, seed, eps,
                           double_ncut)
        for L in data_dict:
            out_file = dest_dir + os.path.sep + "%.2f_%.4f_%d.xlsx" % (eta,
                                                                       eps, L)
            data_dict[L].T.to_excel(out_file, sheet_name="Sheet1")


def time_average_const_eps(eps, L=None, eta=None):
    root = r"E:\data\random_torque\susceptibility"
    if not os.path.exists(root):
        root = root.replace("E", "D")
    data_dir = root + r"\phi\eps=%g" % eps
    dest_dir = root + r"\time_average\eps=%g" % eps

    pat = get_filename_pattern("dat", eta, L, eps)
    data_dict = read_time_average(dest_dir, L=L, eps=eps, key_name="eps")
    files = glob.glob(data_dir + os.path.sep + pat)
    index = ["mean", "var", "n_steps"]

    for filename in files:
        s = os.path.basename(filename)
        s = s.replace(".dat", "").replace("p", "").split(".")
        L = int(s[0])
        eta = int(s[1]) / 1000
        seed = int(s[3])

        if eta not in data_dict:
            data_dict[eta] = {L: pd.DataFrame(index=index)}
        elif L not in data_dict[eta]:
            data_dict[eta][L] = pd.DataFrame(index=index)
        add_new_sample(data_dict[eta][L], filename, L, eta, seed, eps)

    for eta in data_dict:
        for L in data_dict[eta]:
            out_file = dest_dir + os.path.sep + "%.3f_%.4f_%d.xlsx" % (eta,
                                                                       eps, L)
            data_dict[eta][L].T.to_excel(out_file, sheet_name="Sheet1")


def sample_average(eta=None, eps=None):
    mean_dir = r"E:\data\random_torque\susceptibility\time_average"
    if not os.path.exists(mean_dir):
        mean_dir = mean_dir.replace("E", "D")
    if eta is not None:
        mean_dir += r"\eta=%.2f" % eta
        data_dict = read_time_average(mean_dir, eta, transpos=False)
    elif eps is not None:
        mean_dir += r"\eps=%g" % eps
        data_dict = read_time_average(
            mean_dir, eps=eps, transpos=False, key_name="eps")
    excel_file = r"E:\data\random_torque\susceptibility\sample_average"
    if not os.path.exists(excel_file):
        excel_file = excel_file.replace("E", "D")
    if eta is not None:
        excel_file += os.path.sep + r"eta=%g.xlsx" % eta
    elif eps is not None:
        excel_file += os.path.sep + r"eps=%g.xlsx" % eps
    phi_dict, chi_dict, chi_dis_dict, n_dict, phi2_dict = {}, {}, {}, {}, {}
    if eta == 0.18:
        filter_dict = {
            0.001: [19, 27, 38, 512, 724, 1024, 1448, 2048],
            0.01: [19, 27, 38, 54, 76, 107, 152, 215, 1448, 2048],
            0.02: [19, 27, 38, 54, 76, 107, 152, 215, 45]
        }
    for key in data_dict:
        for L in sorted(data_dict[key].keys()):
            if key in filter_dict and L in filter_dict[key]:
                continue
            df = data_dict[key][L]
            phi = df["mean"].mean()
            chi = df["var"].mean() * L * L
            chi_dis = ((df["mean"]**2).mean() - phi * phi) * L * L
            n = df["mean"].size
            phi2 = np.mean(df["mean"]**2)
            if L in phi_dict:
                phi_dict[L][key] = phi
                chi_dict[L][key] = chi
                chi_dis_dict[L][key] = chi_dis
                n_dict[L][key] = n
                phi2_dict[L][key] = phi2
            else:
                phi_dict[L] = {key: phi}
                chi_dict[L] = {key: chi}
                chi_dis_dict[L] = {key: chi_dis}
                n_dict[L] = {key: n}
                phi2_dict[L] = {key: phi2}
    with pd.ExcelWriter(excel_file) as writer:
        pd.DataFrame.from_dict(phi_dict).to_excel(writer, sheet_name="phi")
        pd.DataFrame.from_dict(chi_dict).to_excel(writer, sheet_name="chi")
        pd.DataFrame.from_dict(chi_dis_dict).to_excel(
            writer, sheet_name="chi_dis")
        pd.DataFrame.from_dict(n_dict).to_excel(writer, sheet_name="num")
        pd.DataFrame.from_dict(phi2_dict).to_excel(writer, sheet_name="phi2")


if __name__ == "__main__":
    eta = 0.18
    time_average_const_eta(eta, new_data=True)
    sample_average(eta=eta)
    # eps = 0.03
    # time_average_const_eps(eps)
    # sample_average(eps=eps)
    # time_average_eta18_old()
