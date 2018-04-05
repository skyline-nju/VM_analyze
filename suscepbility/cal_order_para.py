import numpy as np
import glob
import pandas as pd
import os


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
        if L >= 724:
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


def get_filename_pattern(fmt, eta, L=None, eps=None):
    if L is None and eps is None:
        if fmt == "dat":
            pat = "p*.%g.*.dat" % (eta * 1000)
        elif fmt == "xlsx":
            pat = "%.2f_*_*.xlsx" % (eta)
    elif L is not None:
        if eps is None:
            if fmt == "dat":
                pat = "p%d.%g.*.dat" % (L, eta * 1000)
            elif fmt == "xlsx":
                pat = "%.2f_*_%d.xlsx" % (eta, L)
        else:
            if fmt == "dat":
                pat = "p%d.%g.%g.*.dat" % (L, eta * 1000, eps * 10000)
            elif fmt == "xlsx":
                pat = "%.2f_%.4f_%d.xlsx" % (eta, eps, L)
    else:
        if fmt == "dat":
            pat = "p*.%g.%g*.dat" % (eta * 1000, eps * 10000)
        elif fmt == "xlsx":
            pat = "%.2f_%.4f_*.xlsx" % (eta, eps)
    return pat


def read_time_average(eta, L=None, eps=None, transpos=True):
    xlsx_dir = r"E:\data\random_torque\susceptibility\time_average"
    if not os.path.exists(xlsx_dir):
        xlsx_dir = r"D:\data\random_torque\susceptibility\time_average"

    pat = get_filename_pattern("xlsx", eta, L, eps)
    files = glob.glob(xlsx_dir + os.path.sep + pat)
    data_dict = {}
    for filename in files:
        s = os.path.basename(filename).replace(".xlsx", "").split("_")
        my_eps = float(s[1])
        my_L = int(s[2])
        df = pd.read_excel(filename, sheet_name="Sheet1")
        if my_eps not in data_dict:
            if transpos:
                data_dict[my_eps] = {my_L: df.T}
            else:
                data_dict[my_eps] = {my_L: df}
        else:
            if transpos:
                data_dict[my_eps][my_L] = df.T
            else:
                data_dict[my_eps][my_L] = df
    return data_dict


def time_average(eta, L=None, eps=None, new_data=True):
    if not new_data:
        if eta == 0.1:
            data_dir = r"E:\data\random_torque\Phi_vs_L\eta=0.10\serials"
    else:
        data_dir = r"E:\data\random_torque\susceptibility\phi\eta=%.2f" % eta
        if not os.path.exists(data_dir):
            data_dir = data_dir.replace("E", "D")

    dest_dir = r"E:\data\random_torque\susceptibility\time_average"
    if not os.path.exists(dest_dir):
        dest_dir = dest_dir.replace("E", "D")
    pat = get_filename_pattern("dat", eta, L, eps)
    data_dict = read_time_average(eta, L, eps)

    files = glob.glob(data_dir + os.path.sep + pat)
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
    if epsilon is None:
        epsilon = [
            0, 10, 100, 200, 300, 350, 400, 450, 500, 550, 565, 580, 600, 625,
            650, 700, 725, 800, 850
        ]
    index = ["mean", "var", "n_steps"]
    dest_dir = r"E:\data\random_torque\susceptibility\time_average"
    data_dir0 = r"E:\data\random_torque\Phi_vs_L\eta=0.18"
    for eps_i in epsilon:
        eps = eps_i / 10000
        data_dict = read_time_average(eta, eps=eps)
        if len(data_dict) > 0:
            data_dict = data_dict[eps]
        if eps_i <= 500:
            data_dir = data_dir0 + os.path.sep + r"%.3f" % (eps_i / 10000)
        else:
            data_dir = data_dir0 + os.path.sep + r"0.055plus"
        files1 = glob.glob(data_dir + os.path.sep + r"p*.180.%d.*.dat" %
                           (eps_i))
        files2 = glob.glob(data_dir + os.path.sep + r"p*.180.%d.*.dat" %
                           (eps_i // 10))
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


def sample_average(eta):
    data_dict = read_time_average(eta, transpos=False)
    excel_file = r"E:\data\random_torque\susceptibility\sample_average"
    if not os.path.exists(excel_file):
        excel_file = excel_file.replace("E", "D")
    excel_file += os.path.sep + r"eta=%g.xlsx" % eta
    phi_dict, chi_dict, chi_dis_dict, n_dict = {}, {}, {}, {}
    for eps in data_dict:
        for L in data_dict[eps]:
            df = data_dict[eps][L]
            phi = df["mean"].mean()
            chi = df["var"].mean() * L * L
            chi_dis = ((df["mean"]**2).mean() - phi * phi) * L * L
            n = df["mean"].size
            if L in phi_dict:
                phi_dict[L][eps] = phi
                chi_dict[L][eps] = chi
                chi_dis_dict[L][eps] = chi_dis
                n_dict[L][eps] = n
            else:
                phi_dict[L] = {eps: phi}
                chi_dict[L] = {eps: chi}
                chi_dis_dict[L] = {eps: chi_dis}
                n_dict[L] = {eps: n}
    with pd.ExcelWriter(excel_file) as writer:
        pd.DataFrame.from_dict(phi_dict).to_excel(writer, sheet_name="phi")
        pd.DataFrame.from_dict(chi_dict).to_excel(writer, sheet_name="chi")
        pd.DataFrame.from_dict(chi_dis_dict).to_excel(
            writer, sheet_name="chi_dis")
        pd.DataFrame.from_dict(n_dict).to_excel(writer, sheet_name="num")


if __name__ == "__main__":
    eta = 0.1
    # df2 = pd.read_excel(r"..\%.2f_%.4f.xlsx" % (eta, eps), "L=%d" % L)
    # print(df2)
    # df2.to_excel(r"..\tmp.xlsx", sheet_name="a")
    time_average(eta, new_data=True)
    sample_average(eta)
