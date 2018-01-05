import numpy as np
import glob
import os
import pandas as pd

# from openpyxl.writer.excel import ExcelWriter


def read_txt(filename):
    """ Get array of order parameter and angle from txt file."""
    with open(filename) as f:
        lines = f.readlines()
        n = len(lines)
        phi = np.zeros(n)
        theta = np.zeros(n)
        for i, line in enumerate(lines):
            s = line.replace("\n", "").split("\t")
            phi[i] = float(s[-2])
            theta[i] = float(s[-1])
    return phi, theta


def read_hdf5(infile, ncut=0):
    """ Read `infile` and do some statistics about order parameter. """
    df = pd.read_hdf(infile, "phi")
    phi_mean = np.array(df[ncut:].mean())
    phi_var = np.array(df[ncut:].var(ddof=0))
    n = phi_mean.size
    return np.mean(phi_mean), np.mean(phi_var), n


def get_ncut(L):
    if L == 724:
        ncut = 3000
    elif L == 512:
        ncut = 2500
    else:
        ncut = 2000
    return ncut


def update_phi_and_chi(eta, hdf_dir, csv_dir):
    def update_dict(infile):
        basename = os.path.basename(infile)
        s = basename.replace(".h5", "").split("_")
        L = int(s[0])
        eps = float(s[2])
        ncut = get_ncut(L)

        phi, phi_var, num = read_hdf5(infile, ncut)
        xi = phi_var * L * L
        if L in phi_dict:
            phi_dict[L][eps] = phi
            chi_dict[L][eps] = xi
            num_dict[L][eps] = num
        else:
            phi_dict[L] = {eps: phi}
            chi_dict[L] = {eps: xi}
            num_dict[L] = {eps: num}

    def save_to_excel():
        with pd.ExcelWriter(excel_file) as writer:
            pd.DataFrame.from_dict(phi_dict).to_excel(writer, sheet_name='phi')
            pd.DataFrame.from_dict(chi_dict).to_excel(writer, sheet_name='chi')
            pd.DataFrame.from_dict(num_dict).to_excel(writer, sheet_name='num')

    def read_excel():
        with pd.ExcelFile(excel_file) as f:
            phi_dict = pd.read_excel(f, sheet_name="phi").to_dict()
            chi_dict = pd.read_excel(f, sheet_name="chi").to_dict()
            num_dict = pd.read_excel(f, sheet_name="num").to_dict()
        return phi_dict, chi_dict, num_dict

    excel_file = csv_dir + os.path.sep + "eta=%g.xlsx" % eta
    hdf_files = glob.glob(hdf_dir + os.path.sep + "*.h5")

    phi_dict = {}
    chi_dict = {}
    num_dict = {}
    if not os.path.isfile(excel_file):
        for infile in hdf_files:
            update_dict(infile)
            print("loading ", os.path.basename(infile))
        save_to_excel()
        print("create", excel_file)
    else:
        phi_dict, chi_dict, num_dict = read_excel()
        excel_statinfo = os.stat(excel_file)
        flag_updated = False
        for hdf_file in hdf_files:
            hdf_statinfo = os.stat(hdf_file)
            if hdf_statinfo.st_mtime > excel_statinfo.st_mtime:
                print("loading", os.path.basename(hdf_file))
                update_dict(hdf_file)
                flag_updated = True
        if flag_updated:
            save_to_excel()
            print("update", excel_file)
        else:
            print("nothing changed")


def txt_to_hdf(L, eta, eps, txt_dir, hdf_dir, check_by_time=True):
    """ Transform txt files contaning order parameters into hdf5 files."""

    def update_dict(infile):
        phi, theta = read_txt(infile)
        basename = os.path.basename(infile)
        n = phi.size
        if n not in phi_dict:
            phi_dict[n] = {basename: phi}
            theta_dict[n] = {basename: theta}
        else:
            phi_dict[n][basename] = phi
            theta_dict[n][basename] = theta
        print("add", basename)

    def save_df(df_phi, df_theta):
        """ save date frames to hdf5 file. """
        for n in phi_dict:
            print(n)
            if df_phi is None:
                df_phi = pd.DataFrame(phi_dict[n])
                df_theta = pd.DataFrame(theta_dict[n])
            else:
                df_phi = pd.concat([df_phi, pd.DataFrame(phi_dict[n])], axis=1)
                df_theta = pd.concat(
                    [df_theta, pd.DataFrame(theta_dict[n])], axis=1)
        df_phi.to_hdf(hdf_file, "phi")
        df_theta.to_hdf(hdf_file, "theta")

    txt_files = glob.glob(txt_dir + os.path.sep + "p%d.%g.%g.*.dat" %
                          (L, eta * 1000, eps * 10000))
    hdf_file = hdf_dir + os.path.sep + "%d_%g_%g.h5" % (L, eta, eps)
    phi_dict = {}
    theta_dict = {}
    if os.path.isfile(hdf_file):
        if check_by_time:
            hdf_statinfo = os.stat(hdf_file)
            for txt_file in txt_files:
                txt_statinfo = os.stat(txt_file)
                if txt_statinfo.st_mtime > hdf_statinfo.st_mtime:
                    update_dict(txt_file)
            if len(phi_dict) > 0:
                df_phi = pd.read_hdf(hdf_file, "phi")
                df_theta = pd.read_hdf(hdf_file, "theta")
                save_df(df_phi, df_theta)
            else:
                print(os.path.basename(hdf_file), "remain unchanged")

        else:
            df_phi = pd.read_hdf(hdf_file, "phi")
            df_theta = pd.read_hdf(hdf_file, "theta")
            for txt_file in txt_files:
                basename = os.path.basename(txt_file)
                if basename not in df_phi.columns:
                    update_dict(txt_file)
                else:
                    print(basename, "already exists")
            if len(phi_dict) > 0:
                save_df(df_phi, df_theta)
            else:
                print(os.path.basename(hdf_file), "remain unchanged")
    else:
        df_phi = None
        df_theta = None
        for txt_file in txt_files:
            update_dict(txt_file)
        save_df(df_phi, df_theta)
        print("create", hdf_file)


def update_hdf(eta, txt_dir, hdf_dir, check_by_time=True):
    """ Update all hdf files """
    files = find_existed_files(eta, txt_dir)
    para = get_available_para(files)
    for L in sorted(para.keys()):
        print("L = ", L)
        for eps in para[L]:
            txt_to_hdf(L, 0.1, eps / 10000, txt_dir, hdf_dir, check_by_time)


def find_existed_files(eta, txt_dir, L0=None, eps1e4=None):
    """ find existed files at `txt_dir` with given parameters."""
    prefix = txt_dir + os.path.sep
    if L0 is not None and eps1e4 is not None:
        files = glob.glob(prefix + "p%d.%g.%g.*.dat" % (L0, eta * 1000,
                                                        eps1e4))
    elif L0 is not None:
        files = glob.glob(prefix + "p%d.%g.*.dat" % (L0, eta * 1000))
    elif eps1e4 is not None:
        files = glob.glob(prefix + "p*.%g.%g.*.dat" % (eta * 1000, eps1e4))
    else:
        files = glob.glob(prefix + "p*.%g.*.dat" % (eta * 1000))
    return files


def get_available_para(files, is_txt=True):
    """ Get dict `para` having the form {L: [eps1, eps2]} from `files`."""
    para = {}
    for full_name in files:
        basename = os.path.basename(full_name)
        if is_txt:
            s = basename.replace("p", "").split(".")
            L = int(s[0])
            eps = int(s[2])
            if L in para:
                if eps not in para[L]:
                    para[L].append(eps)
            else:
                para[L] = [eps]
        else:
            s = basename.replace(".h5", "").split("_")
            L = int(s[0])
            eps = float(s[2])
            if L in para:
                para[L].append(eps)
            else:
                para[L] = [eps]
    return para


if __name__ == "__main__":
    # txt_dir = r"C:\Users\user\Desktop\test\phi"
    # hdf_dir = r"C:\Users\user\Desktop\test\phi_h5"
    # txt_dir = r"E:\data\random_torque\Phi_vs_L\eta=0.10\serials"
    txt_dir = r"E:\data\random_torque\susceptibility\phi\eta=0.10"
    hdf_dir = r"E:\data\random_torque\susceptibility\phi_hdf"
    csv_dir = r"E:\data\random_torque\susceptibility"

    # update_hdf(0.1, txt_dir, hdf_dir, True)
    update_phi_and_chi(0.1, hdf_dir, csv_dir)
