"""
Calculate the susceptibility from fluctuations on order parameters.
"""
import numpy as np
import argparse
import glob
import os
import sys
import platform


def read(file, ncut=2000):
    """ Read a file contaning time serials of order parameters.

    Parameters:
    --------
    file : str
        The time serials of order parameters for a certain sample.
    ncut : int, optional
        The initial "ncut" time steps is removed after loading the file.

    Returns:
    --------
    mean : float
        Time-averaged order parameter.
    var : float
        Variance of time serials of order parameters.
    """
    with open(file) as f:
        lines = f.readlines()[ncut:]
        phi = np.array([float(line.split("\t")[0]) for line in lines])
        mean = np.mean(phi)
        var = np.var(phi)
    return mean, var


def read_files(L, seed, eps_list, eta=180):
    """ Read files with veried epsilon but fixed L, eta, seed.

    Parameters:
    --------
    L : int
        System size.
    seed : int
        Random seed.
    eps_list : list
        List of epsilon.
    eta : int, optional
        Strength of noise multiplied by 1000.

    Returns:
    --------
    phi : array_like
        Array of time-averaged order parameters.
    chi : array_like
        Array of susceptibility.
    """
    if L == 724:
        ncut = 3000
    elif L == 512:
        ncut = 2500
    else:
        ncut = 2000
    phi = np.zeros(len(eps_list))
    chi = np.zeros_like(phi)
    for i, eps in enumerate(eps_list):
        file = "p%d.%d.%d.%d.dat" % (L, eta, eps, seed)
        phi[i], chi[i] = read(file, ncut)
        chi[i] *= L**2
    return phi, chi


def save_serials(L, n, eta=180):
    """ Save serials of order parameters and susceptibility with varied epsilon.
    """
    path = "%d" % L
    if os.path.exists(path):
        os.chdir(path)
        flag_exist = True
    else:
        flag_exist = False
    if L == 724:
        eps_list = [510, 520, 525, 535, 550, 565, 580]
    elif L == 90:
        eps_list = [
            500, 535, 550, 575, 600, 625, 650, 675, 700, 725, 750, 800, 850
        ]
    elif L == 64:
        eps_list = [550, 600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850]
    else:
        print("eps_list not given")
        sys.exit()
    phi_list = []
    chi_list = []
    for i in range(n):
        seed = i + (L + eta // 10) * 10000
        print("seed = %d" % seed)
        try:
            phi, chi = read_files(L, seed, eps_list)
            phi_list.append(phi)
            chi_list.append(chi)
        except:
            print("Error for i = %d" % i)
    phi_array = np.array([i for i in phi_list])
    chi_array = np.array([i for i in chi_list])
    eps_array = np.array(eps_list)
    if flag_exist:
        outfile = "../%d.npz" % L
    else:
        outfile = "%d.npz" % L
    np.savez(outfile, phi=phi_array, chi=chi_array, eps=eps_array)


def sample_average_Unix(L, eps, first=None, last=None, eta=180):
    """ Calculate sample-averaged order parameter and susceptibility for given
        system size L and strength of disorder eps.
    """
    path = "%d" % L
    if os.path.exists(path):
        os.chdir(path)
    if first is None:
        files = glob.glob("p%d.%d.%d.*.dat" % (L, eta, eps))
    else:
        files = [
            "p%d.180.%d.%d.dat" % (L, eps, (L + eta // 10) * 10000 + j)
            for j in range(first, last)
        ]
    phi = []
    chi = []
    print("%d files" % len(files))
    for file in files:
        print(file)
        try:
            mean, var = read(file)
            phi.append(mean)
            chi.append(var)
        except:
            print("error in %s" % file)
    phi = np.array(phi)
    chi = np.array(chi) * L**2
    print("%d\t%f\t%f\t%d\t%f\t%f\n" % (L, np.mean(phi), np.std(phi), phi.size,
                                        np.mean(chi), np.std(chi)))


def sample_average_Win(eta, L0=None, eps0=None, Lmin=46, use_old_data=False):
    """ Handle time serials of order parameters and susceptibilities on Windows.
    """

    def average(L, eps, FILES):
        """ Average order parameters and susceptibilities over input FILES.
        """
        if L == 724:
            ncut = 3000
        elif L == 512:
            ncut = 2500
        else:
            ncut = 2000
        n = len(FILES)
        print("%d files for L=%d, eta=%g, epsilon=%g..." % (n, L, eta,
                                                            eps / 10000))
        mean_arr = np.zeros(n)
        var_arr = np.zeros(n)
        for i, file_i in enumerate(FILES):
            mean_arr[i], var_arr[i] = read(file_i, ncut)
        var_arr *= L * L
        phi_mean, phi_std = np.mean(mean_arr), np.std(mean_arr)
        xi_mean, xi_std = np.mean(var_arr), np.std(var_arr)
        dict_L[L][eps] = [phi_mean, phi_std, n, xi_mean, xi_std]

    def save_result():
        """ Save result """
        dict_eps = {}
        for L in dict_L:
            for eps in dict_L[L]:
                if eps in dict_eps:
                    dict_eps[eps][L] = dict_L[L][eps]
                else:
                    dict_eps[eps] = {L: dict_L[L][eps]}
        os.chdir(dest_dir)
        for eps in dict_eps.keys():
            file_name = "%.4f.dat" % (eps / 10000)
            try:
                f = open(file_name)
                print("read %s" % file_name)
                lines = f.readlines()
                for line in lines:
                    s = line.replace("\n", "").split("\t")
                    L = int(s[0])
                    if L not in dict_eps[eps].keys():
                        dict_eps[eps][L] = [
                            float(s[1]),
                            float(s[2]),
                            int(s[3]),
                            float(s[4]),
                            float(s[5])
                        ]
                f.close()
            except:
                print("%s doesn't exist" % file_name)
            with open(file_name, "w") as f:
                print("write to %s" % file_name)
                for L in sorted(dict_eps[eps].keys()):
                    data = dict_eps[eps][L]
                    f.write("%d\t%f\t%f\t%d\t%f\t%f\n" %
                            (L, data[0], data[1], data[2], data[3], data[4]))

    dest_dir = r"%s\data\eta=%.2f" % (os.getcwd(), eta)
    data_dir1 = r"E:\data\random_torque\susceptibility\phi\eta=%.2f" % eta
    data_dir2 = r"E:\data\random_torque\Phi_vs_L\eta=%.2f\serials" % eta
    if L0 is not None and eps0 is not None:
        pat = r"%s\p%d.%g.%d.*.dat"
        files = glob.glob(pat % (data_dir1, L0, eta * 1000, eps0))
        if use_old_data:
            files += glob.glob(pat % (data_dir2, L0, eta * 1000, eps0))
    elif L0 is not None:
        pat = r"%s\p%d.%g.*.dat"
        files = glob.glob(pat % (data_dir1, L0, eta * 1000))
        if use_old_data:
            files += glob.glob(pat % (data_dir2, L0, eta * 1000))
    elif eps0 is not None:
        pat = r"%s\p*.%g.%d.*.dat"
        files = glob.glob(pat % (data_dir1, eta * 1000, eps0))
        if use_old_data:
            files += glob.glob(pat % (data_dir2, eta * 1000, eps0))
    else:
        pat = r"%s\p*.%g.*.dat"
        files = glob.glob(pat % (data_dir1, eta * 1000))
        if use_old_data:
            files += glob.glob(pat % (data_dir2, eta * 1000))
    n_total = len(files)
    n_count = 0
    dict_L = {}
    for file_name in files:
        s = file_name.split("\\")[-1]
        s = s.replace("p", "").split(".")
        L = int(s[0])
        eps = int(s[2])
        if L >= Lmin:
            if L in dict_L:
                if eps not in dict_L[L]:
                    dict_L[L][eps] = []
            else:
                dict_L[L] = {eps: []}
    for L in dict_L:
        for eps in dict_L[L]:
            files = glob.glob(r"%s\p%d.%g.%d.*.dat" % (data_dir1, L,
                                                       eta * 1000, eps))
            if use_old_data:
                files += glob.glob(r"%s\p%d.%g.%d.*.dat" % (data_dir2, L,
                                                            eta * 1000, eps))
            print(r"[%d / %d]" % (n_count, n_total))
            average(L, eps, files)
            n_count += len(files)
    save_result()


if __name__ == "__main__":
    if platform.system() == "Windows":
        sample_average_Win(0.1, L0=46)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("-L", type=int, help="System size")
        parser.add_argument("-n", type=int, help="Number of files")
        parser.add_argument(
            "--first", type=int, default=None, help="First index of file")
        parser.add_argument(
            "--last", type=int, default=None, help="Last index of file")
        parser.add_argument(
            "--eps", type=int, help="Strength of disorder multiplied by 10000")
        args = parser.parse_args()

        if args.L and args.n:
            save_serials(args.L, args.n)
        elif args.L and args.eps:
            sample_average_Unix(args.L, args.eps, args.first, args.last)
