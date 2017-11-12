"""
Calculate the susceptibility from fluctuations on order parameters.
"""
import numpy as np
import argparse
import glob
import os
import sys


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


def sample_average(L, eps, first=None, last=None, eta=180):
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
            print(file)
    phi = np.array(phi)
    chi = np.array(chi) * L**2
    print("%d\t%f\t%f\t%d\t%f\t%f\n" % (L, np.mean(phi), np.std(phi), phi.size,
                                        np.mean(chi), np.std(chi)))


if __name__ == "__main__":
    import platform
    if platform.system() == "Windows":
        os.chdir(r"D:\data\susceptibility")
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
        sample_average(args.L, args.eps, args.first, args.last)
