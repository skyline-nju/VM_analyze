import numpy as np
import glob
import os
import pandas as pd


def read_txt(filename):
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


if __name__ == "__main__":
    os.chdir(r"C:\Users\user\Desktop")
    filename = "p46.100.480.11940000.dat"
    phi, theta = read_txt(filename)
    df_phi = pd.DataFrame(phi, columns=[filename])