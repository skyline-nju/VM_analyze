""" Plot standard deviation of gaps between two nearest bands."""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
sys.path.append("../")

try:
    import ana_data
except:
    raise


if __name__ == "__main__":
    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\uniband")

    print("hello, world")
