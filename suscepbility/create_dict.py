"""
Read data from a ".xlsx" file or txt files, and return a dict containing the
needed information.

A ".xlsx" file should have three sheets, with sheet name "phi", "xi", and
"num", respectively. In each sheet, take "L" as columns and "eps" as index.

The filename of txt file is "%.4f.dat" % eps. There are five columns in each
txt file: L, sample average of phi, std of phi over samples, sample size,
sample average of chi, std of chi over samples.
"""

import numpy as np
import pandas as pd
import sys
import os
import glob


def reform_dict(raw_dict,
                k1_min=None,
                k2_min=None,
                form="dict-arr",
                len_m=None):
    """
        Reform the input dict, removing NaN values.

        Parameters:
        --------
        raw_dict: dict
            Input dict such as {k1: {k2: value}}.
        k1_min: float, optional
            Delete k1 below k1_min.
        k2_min: float, optional
            Delete k2 below k2_min.
        form: str, optional
            The form of result dict, one of "dict-arr" and "dict-dict".
        len_m: int, optional
            Remove the array whose size is less than len_m. Only play a role when
            form is "dict-arr".  
        
        Returns:
        --------
        res: dict
            If `form` == "dict-dict", res = {k1: {k2: value}}.
            elif `form` == "dict-arr", res = {k1: np.array([k2], [value])}.
    """
    res = {}
    if form == "dict-dict":
        for k1 in raw_dict:
            if k1_min is not None and k1 < k1_min:
                continue
            for k2 in raw_dict[k1]:
                if k2_min is not None and k2 < k2_min:
                    continue
                if not np.isnan(raw_dict[k1][k2]):
                    if k1 in res:
                        res[k1][k2] = raw_dict[k1][k2]
                    else:
                        res[k1] = {k2: raw_dict[k1][k2]}
    elif form == "dict-arr":
        for k1 in raw_dict:
            if k1_min is not None and k1 < k1_min:
                continue
            arr1 = []
            arr2 = []
            for k2 in sorted(raw_dict[k1].keys()):
                if k2_min is not None and k2 < k2_min:
                    continue
                if not np.isnan(raw_dict[k1][k2]):
                    arr1.append(k2)
                    arr2.append(raw_dict[k1][k2])
            if len_m is None or len(arr1) >= len_m:
                res[k1] = np.array([arr1, arr2])
    else:
        if isinstance(form, str):
            print("augument 'form' should be one of 'dict-dict', 'dict-arr'")
        else:
            print("augument 'form' should be a string")
        sys.exit()
    return res


def create_dict_from_xlsx(filename,
                          sheet_name="chi",
                          key_name="L",
                          eps_min=None,
                          L_min=None,
                          form="dict-arr",
                          len_m=None):
    """
        Create a dict from xlsx file.

        Parameters:
        --------
        filename: str
            Input xlsx file.
        sheet_name: str, optional
            "chi" or "phi".
        key_name: str, optional
            "L" or "eps".
        eps_min: float, optional
            Min epsilon.
        L_min: int, optional
            Min L.
        form: str, optional
            "dict-arr" or "dict-dict".
        len_m: int, optional
            If not None, only return array whose size is larger than len_m.

        Returns:
        --------
        res: dict
            Result dict.
    """
    with pd.ExcelFile(filename) as f:
        df = pd.read_excel(f, sheet_name=sheet_name)
    if key_name == "eps":
        orient = "index"
        k1_min = eps_min
        k2_min = L_min
    elif key_name == "L":
        orient = "dict"
        k1_min = L_min
        k2_min = eps_min
    else:
        print("augument 'key_name' should be one of 'L', 'eps'")
        sys.exit()
    raw_dict = df.to_dict(orient)
    res = reform_dict(raw_dict, k1_min, k2_min, form, len_m)
    return res


def create_dict_from_txt(path,
                         value_name="chi",
                         key_name="L",
                         eps_min=None,
                         L_min=None,
                         form="dict-arr",
                         len_m=None):
    """
        Create a dict from txt files.

        Parameters:
        --------
        path: str
            Input file path.
        value_name: str, optional
            "chi", "chi_dis", "phi"
        key_name: str, optional
            "L" or "eps".
        eps_min: float, optional
            Min epsilon.
        L_min: int, optional
            Min L.
        form: str, optional
            "dict-arr" or "dict-dict".
        len_m: int, optional
            If not None, only return array whose size is larger than len_m.

        Returns:
        --------
        res: dict
            Result dict.
    """
    files = glob.glob(path + os.path.sep + "0.*.dat")
    raw_dict = {}
    for fullname in files:
        basename = os.path.basename(fullname)
        eps = float(basename.replace(".dat", ""))
        with open(fullname) as f:
            lines = f.readlines()
            for line in lines:
                s = line.replace("\n", "").split("\t")
                L = int(s[0])
                if value_name == "chi":
                    value = float(s[4])
                elif value_name == "chi_dis":
                    value = (float(s[2]) * L) ** 2
                elif value_name == "phi":
                    value = float(s[1])
                elif value_name == "num":
                    value = int(s[3])
                if eps in raw_dict:
                    raw_dict[eps][L] = value
                else:
                    raw_dict[eps] = {L: value}
    if key_name == "eps":
        res = reform_dict(raw_dict, eps_min, L_min, form, len_m)
    elif key_name == "L":
        raw_dict = pd.DataFrame.from_dict(raw_dict, orient='index').to_dict()
        res = reform_dict(raw_dict, L_min, eps_min, form, len_m)
    return res


if __name__ == "__main__":
    eta = 0.18
    path = r"data\eta=%g" % eta
    phi_dict = create_dict_from_txt(path, value_name="phi", form="dict-dict")
    chi_dict = create_dict_from_txt(path, value_name="chi", form="dict-dict")
    num_dict = create_dict_from_txt(path, value_name="num", form="dict-dict")
    with pd.ExcelWriter(r"data\eta=0.18.xlsx") as w:
        pd.DataFrame.from_dict(phi_dict).to_excel(w, sheet_name="phi")
        pd.DataFrame.from_dict(chi_dict).to_excel(w, sheet_name="chi")
        pd.DataFrame.from_dict(num_dict).to_excel(w, sheet_name="num")
        
