def swap_key(dict0):
    """ Swap keys of a nested dict.

        Parameters:
        --------
        dict0: dict
            A dict with two layer keys: {A:{B:object}}

        Returns:
        dict1: dict
            A dict with two layer keys: {B:{A:object}}
    """
    dict1 = {}
    for key1 in dict0:
        for key2 in dict0[key1]:
            if key2 in dict1:
                dict1[key2][key1] = dict0[key1][key2]
            else:
                dict1[key2] = {key1: dict0[key1][key2]}
    return dict1


def dict2str(dict0, *args, sep='.'):
    """ Transform dict0 into a string seperated by sep by the order of args.

        Parameters:
        --------
            dict0: dict
                Dict of paramters.
            *args: list
                List of keys.
            sep: str
                Seperator betwwen two numbers in target string.

        Returns:
        --------
            res: str
                String seperated by sep.
    """
    if len(dict0) == 0:
        res = "*"
    else:
        strList = []
        for arg in args:
            if arg in dict0:
                strList.append(dict0[arg])
            else:
                strList.append("*")
        res = sep.join(strList)
    return res


def list2str(list0, *args, sep='.'):
    """ Transform list0 into a string seperated by sep by the order of args.

        Parameters:
        --------
            list0: list
                List with form like: key1, value1, key2, value2...
            *args: list
                List of keys.
            sep: str
                Seperator betwwen two numbers in target string.

        Returns:
        --------
            res: str
                String seperated by sep.
    """
    res = ""
    if len(list0) == 0:
        res = "*"
    else:
        for arg in args:
            if arg in list0 or "-%s" % arg in list0:
                idx = list0.index(arg)
                if len(res) == 0:
                    res = "%s" % (list0[idx + 1])
                else:
                    res += "%s%s" % (sep, list0[idx + 1])
            else:
                if len(res) == 0:
                    res = "*"
                else:
                    res += "%s*" % (sep)
    return res


def get_para(file, sep="."):
    """ Get parameters from filename.

        Parameters:
        --------
            file: str
                Name of input file.

        Returns:
        --------
            para: list
                eta, eps, Lx, Ly, seed
    """
    strList = (file.split("_")[1]).split(sep)
    para = [int(i) for i in strList[:-1]]
    return para
