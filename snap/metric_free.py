# import load_snap
import os
import plot_snap
import glob


def plot_two_panel_all(infile=None):
    if infile is None:
        files = glob.glob("*.bin")
        for filename in files:
            path = filename.replace(".bin", "")
            if not os.path.exists(path):
                os.mkdir("%s" % (path))
            plot_snap.plot_two_panel(
                filename, save=True, overwrite=False, show_tau=True)
    else:
        filename = os.path.basename(infile)
        path = filename.replace(".bin", "")
        if not os.path.exists(path):
            os.mkdir("%s" % path)
        plot_snap.plot_two_panel(
            filename, save=True, overwrite=False, show_tau=True)


def make_movie_all(template="%04d.jpg"):
    files = glob.glob("*.bin")
    for bin_name in files:
        para = plot_snap.get_para(bin_name)
        mv_name = "%d_%g_%g_%g_%g_%d.mp4" % (para["L"], para["eta"],
                                             para["eps"], para["tau"],
                                             para["rho0"], para["seed"])
        img_dir = bin_name.replace(".bin", "")
        imgs = img_dir + os.path.sep + template
        if os.path.isfile(mv_name):
            if os.stat(mv_name).st_mtime < os.stat(bin_name).st_mtime:
                os.remove(mv_name)
                plot_snap.make_movie(imgs, mv_name)
        else:
            plot_snap.make_movie(imgs, mv_name)


if __name__ == "__main__":
    import platform
    import sys
    if platform.system() == "Windows":
        # os.chdir(r"E:\data\random_torque\metric_free\snapshot")
        # os.chdir(r"E:\data\random_torque\metric_free\rotate_metric_free")
        # os.chdir(r"D:\code\VM\VM_metric_free\build\snap")
        # os.chdir(r"E:\data\random_torque\metric_free\rotate")
        os.chdir(r"E:\data\random_torque\metric_free\tmp")
        plot_two_panel_all()
    else:
        try:
            os.chdir("snap")
        except:
            print("No direction 'snap/'.")
            sys.exit()
        if len(sys.argv) == 2:
            if sys.argv[1] == "mv":
                make_movie_all()
            elif sys.argv[1] == "fig":
                plot_two_panel_all()
            else:
                plot_two_panel_all(sys.argv[1])
        else:
            print("argument should be one of `mv` and 'fig'.")
