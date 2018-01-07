# import load_snap
import os
import plot_snap
import glob


def plot_two_panel_all():
    files = glob.glob("*.bin")
    for filename in files:
        path = filename.replace(".bin", "")
        if not os.path.exists(path):
            os.mkdir("%s" % (path))
        plot_snap.plot_two_panel(
            filename, save=True, overwrite=False, show_tau=True)


def make_movie_all(template="%04d.jpg"):
    files = glob.glob("*.bin")
    for bin_name in files:
        img_dir = bin_name.replace(".bin", "")
        mv_name = img_dir + ".mp4"
        if not os.path.isfile(mv_name) or os.stat(mv_name).st_mtime < os.stat(
                bin_name).st_mtime:
            imgs = img_dir + os.path.sep + template
            plot_snap.make_movie(imgs, mv_name, 1)


if __name__ == "__main__":
    # os.chdir(r"E:\data\random_torque\metric_free\snapshot")
    # os.chdir(r"E:\data\random_torque\metric_free\rotate_metric_free")
    # os.chdir(r"D:\code\VM\VM_metric_free\build\snap")
    # os.chdir(r"E:\data\random_torque\metric_free\rotate")
    os.chdir(r"E:\data\random_torque\metric_free\tmp")
    plot_two_panel_all()
    # make_movie_all()
    # mv_files = glob.glob("*.mp4")
    # for name in mv_files:
    #     s = name.replace(".mp4", "").split("_")
    #     eta = float(s[1])
    #     L = float(s[3])
    #     if len(s) == 10:
    #         tau = float(s[9])
    #     else:
    #         tau = 0
    #     new_name = "L=%d eta=%g rho=%g tau=%gpi.mp4" % (L, eta, 4, tau)
    #     os.rename(name, new_name)
