# import load_snap
import os
import plot_snap
import glob
# import load_snap
# import matplotlib.pyplot as plt

if __name__ == "__main__":
    # os.chdir(r"E:\data\random_torque\metric_free\snapshot")
    os.chdir(r"E:\data\random_torque\metric_free\rotate_metric_free")
    files = glob.glob("*.bin")
    for filename in files:
        path = filename.replace(".bin", "")
        if not os.path.exists(path):
            os.mkdir("%s" % (path))
        plot_snap.plot_two_panel(filename, save=True, overwrite=False)
    # snap = load_snap.CoarseGrainSnap(filename)
    # frames = snap.gene_frames()
    # for frame in frames:
    #     t, vxm, vym, num, vx, vy = frame
    #     print(t, vxm, vym)
    #     plt.imshow(num)
    #     plt.show()
    #     plt.close()
    # files = glob.glob("cHff_0.1_*.bin")
    # for file in files:
    #     os.mkdir("%s" % (file.replace(".bin", "")))
    #     plot_snap.plot_two_panel(file, t_list="full", save=True)