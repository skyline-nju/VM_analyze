"""
    Plot coarse-grained snapshot of density and velocity fields.
"""

import os
import numpy as np
import glob
import platform
import matplotlib
if platform.system() != "Windows":
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
else:
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb


def map_v_to_rgb(theta, module, m_max=None):
    """
    Transform orientation and magnitude of velocity into rgb.

    Parameters:
    --------
    theta: array_like
        Orietation of velocity field.
    module: array_like
        Magnitude of velocity field.
    m_max: float, optional
        Max magnitude to show.

    Returns:
    --------
    RGB: array_like
        RGB corresponding to velocity fields.
    """
    H = theta / 360
    V = module
    if m_max is not None:
        V[V > m_max] = m_max
    V /= m_max
    S = np.ones_like(H)
    HSV = np.dstack((H, S, V))
    RGB = hsv_to_rgb(HSV)
    return RGB


def add_colorbar(ax, mmin, mmax, theta_min=0, theta_max=360, orientation="h"):
    """ Add colorbar for the RGB image plotted by plt.imshow() """
    V, H = np.mgrid[0:1:50j, 0:1:180j]
    if orientation == "v":
        V = V.T
        H = H.T
        box = [mmin, mmax, theta_min, theta_max]
    else:
        box = [theta_min, theta_max, mmin, mmax]
    S = np.ones_like(V)
    HSV = np.dstack((H, S, V))
    RGB = hsv_to_rgb(HSV)
    ax.imshow(RGB, origin='lower', extent=box, aspect='auto')
    theta_ticks = [0, 45, 90, 135, 180, 225, 270, 315, 360]

    if orientation == "h":
        ax.set_xticks(theta_ticks)
        ax.set_xticklabels([r"$%d\degree$" % i for i in theta_ticks])
        ax.set_ylabel(r'module $\rho |v|$', fontsize="large")
        ax.set_xlabel("orientation", fontsize="large")
    else:
        ax.yaxis.set_label_position('right')
        ax.yaxis.set_ticks_position("right")
        ax.set_yticks(theta_ticks)
        ax.set_yticklabels([r"$%d\degree$" % i for i in theta_ticks])
        ax.set_ylabel(r'orientation $\theta$', fontsize="large")
        ax.set_xlabel(r"module $\rho |v|$", fontsize="large")


def map_sin2theta_to_rgb(theta, module, Hue, m_max):
    """
    Map sin(2 theta)^2 and module into rgb.
    """
    S0 = np.sin(2 * theta)**2 - 0.5
    S = np.abs(S0) * 2
    V = module / m_max
    H = np.ones_like(S) * Hue
    Hue2 = Hue + 0.5
    if Hue2 > 1:
        Hue2 -= 1
    H[S0 > 0] = Hue + 0.5
    HSV = np.dstack((H, S, V))
    RGB = hsv_to_rgb(HSV)
    return RGB


def add_cb_sin2theta(ax, Hue):
    S, V = np.mgrid[0:1:50j, 0:1:50j]
    box = [0, 1, 0, 1]
    H = np.ones_like(V) * Hue
    HSV = np.dstack((H, V, S))
    RGB = hsv_to_rgb(HSV)
    ax.imshow(RGB, origin="lower", extent=box, aspect="auto")


def plot_sin2theta(bin_file):
    from load_snap import CoarseGrainSnap
    para = get_para(bin_file)
    domain = [0, para["L"], 0, para["L"]]
    snap = CoarseGrainSnap(bin_file)
    frames = snap.gene_frames()
    for frame in frames:
        t, vxm, vym, num, vx, vy = frame
        ax = plt.subplot(111)
        theta = np.arctan2(vy, vx)
        module = np.sqrt(vx**2 + vy**2)
        RGB = map_sin2theta_to_rgb(theta, module, 0, para["rho0"])
        ax.imshow(RGB, origin="lower", extent=domain, interpolation="none")
        plt.show()
        plt.close()


def get_para(filename):
    s = filename.replace(".bin", "").split("_")
    dict_para = {}
    dict_para["eta"] = float(s[1])
    dict_para["eps"] = float(s[2])
    dict_para["L"] = int(s[3])
    dict_para["ncols"] = int(s[5])
    dict_para["N"] = int(s[7])
    dict_para["rho0"] = dict_para["N"] / dict_para["L"]**2
    dict_para["seed"] = int(s[8])
    if len(s) == 10:
        dict_para["tau"] = float(s[9])
    return dict_para


def get_fig_title(para, phi, t, show_tau=False, prefix=""):
    t1 = r"$\eta=%g,\ \epsilon=%g,\ " % (para["eta"], para["eps"])
    t2 = r"\rho_0=%g,\ L=%d,\ \phi=%.4f,\ t=%d$" % (para["rho0"], para["L"],
                                                    phi, t)
    if "tau" in para:
        title = prefix + t1 + r"\tau=%g\pi,\ " % (para["tau"]) + t2
    else:
        if show_tau:
            title = prefix + t1 + r"\tau=0,\ " + t2
        else:
            title = prefix + t1 + t2
    return title


def plot_two_panel(bin_file,
                   t_list=None,
                   save=False,
                   overwrite=False,
                   v_normed=True,
                   show_tau=False,
                   title_prefix=""):
    """ Plot density and velocity filed on left, right pannel, respectively.
        One picture per frame.

    Parameters:
    ---------
    bin_file: str
        Input binary file.
    t_list: array_like, optional
        Which frames to show. If None, show all frames.
    save: bool, optional
        Whether to save the images to disk.
    overwrite: bool, optional
        Whether to overwrite old images.
    v_normed: bool, optional
        Whether the velocities have been normed by density.
    show_tau: bool, optional
         If True, show the value of `tau` in the title of image.
    title_prefix: string, optional
        The prefix of image title.
    """
    if save:
        folder = bin_file.replace(".bin", "")
        path = folder + os.path.sep
        fig_template = path + "%04d.jpg"
        old_fig = glob.glob(path + "*.jpg")
    para = get_para(bin_file)

    snap = load_snap.CoarseGrainSnap(bin_file)
    frames = snap.gene_frames()
    domain = [0, para["L"], 0, para["L"]]
    lBox = para["L"] // para["ncols"]
    dA = lBox**2
    x = y = np.linspace(lBox / 2, para["L"] - lBox / 2, para["ncols"])
    # rho_level = np.linspace(0, 5, 11)
    if para["rho0"] < 2:
        rho_level = np.linspace(0, 4, 9)
    elif para["rho0"] < 4:
        rho_level = np.linspace(0, 6, 13)
    else:
        rho_level = np.linspace(0, 10, 9)
    iframe = 1
    for frame in frames:
        t, vxm, vym, num, vx, vy = frame
        if t_list is None or t in t_list:
            if save:
                fig_name = fig_template % iframe
                if fig_name in old_fig:
                    if overwrite:
                        print("overwrite the %d-th frame at t = %d" % (iframe,
                                                                       t))
                    else:
                        print("skip the %d-th frame at t = %d" % (iframe, t))
                        iframe += 1
                        continue
                else:
                    print("save the %d-th frame at t = %d" % (iframe, t))
                iframe += 1
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 7.5))
            rho = num / dA
            contour = ax1.contourf(x, y, rho, rho_level, extend="max")
            ax1.axis("scaled")
            ax1.axis(domain)

            v_orient = np.arctan2(vy, vx) / np.pi * 180
            v_orient[v_orient < 0] += 360
            if v_normed:
                v_module = np.sqrt(vx**2 + vy**2)
            else:
                v_module = np.sqrt(vx**2 + vy**2) * rho
            RGB = map_v_to_rgb(v_orient, v_module, m_max=4)
            ax2.imshow(RGB, extent=domain, origin="lower")
            ax2.axis('scaled')
            ax2.axis(domain)
            phi = np.sqrt(vxm**2 + vym**2)

            plt.suptitle(
                get_fig_title(para, phi, t, show_tau, title_prefix),
                fontsize="xx-large",
                y=0.99)
            plt.tight_layout(rect=[0, 0, 1, 0.98])

            bbox1 = ax1.get_position().get_points().flatten()
            bbox2 = ax2.get_position().get_points().flatten()
            fig.subplots_adjust(bottom=0.24)
            bbox1[1], bbox1[3] = 0.14, 0.04
            bbox1[2] = bbox1[2] - bbox1[0] - 0.03
            bbox2[1], bbox2[3] = 0.08, 0.14
            bbox2[2] = bbox2[2] - bbox2[0]
            cb_ax1 = fig.add_axes(bbox1)
            cb_ax2 = fig.add_axes(bbox2)
            cb1 = fig.colorbar(contour, cax=cb_ax1, orientation="horizontal")
            cb1.set_label(r"density $\rho$", fontsize="x-large")
            add_colorbar(cb_ax2, 0, para["rho0"], 0, 360)
            if save:
                plt.savefig(fig_name)
            else:
                plt.show()
            plt.close()


def plot_serial_snap(file, save=False, rescale=False):
    """ Plot density (upper row) and velocity (lower row), with incresing time
        from left to right.

    Parameters:
    --------
    file: str
        Input file.
    save: bool, optional
        If true, save the figure into disk.
    rescale: bool, optional
        If true, xlim, ylim increase linearly with incresing time.
    """
    from load_snap import CoarseGrainSnap
    t_list = [400, 800, 1600, 3200]
    snap = CoarseGrainSnap(file)
    frames = snap.gene_frames()
    s = file.replace(".bin", "").split("_")
    eta = float(s[1])
    eps = float(s[2])
    L = int(s[3])
    ncols = int(s[5])
    lBox = L // ncols
    dA = lBox**2
    x = y = np.linspace(lBox / 2, L - lBox / 2, ncols)
    rho_level = np.linspace(0, 3, 7)
    fig, axes = plt.subplots(nrows=2, ncols=len(t_list), figsize=(12, 6))
    col = 0
    if rescale:
        i = ncols // 8
    else:
        i = ncols
    for frame in frames:
        t, vxm, vym, num, vx, vy = frame
        if t in t_list:
            rho = num / dA
            contour = axes[0][col].contourf(
                x[0:i], y[0:i], rho[0:i, 0:i], rho_level, extend="max")
            axes[0][col].axis("scaled")
            axes[0][col].axis([0, i * lBox, 0, i * lBox])
            axes[0][col].axis("off")
            axes[0][col].set_title(r"$t=%d$" % (t), fontsize="x-large")

            v_orient = np.arctan2(vy[0:i, 0:i], vx[0:i, 0:i]) / np.pi * 180
            v_orient[v_orient < 0] += 360
            v_module = np.sqrt(vx[0:i, 0:i]**2 + vy[0:i, 0:i]**2) * rho[0:i, 0:
                                                                        i]
            RGB = map_v_to_rgb(v_orient, v_module, m_max=4)
            axes[1][col].axis('scaled')
            axes[1][col].imshow(
                RGB, extent=[0, i * lBox, 0, i * lBox], origin="lower")
            axes[1][col].axis([0, i * lBox, 0, i * lBox])
            axes[1][col].axis("off")
            col += 1
            if rescale:
                i *= 2

    # axes[0][0].set_title("density", fontsize="x-large", loc="left")
    # axes[1][0].set_title("velocity", fontsize="x-large", loc="left")
    plt.suptitle(
        r"$\eta=%g,\ \rho_0=1,\ \epsilon=%g,\ L=%d$" % (eta, eps, L),
        fontsize="xx-large",
        y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=0.005, w_pad=0.005)

    fig.subplots_adjust(right=0.90)

    # positon of the last column: x0, y0, x1, y1
    bbox1 = axes[0][col - 1].get_position().get_points().flatten()
    bbox2 = axes[1][col - 1].get_position().get_points().flatten()

    # add axes for colobar, whose position is [left, bottom, width, height]
    dy = 0.015
    cb_ax1 = fig.add_axes(
        [0.92, bbox1[1] + dy, 0.02, bbox1[3] - bbox1[1] - 2 * dy])
    cb1 = fig.colorbar(contour, cax=cb_ax1)
    cb1.set_label(r"density $\rho$", fontsize="x-large")
    cb_ax2 = fig.add_axes(
        [0.91, bbox2[1] + dy, 0.04, bbox2[3] - bbox2[1] - 2 * dy])
    add_colorbar(cb_ax2, 0, 4, 0, 360, orientation="v")
    if save:
        plt.savefig(
            r"../fig/snap_%d_%g_%g.jpg" % (L, eta * 100, eps * 100),
            bbox_inches="tight",
            pad_inches=0.02,
            dpi=300)
    else:
        plt.show()
    plt.close()


def make_movie(img_template, mv_name, start_num=0, rate=15, vframes=None):
    """ Make a movie by merging the images utilizing ffmpeg. """
    import subprocess
    strcmd = r"ffmpeg -f image2 -r %d -start_number %d -i %s " % (rate,
                                                                  start_num,
                                                                  img_template)
    if vframes is not None:
        strcmd += "-vframes %d %s " % vframes
    strcmd += "-preset veryslow -crf 34 %s" % (mv_name)
    subprocess.call(strcmd, shell=True)


if __name__ == "__main__":
    # os.chdir("data")
    # file = r"cHff_0.1_0_8192_8192_1024_1024_67108864_17102532.bin"
    # file = r"cHff_0.18_0_8192_8192_1024_1024_67108864_17091901.bin"
    # file = r"cHff_0.18_0_8192_8192_4096_4096_67108864_17111451.bin"
    # file = r"cHff_0.35_0_8192_8192_1024_1024_67108864_17092802.bin"
    # file = r"cHff_0.18_0.02_8192_8192_1024_1024_67108864_17120201.bin"
    # file = r"cHff_0.18_0.04_8192_8192_1024_1024_67108864_17120201.bin"
    # file = r"cHff_0.18_0.06_8192_8192_1024_1024_67108864_17120201.bin"
    # file = r"cHff_0.4_0_8192_8192_1024_1024_67108864_17110541.bin"
    # plot_two_panel(file, v_normed=False)
    # plot_serial_snap(file, save=True, rescale=False)

    os.chdir(r"E:\data\random_torque\metric_free\rotate_metric_free")
    bin_file = r"cHff_0.1_0_512_512_512_512_1048576_124_0.1.bin"
    plot_sin2theta(bin_file)
