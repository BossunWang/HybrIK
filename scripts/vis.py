import os
from pathlib import Path
from tempfile import TemporaryDirectory

import librosa as lr
import numpy as np
import soundfile as sf
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import matplotlib.pyplot as plt


# from SMPLX
# indices of parents for each joints
# parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
# parents[0] = -1
# self.register_buffer('parents', parents)

smpl_parents = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    15,
    15,
    15,
    20,
    25,
    26,
    20, # left hand
    28,
    29,
    20,
    31,
    32,
    20,
    34,
    35,
    20,
    37,
    38,
    21, # right hand
    40,
    41,
    21,
    43,
    44,
    21,
    46,
    47,
    21,
    49,
    50,
    21,
    52,
    53,
]

def set_line_data_3d(line, x):
    line.set_data(x[:, :2].T)
    line.set_3d_properties(x[:, 2])


def set_scatter_data_3d(scat, x, c):
    scat.set_offsets(x[:, :2])
    scat.set_3d_properties(x[:, 2], "z")
    scat.set_facecolors([c])


def get_axrange(poses):
    pose = poses[0]
    x_min = pose[:, 0].min()
    x_max = pose[:, 0].max()

    y_min = pose[:, 1].min()
    y_max = pose[:, 1].max()

    z_min = pose[:, 2].min()
    z_max = pose[:, 2].max()

    xdiff = x_max - x_min
    ydiff = y_max - y_min
    zdiff = z_max - z_min

    biggestdiff = max([xdiff, ydiff, zdiff])
    return biggestdiff


def plot_single_pose(num, poses, lines, ax, axrange, scat):
    pose = poses[num]

    for i, (p, line) in enumerate(zip(smpl_parents, lines)):
        # don't plot root
        if i == 0:
            continue
        # stack to create a line
        data = np.stack((pose[i], pose[p]), axis=0)
        set_line_data_3d(line, data)

    if num == 0:
        if isinstance(axrange, int):
            axrange = (axrange, axrange, axrange)
        xcenter, ycenter, zcenter = 0, 0, 0
        stepx, stepy, stepz = axrange[0] / 2, axrange[1] / 2, axrange[2] / 2

        x_min, x_max = xcenter - stepx, xcenter + stepx
        y_min, y_max = ycenter - stepy, ycenter + stepy
        z_min, z_max = zcenter - stepz, zcenter + stepz

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)


def skeleton_render(
        poses,
        epoch=0,
        out="renders",
        name="",
        sound=True,
        stitch=False,
        contact=None,
        render=True
):
    # generate the pose with FK
    Path(out).mkdir(parents=True, exist_ok=True)
    num_steps = poses.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # set floor the first frame of first frame
    point = np.array([0, 0, np.min(poses[0, :, :])])
    normal = np.array([0, 0, 1])
    d = -point.dot(normal)
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 2), np.linspace(-1.5, 1.5, 2))
    z = (-normal[0] * xx - normal[1] * yy - d) * 1.0 / normal[2]
    # plot the plane
    ax.plot_surface(xx, yy, z, zorder=-11, cmap=cm.twilight)
    # Create lines initially without data
    lines = [
        ax.plot([], [], [], zorder=10, linewidth=1.5)[0]
        for _ in smpl_parents
    ]
    scat = [
        ax.scatter([], [], [], zorder=10, s=0, cmap=ListedColormap(["r", "g", "b"]))
        for _ in range(4)
    ]
    axrange = 3

    if render:
        # Creating the Animation object
        anim = animation.FuncAnimation(
            fig,
            plot_single_pose,
            num_steps,
            fargs=(poses, lines, ax, axrange, scat),
            interval=1000 // 30,
        )

        # actually save the gif
        path = os.path.normpath(name)
        pathparts = path.split(os.sep)
        gifname = os.path.join(out, f"{pathparts[0]}.gif")
        anim.save(gifname, savefig_kwargs={"transparent": True, "facecolor": "none"}, )

        outname = os.path.join(
            out,
            f'{epoch}_{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.mp4',
        )

        print(outname)

        out = os.system(
            f"ffmpeg -loglevel error -stream_loop 0 -y -i '{gifname}' -shortest -c:v libx264 -c:a aac -q:a 4 '{outname}'"
        )
    else:
        # debug
        plot_single_pose(0, poses, lines, ax, axrange, scat)
        plt.show()
    plt.close()

