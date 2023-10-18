from vis import skeleton_render
import vis
import numpy as np
import os
from matplotlib import pyplot as plt
from rffridge import RFFRidgeRegression
from rffgpr import RFFGaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic
from sklearn.kernel_ridge import KernelRidge
from scipy.signal import argrelextrema
from BZcurve import bezier_curve

def run(out_dir, pred_xyz_full_list, cam_root_list, transl_list):
    smpl_parents_size = len(vis.smpl_parents)

    # Body
    # pred_xyz_full_list[:, :, :25]

    # Fit kernel ridge regression using random Fourier features.
    rff_dim = len(transl_list) // 5
    X = np.arange(len(transl_list))[:, None]
    Y = transl_list

    # Fit kernel ridge regression using random Fourier features.
    clfx = RFFRidgeRegression(rff_dim=rff_dim)
    clfy = RFFRidgeRegression(rff_dim=rff_dim)
    clfz = RFFRidgeRegression(rff_dim=rff_dim)
    clfx.fit(X, Y[:, 0, 0].reshape(-1))
    clfy.fit(X, Y[:, 0, 1].reshape(-1))
    clfz.fit(X, Y[:, 0, 2].reshape(-1))
    y_predx = clfx.predict(X)
    y_predy = clfy.predict(X)
    y_predz = clfz.predict(X)

    # Fit kernel ridege regression using an RBF kernel.
    # Z are incorrect for non-normalize
    # fitting not good for normalize
    # clfx = KernelRidge(kernel=RBF())
    # clfy = KernelRidge(kernel=RBF())
    # clfz = KernelRidge(kernel=RBF())
    # min_x = np.min(Y[:, :, 0])
    # min_y = np.min(Y[:, :, 1])
    # min_z = np.min(Y[:, :, 2])
    # max_x = np.max(Y[:, :, 0])
    # max_y = np.max(Y[:, :, 1])
    # max_z = np.max(Y[:, :, 2])
    # inter_x = max_x - min_x
    # inter_y = max_y - min_y
    # inter_z = max_z - min_z
    # clfx.fit(X, (Y[:, 0, 0].reshape(-1) - min_x) / inter_x)
    # clfy.fit(X, (Y[:, 0, 1].reshape(-1) - min_y) / inter_y)
    # clfz.fit(X, (Y[:, 0, 2].reshape(-1) - min_z) / inter_z)
    # y_predx = clfx.predict(X)
    # y_predy = clfy.predict(X)
    # y_predz = clfz.predict(X)
    #
    # y_predx = y_predx * inter_x + min_x
    # y_predy = y_predy * inter_y + min_y
    # y_predz = y_predz * inter_z + min_z

    # Fit Gaussian Process regression using random Fourier features.
    # clfx = RFFGaussianProcessRegressor(rff_dim=rff_dim)
    # clfy = RFFGaussianProcessRegressor(rff_dim=rff_dim)
    # clfz = RFFGaussianProcessRegressor(rff_dim=rff_dim)
    # clfx.fit(X, Y[:, 0, 0].reshape(-1))
    # clfy.fit(X, Y[:, 0, 1].reshape(-1))
    # clfz.fit(X, Y[:, 0, 2].reshape(-1))
    # y_predx, y_covx = clfx.predict(X)
    # y_predy, y_covy = clfy.predict(X)
    # y_predz, y_covz = clfz.predict(X)

    motion_peaks_x = argrelextrema(y_predx, np.greater)[0]
    motion_valleys_x = argrelextrema(y_predx, np.less)[0]

    motion_peaks_y = argrelextrema(y_predy, np.greater)[0]
    motion_valleys_y = argrelextrema(y_predy, np.less)[0]

    motion_peaks_z = argrelextrema(y_predz, np.greater)[0]
    motion_valleys_z = argrelextrema(y_predz, np.less)[0]

    pairs_pointx = np.stack((X.reshape(-1)[motion_peaks_x], y_predx[motion_peaks_x]), axis=-1)
    xvals1, yvals1 = bezier_curve(pairs_pointx, nTimes=len(transl_list))

    pairs_pointy = np.stack((X.reshape(-1)[motion_peaks_y], y_predy[motion_peaks_y]), axis=-1)
    xvals2, yvals2 = bezier_curve(pairs_pointy, nTimes=len(transl_list))

    pairs_pointz = np.stack((X.reshape(-1)[motion_peaks_z], y_predz[motion_peaks_z]), axis=-1)
    xvals3, yvals3 = bezier_curve(pairs_pointz, nTimes=len(transl_list))

    plt.figure('RFF ridge regression x')
    cmapb = plt.cm.get_cmap('Blues')
    cmapr = plt.cm.get_cmap('Reds')
    cmapg = plt.cm.get_cmap('Greens')
    plt.scatter(X, Y[:, 0, 0], s=30, c=[cmapb(0.3)])
    plt.scatter(X[motion_peaks_x], y_predx[motion_peaks_x], s=30, c=[cmapb(0.9)])
    plt.plot(X, y_predx, c=cmapr(0.9))
    plt.plot(xvals1, yvals1, c=cmapg(0.9))
    plt.savefig(os.path.join(out_dir, 'RFF_x'))
    plt.close()
    plt.cla()
    plt.clf()

    plt.figure('RFF ridge regression y')
    plt.scatter(X, Y[:, 0, 1], s=30, c=[cmapb(0.3)])
    plt.scatter(X[motion_peaks_y], y_predy[motion_peaks_y], s=30, c=[cmapb(0.9)])
    plt.plot(X, y_predy, c=cmapr(0.9))
    plt.plot(xvals2, yvals2, c=cmapg(0.9))
    plt.savefig(os.path.join(out_dir, 'RFF_y'))
    plt.close()
    plt.cla()
    plt.clf()

    plt.figure('RFF ridge regression z')
    plt.scatter(X, Y[:, 0, 2], s=30, c=[cmapb(0.3)])
    plt.scatter(X[motion_peaks_z], y_predz[motion_peaks_z], s=30, c=[cmapb(0.9)])
    plt.plot(X, y_predz, c=cmapr(0.9))
    plt.plot(xvals3, yvals3, c=cmapg(0.9))
    plt.savefig(os.path.join(out_dir, 'RFF_z'))
    plt.close()
    plt.cla()
    plt.clf()

    # plt.show()

    # plt.figure('root pos x')
    # plt.plot(np.arange(len(pred_xyz_full_list)), pred_xyz_full_list[:, 0, 0])
    # plt.figure('root pos y')
    # plt.plot(np.arange(len(pred_xyz_full_list)), pred_xyz_full_list[:, 0, 1])
    # plt.figure('root pos z')
    # plt.plot(np.arange(len(pred_xyz_full_list)), pred_xyz_full_list[:, 0, 2])
    #
    # plt.figure('camera root x')
    # plt.plot(np.arange(len(cam_root_list)), cam_root_list[:, 0, 0])
    # plt.figure('camera root y')
    # plt.plot(np.arange(len(cam_root_list)), cam_root_list[:, 0, 1])
    # plt.figure('camera root z')
    # plt.plot(np.arange(len(cam_root_list)), cam_root_list[:, 0, 2])
    #
    # plt.figure('transl x')
    # plt.plot(np.arange(len(transl_list)), transl_list[:, 0, 0])
    # plt.figure('transl y')
    # plt.plot(np.arange(len(transl_list)), transl_list[:, 0, 1])
    # plt.figure('transl z')
    # plt.plot(np.arange(len(transl_list)), transl_list[:, 0, 2])
    # plt.show()

    os.makedirs(os.path.join(out_dir, 'res_pose'), exist_ok=True)

    yvals1 = yvals1[::-1]
    yvals2 = yvals2[::-1]
    yvals3 = yvals3[::-1]

    transl_list[:, :, 0] = yvals1[:, None]
    transl_list[:, :, 1] = yvals2[:, None]
    transl_list[:, :, 2] = -yvals3[:, None]

    # transl_list[:, :, 0] = y_predx[:, None]
    # transl_list[:, :, 1] = y_predy[:, None]
    # transl_list[:, :, 2] = y_predz[:, None]

    pred_xyz_full_list = pred_xyz_full_list * 2.2 + transl_list
    # normalize to 0 ~ 1
    min_x = np.min(pred_xyz_full_list[:, :, 0])
    min_y = np.min(pred_xyz_full_list[:, :, 1])
    min_z = np.min(pred_xyz_full_list[:, :, 2])

    inter_x = np.max(pred_xyz_full_list[:, :, 0]) - min_x
    inter_y = np.max(pred_xyz_full_list[:, :, 1]) - min_y
    inter_z = np.max(pred_xyz_full_list[:, :, 2]) - min_z

    pred_xyz_full_list[:, :, 0] = (pred_xyz_full_list[:, :, 0] - min_x) / inter_x
    pred_xyz_full_list[:, :, 1] = (pred_xyz_full_list[:, :, 1] - min_y) / inter_y
    pred_xyz_full_list[:, :, 2] = (pred_xyz_full_list[:, :, 2] - min_z) / inter_z

    pred_xyz_full_list_tmp = pred_xyz_full_list.copy()
    pred_xyz_full_list[:, :, 0] = pred_xyz_full_list_tmp[:, :, 0]
    pred_xyz_full_list[:, :, 1] = pred_xyz_full_list_tmp[:, :, 2]
    pred_xyz_full_list[:, :, 2] = -pred_xyz_full_list_tmp[:, :, 1]

    # translate to (0, 0, 0)
    offset = pred_xyz_full_list[0, 0, :]
    pred_xyz_full_list -= offset[None, None, :3]

    skeleton_render(pred_xyz_full_list, out=os.path.join(out_dir, 'res_pose'), name='test', render=True)


if __name__ == '__main__':
    out_dir = "../(G)-idle - Queencard MIRROR_output"
    pred_xyz_full_list = np.load(os.path.join(out_dir, 'pose_data.npy'))
    cam_root_list = np.load(os.path.join(out_dir, 'cam_data.npy'))
    transl_list = np.load(os.path.join(out_dir, 'transl_data.npy'))

    run(out_dir, pred_xyz_full_list, cam_root_list, transl_list)

    out_dir = "../Queencard-tutorial_slowmotion_output"
    pred_xyz_full_list = np.load(os.path.join(out_dir, 'pose_data.npy'))
    cam_root_list = np.load(os.path.join(out_dir, 'cam_data.npy'))
    transl_list = np.load(os.path.join(out_dir, 'transl_data.npy'))

    run(out_dir, pred_xyz_full_list, cam_root_list, transl_list)

    out_dir = "../gBR_sBM_c04_d04_mBR3_ch08_output"
    pred_xyz_full_list = np.load(os.path.join(out_dir, 'pose_data.npy'))
    cam_root_list = np.load(os.path.join(out_dir, 'cam_data.npy'))
    transl_list = np.load(os.path.join(out_dir, 'transl_data.npy'))

    run(out_dir, pred_xyz_full_list, cam_root_list, transl_list)