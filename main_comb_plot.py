from copy import copy
import numpy as np
from scipy import io as sio
from pathos import multiprocessing as mp
from tqdm import tqdm
from matplotlib import pyplot as plt

from model.AlterMag import *
from task import inverse_Hamiltonian as ih
from task import quasi_particle_interference as qpi
from task import origin_style_plot as oplt
from main import convolve_matrix_2d, gaussian_delta

SR2 = np.sqrt(2)
atom_r = 0.4
rez = 4


def main_qpi_disp():
    fname_0 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.1,Vi0.2,E0" + ".mat"
    fname_2 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.1,Vi0.2,E0.02" + ".mat"
    fname_4 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.1,Vi0.2,E0.04" + ".mat"
    fname_6 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.1,Vi0.2,E0.06" + ".mat"
    fname_8 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.1,Vi0.2,E0.08" + ".mat"
    fname_10 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.1,Vi0.2,E0.1" + ".mat"
    fname_m2 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.1,Vi0.2,E-0.02" + ".mat"
    fname_m4 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.1,Vi0.2,E-0.04" + ".mat"
    fname_m6 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.1,Vi0.2,E-0.06" + ".mat"
    fname_m8 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.1,Vi0.2,E-0.08" + ".mat"
    fname_m10 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.1,Vi0.2,E-0.1" + ".mat"

    fname_ls = [
        fname_m10,
        fname_m8,
        fname_m6,
        fname_m4,
        fname_m2,
        fname_0,
        fname_2,
        fname_4,
        fname_6,
        fname_8,
        fname_10,
    ]

    def get_qpi_x(fname):
        # 读取
        file = sio.loadmat(fname)
        x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
        # DoS_0(x1,x2)
        x0_ls = x_ls[0]
        y0_ls = y_ls[:, 0]
        x1_ls, y1_ls = [np.linspace(0, 200, rez * 200 + 1)] * 2
        gd = lambda x: gaussian_delta(x, sigma=atom_r)
        pos_atom = np.repeat([[0, 0], [0.5, 0], [0.5, -0.5], [0, -0.5]], 2, axis=0)
        ldos_0_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
            ldos_diff_ls * np.tile([1, +1], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
        )
        # QPI
        qpi_0_ls = qpi.calculate_qpi_from_ldos_2d(ldos_0_ls)
        qpi_x = [qpi_0_ls[rez * 100 + i, rez * 100 + i] for i in np.arange(rez * 100)]
        return qpi_x

    qpi_x_ls = [get_qpi_x(fname) for fname in tqdm(fname_ls)]

    # 保存
    eng_ls = np.arange(-0.1, 0.11, 0.02)
    sio.savemat("./data/" + "QPI(qx,E).mat", {"eng_ls": eng_ls, "qpi_x_ls": qpi_x_ls})


def main_qpi_disp_plot():
    fname = "./data/" + "QPI(qx,E).mat"

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 1, 5 * 1), dpi=100).set_layout_engine("compressed")

    # 读取
    file = sio.loadmat(fname)
    eng_ls, qpi_x_ls = file["eng_ls"][0], file["qpi_x_ls"]
    # 子图
    plt.subplot(1, 1, 1)
    plt.pcolormesh(np.arange(qpi_x_ls.shape[1]), eng_ls, np.abs(qpi_x_ls), cmap="viridis", clim=(0, 8))
    plt.xlim([60, 160])
    plt.xlabel(r"$q_x$")
    plt.ylabel(r"$E$")
    plt.colorbar()
    #
    # oplt.number_subplots()
    plt.show()


def main_qpi_d90():
    fname_2 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi0.2,E0.04" + ".mat"

    # 读取
    file = sio.loadmat(fname_2)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    # DoS_0(x1,x2)
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    x1_ls, y1_ls = [np.linspace(0 + 0, 200, rez * 200)] * 2
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    pos_atom = np.repeat([[0, 0], [0.5, 0], [0.5, -0.5], [0, -0.5]], 2, axis=0)
    ldos_0_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, 1], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # QPI
    qpi_0_ls = qpi.calculate_qpi_from_ldos_2d(ldos_0_ls, True)[1:, 1:]
    qpi_9_ls = rotate90_mtx(qpi_0_ls)
    # qpi_9_ls = mirror110_mtx(qpi_0_ls)

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 3, 5 * 1), dpi=100).set_layout_engine("compressed")
    # 子图
    plt.subplot(131)
    plt.pcolormesh(np.abs(qpi_0_ls), cmap="viridis", clim=(0, 8))
    plt.axis("scaled")
    plt.xlabel(r"$q_1$")
    plt.ylabel(r"$q_2$")
    plt.colorbar()
    # 子图
    plt.subplot(132)
    plt.pcolormesh(np.abs(qpi_9_ls), cmap="viridis", clim=(0, 8))
    plt.axis("scaled")
    plt.xlabel(r"$q_1$")
    plt.ylabel(r"$q_2$")
    plt.colorbar()
    # 子图
    qpi_d_ls = np.abs(qpi_0_ls) - np.abs(qpi_9_ls)
    plt.subplot(133)
    plt.pcolormesh(qpi_d_ls, cmap="RdBu_r", clim=(-4, 4))
    plt.axis("scaled")
    plt.xlabel(r"$q_1$")
    plt.ylabel(r"$q_2$")
    plt.colorbar()
    #
    oplt.number_subplots()
    plt.show()

    sio.savemat("./data/QPI_d90.mat", {"qpi_d90": qpi_d_ls})
    # sio.savemat("./data/QPI_dm110.mat", {"qpi_dm110": qpi_d_ls})


def rotate90_mtx(mtx0):
    mtx9 = np.transpose(mtx0)[:, ::-1]
    return mtx9


def mirror110_mtx(mtx0):
    mtx11 = mtx0[::-1, :]
    return mtx11


def main_s1_before_fold():
    fname_ek = "./data/" + "E(k)_KV2Se2O-kk" + ".mat"
    fname_fs = "./data/" + "FmSurf_KV2Se2O-kk" + ".mat"

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 3, 5 * 1 + 1), dpi=100).set_layout_engine("compressed")
    #
    # 示意图
    plt.subplot(131)
    plt.xticks([])
    plt.yticks([])
    #
    # 色散
    # 读取
    file = sio.loadmat(fname_ek)
    k_sym_ls, eng_ls, spin_ls = file["k_path"][:, 0], file["eng_ls"], file["spin_ls"]
    # 子图
    plt.subplot(132)
    for ib in range(eng_ls.shape[1]):
        oplt.colored_line(k_sym_ls[:401], eng_ls[:401, ib], spin_ls[:401, ib], plt.gca(), cmap="RdBu", clim=(-1.3, 1.3))
    plt.plot(k_sym_ls[201:], eng_ls[201:], c="0.3")
    plt.axhline(0, ls=":", lw=1, color="tab:gray")
    plt.yticks([-0.8, -0.4, 0, 0.4, 0.8])
    plt.axvline(np.pi, color="k")
    plt.axvline(np.pi * 2, color="k")
    plt.xticks([0, np.pi, np.pi * 2, k_sym_ls[-1]], labels=[r"$\Gamma$", "X", "M", r"$\Gamma$"])
    plt.ylim([-1, 1])
    plt.ylabel(r"$E$ (eV)")
    #
    # 费米面
    # 读取
    file = sio.loadmat(fname_fs)
    kx_ls, ky_ls, dos_ls = file["kx_ls"], file["ky_ls"], file["dos_ls"]
    # 子图
    plt.subplot(133)
    plt.pcolormesh(kx_ls / np.pi, ky_ls / np.pi, dos_ls, cmap="RdBu", clim=(-16, 16))
    plt.axis("scaled")
    plt.yticks([-1, 0, 1])
    plt.xlabel(r"$k_x \ (\pi/a)$")
    plt.ylabel(r"$k_y \ (\pi/a)$")
    plt.colorbar()
    #
    oplt.number_subplots()
    plt.show()


def main_s2_folded():
    fname_ek0 = "./data/" + "E(k)_KV2Se2O-kk" + ".mat"
    fname_fs0 = "./data/" + "FmSurf_KV2Se2O-kk_Nk800x" + ".mat"
    fname_ek_w = "./data/" + "E(k)_KV2Se2O-SDW-kk" + ".mat"
    fname_dos = "./data/" + "LDoS_KV2Se2O-SDW-xy_ε5e-3" + ".mat"
    fname_dos2 = "./data/" + "LDoS_KV2Se2O-SDW-xy_Vi-0.02_ε1e-3" + ".mat"

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 4, 5 * 2 + 1), dpi=60).set_layout_engine("compressed")
    #
    # 示意图
    plt.subplot(2, 4, (1, 4))
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(" ")
    #
    # 色散
    # 读取
    file = sio.loadmat(fname_ek0)
    eng0_ls = file["eng_ls"]
    file = sio.loadmat(fname_ek_w)
    k_sym_ls, eng_ls, spin_ls = file["k_path"][:, 0], file["eng_ls"], file["spin_ls"]
    # 子图
    plt.subplot(245)
    plt.plot(k_sym_ls, eng0_ls, c="0.8", lw=4)
    for ib in range(eng_ls.shape[1]):
        oplt.colored_line(k_sym_ls[:401], eng_ls[:401, ib], spin_ls[:401, ib], plt.gca(), cmap="RdBu", clim=(-1.3, 1.3))
    plt.plot(k_sym_ls[201:], eng_ls[201:], c="0.3")
    plt.axhline(0, ls=":", lw=1, color="tab:gray")
    plt.yticks([-0.4, 0, 0.4])
    plt.axvline(np.pi, color="k")
    plt.axvline(np.pi * 2, color="k")
    plt.xticks([0, np.pi, np.pi * 2, k_sym_ls[-1]], labels=[r"$\Gamma$", "X", "M", r"$\Gamma$"])
    plt.ylim([-0.7, 0.7])
    plt.ylabel(r"$E$ (eV)")
    #
    # 读取
    file = sio.loadmat(fname_dos)
    eng_ls, dos_ls = file["eng_ls"][0], file["dos_ls"][0]
    file = sio.loadmat(fname_dos2)
    eng_ls2, dos_ls2 = file["eng_ls"][0], file["dos_ls"][0]
    # 子图
    plt.subplot(246)
    plt.plot(eng_ls, dos_ls, "k")
    # plt.plot(eng_ls2, dos_ls2, "r")
    # plt.plot(eng_ls, dos_ls2 - dos_ls, "b")
    plt.xticks([-0.04, 0, 0.04])
    plt.ylim([0, 2.8])
    plt.xlabel(r"$E$ (eV)")
    plt.ylabel(r"DoS")
    #
    # 费米面
    # 读取
    file = sio.loadmat(fname_fs0)
    kx_ls, ky_ls, dos_ls = file["kx_ls"], file["ky_ls"], file["dos_ls"]
    dos_xfold = np.concatenate((dos_ls[:, 400:], dos_ls[:, :400]), axis=1)
    dos_mfold = np.concatenate((dos_xfold[400:], dos_xfold[:400]), axis=0)
    # 子图
    plt.subplot(247)
    pcm0 = plt.pcolormesh(kx_ls / np.pi, ky_ls / np.pi, dos_ls, cmap="RdBu", clim=(-16, 16))
    plt.pcolormesh(kx_ls / np.pi, ky_ls / np.pi, dos_mfold, cmap="RdBu", clim=(-16, 16), alpha=0.3)
    plt.axis("scaled")
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])
    plt.xlabel(r"$k_x \ (\pi/a)$")
    plt.ylabel(r"$k_y \ (\pi/a)$")
    # plt.colorbar(pcm0)
    #
    # 读取
    file = sio.loadmat(fname_fs0)
    kx_ls, ky_ls, dos_ls = file["kx_ls"], file["ky_ls"], file["dos_ls"]
    dos_mfold = np.concatenate((dos_xfold[400:], dos_xfold[:400]), axis=0)
    # 子图
    plt.subplot(248)
    pcm0 = plt.pcolormesh(kx_ls / np.pi, ky_ls / np.pi, dos_ls, cmap="RdBu", clim=(-16, 16))
    plt.pcolormesh(kx_ls / np.pi, ky_ls / np.pi, dos_xfold, cmap="RdBu", clim=(-16, 16), alpha=0.3)
    plt.axis("scaled")
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])
    plt.xlabel(r"$k_x \ (\pi/a)$")
    plt.ylabel(r"$k_y \ (\pi/a)$")
    #
    oplt.number_subplots()
    plt.show()


def main_s3_ldos():
    fname_all = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi0.2,E0.07.mat"
    fname_3 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi(0,0,0.2,0),E0.07.mat"
    fname_4 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi(0,0,0,0.2),E0.07.mat"
    # mag imp
    # fname_all = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi-0.2τ0σzsz,E0.07.mat"
    # fname_3 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi(0,0,-0.2sz,0),E0.07.mat"
    # fname_4 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi(0,0,0,0.2sz),E0.07.mat"
    #
    center = np.array((100, 100))
    clim_d = (-0.1, 0.1)
    atom_r = 0.4
    cmap0 = "Spectral_r"

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 3, 5 * 2), dpi=100).set_layout_engine("compressed")

    # 读取
    file = sio.loadmat(fname_all)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    pos_imp_ = (0.25, 0.25)
    #
    # DoS_0(x1,x2)
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    x1_ls, y1_ls = [np.linspace(*(center + (-50, 50)), 401)] * 2
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)
    ldos_0_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, +1], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # cut line
    ldos_o = (ldos_diff_ls * np.tile([1, +1], 4)).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, pos_atom)
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 8, 81)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    pos_c2 = np.transpose([-x_cut, +x_cut]) + pos_imp
    pos_c3 = np.transpose([-x_cut, -x_cut]) + pos_imp
    ldos_c0_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_c1_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    ldos_c2_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c2)
    ldos_c3_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c3)
    # 子图
    plt.subplot(231)
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_0_ls), cmap=cmap0, clim=clim_d)
    plt.axis("scaled")
    plt.plot(*pos_c0.T, "k-", lw=1)
    plt.plot(*pos_c1.T, "r--", lw=1)
    # plt.plot(*pos_c2.T, c="0.7", lw=1)
    # plt.plot(*pos_c3.T, c="0.7", lw=1)
    plt.xlim(center + (-15, 15))
    plt.ylim(center + (-15, 15))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    # plt.colorbar()
    #
    # DoS_z(x),(y)
    plt.subplot(234)
    plt.axhline(0.0, ls="-.", lw=1, color="0.7")
    plt.plot(x_cut * SR2, ldos_c0_ls, "k-", label=r"$x$")
    plt.plot(x_cut * SR2, ldos_c1_ls, "r--", label=r"$y$")
    # plt.plot(x_cut * SR2, ldos_c2_ls, "-", label=r"$-x$")
    # plt.plot(x_cut * SR2, ldos_c3_ls, "--", label=r"$-y$")
    plt.xlabel(r"Distance")
    plt.ylabel(r"$\delta$ LDoS")
    plt.legend(loc="lower right")
    #
    # 读取
    file = sio.loadmat(fname_3)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    pos_imp_ = [(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 0)][2]
    #
    # DoS_0(x1,x2)
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    x1_ls, y1_ls = [np.linspace(*(center + (-50, 50)), 401)] * 2
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    # pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)
    ldos_0_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, +1], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # cut line
    ldos_o = (ldos_diff_ls * np.tile([1, +1], 4)).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, pos_atom)
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 8, 81)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    pos_c2 = np.transpose([-x_cut, +x_cut]) + pos_imp
    pos_c3 = np.transpose([-x_cut, -x_cut]) + pos_imp
    ldos_c0_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_c1_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    ldos_c2_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c2)
    ldos_c3_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c3)
    # 子图
    plt.subplot(232)
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_0_ls), cmap=cmap0, clim=clim_d)
    plt.axis("scaled")
    plt.plot(*pos_c0.T, "k-", lw=1)
    plt.plot(*pos_c1.T, "r--", lw=1)
    # plt.plot(*pos_c2.T, c="0.7", lw=1)
    # plt.plot(*pos_c3.T, c="0.7", lw=1)
    plt.xlim(center + (-15, 15))
    plt.ylim(center + (-15, 15))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    # plt.colorbar()
    #
    # DoS_z(x),(y)
    plt.subplot(235)
    plt.axhline(0.0, ls="-.", lw=1, color="0.7")
    plt.plot(x_cut * SR2, ldos_c0_ls, "k-", label=r"$x$")
    plt.plot(x_cut * SR2, ldos_c1_ls, "r--", label=r"$y$")
    # plt.plot(x_cut * SR2, ldos_c2_ls, "-", label=r"$-x$")
    # plt.plot(x_cut * SR2, ldos_c3_ls, "--", label=r"$-y$")
    plt.yticks([-0.06, -0.03, 0, 0.03])
    plt.xlabel(r"Distance")
    # plt.ylabel(r"$\delta LDoS_0$")
    plt.legend(loc="lower right")
    #
    # 读取
    file = sio.loadmat(fname_4)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    pos_imp_ = [(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 0)][3]
    #
    # DoS_0(x1,x2)
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    x1_ls, y1_ls = [np.linspace(*(center + (-50, 50)), 401)] * 2
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    # pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)
    ldos_0_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, +1], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # cut line
    ldos_o = (ldos_diff_ls * np.tile([1, +1], 4)).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, pos_atom)
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 8, 81)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    pos_c2 = np.transpose([-x_cut, +x_cut]) + pos_imp
    pos_c3 = np.transpose([-x_cut, -x_cut]) + pos_imp
    ldos_c0_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_c1_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    ldos_c2_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c2)
    ldos_c3_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c3)
    # 子图
    plt.subplot(233)
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_0_ls), cmap=cmap0, clim=clim_d)
    plt.axis("scaled")
    plt.plot(*pos_c0.T, "k-", lw=1)
    plt.plot(*pos_c1.T, "r--", lw=1)
    # plt.plot(*pos_c2.T, c="0.7", lw=1)
    # plt.plot(*pos_c3.T, c="0.7", lw=1)
    plt.xlim(center + (-15, 15))
    plt.ylim(center + (-15, 15))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.colorbar()
    #
    # DoS_z(x),(y)
    plt.subplot(236)
    plt.axhline(0.0, ls="-.", lw=1, color="0.7")
    plt.plot(x_cut * SR2, ldos_c0_ls, "k-", label=r"$x$")
    plt.plot(x_cut * SR2, ldos_c1_ls, "r--", label=r"$y$")
    # plt.plot(x_cut * SR2, ldos_c2_ls, "-", label=r"$-x$")
    # plt.plot(x_cut * SR2, ldos_c3_ls, "--", label=r"$-y$")
    plt.yticks([-0.06, -0.03, 0, 0.03])
    plt.xlabel(r"Distance")
    # plt.ylabel(r"$\delta LDoS_0$")
    plt.legend(loc="lower right")
    #
    oplt.number_subplots()
    plt.show()


def main_s4_qpi():
    fname_x = "./data/" + "QPI_x(E)_KV2Se2O-SDW-xy_SDW0.02,Imp-0.2.mat"
    fname_x = "./data/" + "QPI_x(E).mat"
    fname_q3 = "./data/" + "QPI_KV2Se2O-SDW_(t1)0.03,Vi-0.2,E0.03.mat"
    fname_q6 = "./data/" + "QPI_KV2Se2O-SDW_(t1)0.03,Vi-0.2,E0.06.mat"

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 3, 5 + 0.5), dpi=90)
    #
    # QPI
    # 读取
    file = sio.loadmat(fname_q3)
    qpi_0_ls = file["qpi_0_ls"]
    n_q = qpi_0_ls.shape[0]
    q_ls = np.linspace(-4, 4, n_q)
    # 子图
    plt.subplot(131)
    plt.pcolormesh(q_ls, q_ls, np.abs(qpi_0_ls), cmap="viridis", clim=(0, 8))
    plt.axis("scaled")
    plt.xticks([-2, 0, 2])
    plt.yticks([-2, 0, 2])
    plt.tick_params(which="both", color="0.7")
    plt.xlabel(r"$q_1 \ (\pi/a')$")
    plt.ylabel(r"$q_2 \ (\pi/a')$")
    # plt.colorbar()
    #
    # 读取
    file = sio.loadmat(fname_q6)
    qpi_0_ls = file["qpi_0_ls"]
    # 子图
    plt.subplot(132)
    plt.pcolormesh(q_ls, q_ls, np.abs(qpi_0_ls), cmap="viridis", clim=(0, 8))
    plt.axis("scaled")
    plt.xticks([-2, 0, 2])
    plt.yticks([-2, 0, 2])
    plt.tick_params(which="both", color="0.7")
    plt.xlabel(r"$q_1 \ (\pi/a')$")
    plt.ylabel(r"$q_2 \ (\pi/a')$")
    # plt.colorbar()
    #
    # 读取
    file = sio.loadmat(fname_x)
    eng_ls = file["eng_ls"][0]
    qpi_x_ls = file["qpi_x_ls"]
    # 子图
    plt.subplot(133)
    plt.pcolormesh(np.linspace(0, 4, qpi_x_ls.shape[1]), eng_ls, np.abs(qpi_x_ls), clim=(0, 8), cmap="cividis")
    plt.xlim([0.5, 1.5])
    plt.xticks([0.6, 1.0, 1.4])
    plt.tick_params(which="both", color="0.7")
    plt.xlabel(r"$q_x\ (\pi/a)$")
    plt.ylabel(r"$E$ (eV)")
    plt.colorbar()
    #
    oplt.number_subplots()
    plt.show()


def main_s5_d90():
    fname_p4 = "./data/QPI_d90_KV2Se2O-SDW_(t1)0.03,Vi0.2,E0.04.mat"
    fname_m4 = "./data/QPI_d90_KV2Se2O-SDW_(t1)0.03,Vi0.2,E-0.04.mat"
    fname_p4s = "./data/QPI_d90_KV2Se2O-SDW_(t1)0.03,Vi0.2,E0.04_↑1,↓0.5.mat"
    fname_m4s = "./data/QPI_d90_KV2Se2O-SDW_(t1)0.03,Vi0.2,E-0.04_↑0.5,↓1.mat"
    # fname_p4 = "./data/QPI_d90_KV2Se2O-SDW_(t1)0.03,Vi(0.2,0,0,0),E0.04.mat"
    # fname_m4 = "./data/QPI_d90_KV2Se2O-SDW_(t1)0.03,Vi(0.2,0,0,0),E-0.04.mat"
    # fname_p4s = "./data/QPI_d90_KV2Se2O-SDW_(t1)0.03,Vi(0.2,0,0,0),E0.04_↑1,↓0.5.mat"
    # fname_m4s = "./data/QPI_d90_KV2Se2O-SDW_(t1)0.03,Vi(0.2,0,0,0),E-0.04_↑0.5,↓1.mat"

    xtick0 = [-4, -2, 0, 2, 4]
    range1d_ = np.asarray([0.2, 1.8])

    def plot_arrow():
        plt.arrow(
            range1d_[0],
            range1d_[0],
            range1d_[1] - range1d_[0],
            range1d_[1] - range1d_[0],
            color="r",
            head_width=0.2,
            head_length=0.2,
        )
        plt.text(range1d_[1] + 0.2, range1d_[1] + 0.0, r"$q_x$", c="r")
        plt.arrow(
            -range1d_[0],
            range1d_[0],
            -range1d_[1] + range1d_[0],
            range1d_[1] - range1d_[0],
            color="k",
            head_width=0.2,
            head_length=0.2,
        )
        plt.text(-range1d_[1] - 0.0, range1d_[1] + 0.2, r"$q_y$")

    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 4, 5 * 2 + 1), dpi=90).set_layout_engine("compressed", wspace=0.01)
    #
    # 读取
    file = sio.loadmat(fname_p4)
    qpi_ls_p4 = file["qpi_d90"]
    n_q = qpi_ls_p4.shape[0]
    q_ls = np.linspace(-4, 4, n_q)
    # 子图
    plt.subplot(241)
    plt.pcolormesh(q_ls, q_ls, qpi_ls_p4, cmap="RdBu_r", clim=(-2, 2))
    plt.axis("scaled")
    plot_arrow()
    plt.xticks(xtick0)
    plt.tick_params(labelbottom=False, labelleft=False)
    # plt.xlabel(r"$q_1/(\pi/\sqrt{2})$")
    # plt.ylabel(r"$q_2/(\pi/\sqrt{2})$")
    plt.text(-3.6, 3.2, r"$+40$ meV")
    plt.colorbar(location="left", shrink=0.7, aspect=14, pad=-0.1)
    #
    # 读取
    file = sio.loadmat(fname_m4)
    qpi_ls_m4 = file["qpi_d90"]
    # 子图
    plt.subplot(245)
    plt.pcolormesh(q_ls, q_ls, qpi_ls_m4, cmap="RdBu_r", clim=(-2, 2))
    plt.axis("scaled")
    plot_arrow()
    plt.xticks(xtick0)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.text(-3.6, 3.2, r"$-40$ meV")
    plt.colorbar(location="left", shrink=0.7, aspect=14, pad=-0.1)
    #
    # 处理
    range1d = np.arange(int(n_q / 8 * range1d_[0]), int(n_q / 8 * range1d_[1]))
    qpi_x = [qpi_ls_p4[n_q // 2 + i, n_q // 2 + i] for i in range1d]
    qpi_y = [qpi_ls_p4[n_q // 2 - i, n_q // 2 + i] for i in range1d]
    ylim0 = [-0.08, 0.08]
    # ylim0 = [-8, 8]
    # 子图
    plt.subplot(242)
    plt.plot(q_ls[n_q // 2 + range1d], qpi_x, c="r", label=r"$q_x$")
    plt.plot(q_ls[n_q // 2 + range1d], qpi_y, c="k", label=r"$q_y$")
    plt.axvspan(0.8, 1.2, color="0.9")
    plt.ylim(ylim0)
    plt.ylabel("Intensity", labelpad=-4)
    plt.xlabel(r"$q \ (\pi/a)$")
    plt.legend()
    #
    # 处理
    qpi_x = [qpi_ls_m4[n_q // 2 + i, n_q // 2 + i] for i in range1d]
    qpi_y = [qpi_ls_m4[n_q // 2 - i, n_q // 2 + i] for i in range1d]
    # 子图
    plt.subplot(246)
    plt.plot(q_ls[n_q // 2 + range1d], qpi_x, c="r", label=r"$q_x$")
    plt.plot(q_ls[n_q // 2 + range1d], qpi_y, c="k", label=r"$q_y$")
    plt.axvspan(0.8, 1.2, color="0.9")
    plt.ylim(ylim0)
    plt.ylabel("Intensity", labelpad=-4)
    plt.xlabel(r"$q \ (\pi/a)$")
    plt.legend()
    #
    # 读取
    file = sio.loadmat(fname_p4s)
    qpi_ls_p4s = file["qpi_d90"]
    # 子图
    plt.subplot(243)
    plt.pcolormesh(q_ls, q_ls, qpi_ls_p4s, cmap="RdBu_r", clim=(-8, 8))
    plt.axis("scaled")
    plot_arrow()
    plt.xticks(xtick0)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.text(-3.6, 3.2, r"$+40$ meV")
    plt.colorbar(location="left", shrink=0.7, aspect=14, pad=-0.1)
    #
    # 读取
    file = sio.loadmat(fname_m4s)
    qpi_ls_m4s = file["qpi_d90"]
    # 子图
    plt.subplot(247)
    plt.pcolormesh(q_ls, q_ls, qpi_ls_m4s, cmap="RdBu_r", clim=(-8, 8))
    plt.axis("scaled")
    plot_arrow()
    plt.xticks(xtick0)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.text(-3.6, 3.2, r"$-40$ meV")
    plt.colorbar(location="left", shrink=0.7, aspect=14, pad=-0.1)
    #
    # 处理
    qpi_x = [qpi_ls_p4s[n_q // 2 + i, n_q // 2 + i] for i in range1d]
    qpi_y = [qpi_ls_p4s[n_q // 2 - i, n_q // 2 + i] for i in range1d]
    ylim0 = [-8, 8]
    # 子图
    plt.subplot(244)
    plt.plot(q_ls[n_q // 2 + range1d], qpi_x, c="r", label=r"$q_x$")
    plt.plot(q_ls[n_q // 2 + range1d], qpi_y, c="k", label=r"$q_y$")
    plt.axvspan(0.8, 1.2, color="0.9")
    plt.ylim(ylim0)
    plt.ylabel("Intensity", labelpad=-4)
    plt.xlabel(r"$q \ (\pi/a)$")
    plt.legend()
    #
    # 处理
    qpi_x = [qpi_ls_m4s[n_q // 2 + i, n_q // 2 + i] for i in range1d]
    qpi_y = [qpi_ls_m4s[n_q // 2 - i, n_q // 2 + i] for i in range1d]
    # 子图
    plt.subplot(248)
    plt.plot(q_ls[n_q // 2 + range1d], qpi_x, c="r", label=r"$q_x$")
    plt.plot(q_ls[n_q // 2 + range1d], qpi_y, c="k", label=r"$q_y$")
    plt.axvspan(0.8, 1.2, color="0.9")
    plt.ylim(ylim0)
    plt.ylabel("Intensity", labelpad=-4)
    plt.xlabel(r"$q \ (\pi/a)$")
    plt.legend()
    #
    oplt.number_subplots(offset=(-1.7, -0.5), in_layout=np.arange(8))
    plt.show()


def main_s5_dm110():
    # fname_p4 = "./data/QPI_dm110_KV2Se2O-SDW_(t1)0.03,Vi0.2,E0.04.mat"
    # fname_m4 = "./data/QPI_dm110_KV2Se2O-SDW_(t1)0.03,Vi0.2,E-0.04.mat"
    # fname_p4s = "./data/QPI_dm110_KV2Se2O-SDW_(t1)0.03,Vi0.2,E0.04_↑1,↓0.5.mat"
    # fname_m4s = "./data/QPI_dm110_KV2Se2O-SDW_(t1)0.03,Vi0.2,E-0.04_↑0.5,↓1.mat"
    fname_p4 = "./data/QPI_dm110_KV2Se2O-SDW_(t1)0.03,Vi(0.2,0,0,0),E0.04.mat"
    fname_m4 = "./data/QPI_dm110_KV2Se2O-SDW_(t1)0.03,Vi(0.2,0,0,0),E-0.04.mat"
    fname_p4s = "./data/QPI_dm110_KV2Se2O-SDW_(t1)0.03,Vi(0.2,0,0,0),E0.04_↑1,↓0.5.mat"
    fname_m4s = "./data/QPI_dm110_KV2Se2O-SDW_(t1)0.03,Vi(0.2,0,0,0),E-0.04_↑0.5,↓1.mat"

    xtick0 = [-4, -2, 0, 2, 4]
    range1d_ = np.asarray([0.2, 1.8])

    def plot_arrow():
        plt.arrow(
            range1d_[0],
            range1d_[0],
            range1d_[1] - range1d_[0],
            range1d_[1] - range1d_[0],
            color="r",
            head_width=0.2,
            head_length=0.2,
        )
        plt.text(range1d_[1] + 0.2, range1d_[1] + 0.0, r"$q_x$", c="r")
        plt.arrow(
            -range1d_[0],
            range1d_[0],
            -range1d_[1] + range1d_[0],
            range1d_[1] - range1d_[0],
            color="k",
            head_width=0.2,
            head_length=0.2,
        )
        plt.text(-range1d_[1] - 0.0, range1d_[1] + 0.2, r"$q_y$")

    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 4, 5 * 2 + 1), dpi=90).set_layout_engine("compressed", wspace=0.01)
    #
    # 读取
    file = sio.loadmat(fname_p4)
    qpi_ls_p4 = file["qpi_dm110"]
    n_q = qpi_ls_p4.shape[0]
    q_ls = np.linspace(-4, 4, n_q)
    # 子图
    plt.subplot(241)
    plt.pcolormesh(q_ls, q_ls, qpi_ls_p4, cmap="RdBu_r", clim=(-8, 8))
    plt.axis("scaled")
    plot_arrow()
    plt.xticks(xtick0)
    plt.tick_params(labelbottom=False, labelleft=False)
    # plt.xlabel(r"$q_1/(\pi/\sqrt{2})$")
    # plt.ylabel(r"$q_2/(\pi/\sqrt{2})$")
    plt.text(-3.6, 3.2, r"$+40$ meV")
    plt.colorbar(location="left", shrink=0.7, aspect=14, pad=-0.1)
    #
    # 读取
    file = sio.loadmat(fname_m4)
    qpi_ls_m4 = file["qpi_dm110"]
    # 子图
    plt.subplot(245)
    plt.pcolormesh(q_ls, q_ls, qpi_ls_m4, cmap="RdBu_r", clim=(-8, 8))
    plt.axis("scaled")
    plot_arrow()
    plt.xticks(xtick0)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.text(-3.6, 3.2, r"$-40$ meV")
    plt.colorbar(location="left", shrink=0.7, aspect=14, pad=-0.1)
    #
    # 处理
    range1d = np.arange(int(n_q / 8 * range1d_[0]), int(n_q / 8 * range1d_[1]))
    qpi_x = [qpi_ls_p4[n_q // 2 + i, n_q // 2 + i] for i in range1d]
    qpi_y = [qpi_ls_p4[n_q // 2 - i, n_q // 2 + i] for i in range1d]
    ylim0 = [-8, 8]
    # ylim0 = [-8, 8]
    # 子图
    plt.subplot(242)
    plt.plot(q_ls[n_q // 2 + range1d], qpi_x, c="r", label=r"$q_x$")
    plt.plot(q_ls[n_q // 2 + range1d], qpi_y, c="k", label=r"$q_y$")
    plt.axvspan(0.8, 1.2, color="0.9")
    plt.ylim(ylim0)
    plt.ylabel("Intensity", labelpad=-4)
    plt.xlabel(r"$q \ (\pi/a)$")
    plt.legend()
    #
    # 处理
    qpi_x = [qpi_ls_m4[n_q // 2 + i, n_q // 2 + i] for i in range1d]
    qpi_y = [qpi_ls_m4[n_q // 2 - i, n_q // 2 + i] for i in range1d]
    # 子图
    plt.subplot(246)
    plt.plot(q_ls[n_q // 2 + range1d], qpi_x, c="r", label=r"$q_x$")
    plt.plot(q_ls[n_q // 2 + range1d], qpi_y, c="k", label=r"$q_y$")
    plt.axvspan(0.8, 1.2, color="0.9")
    plt.ylim(ylim0)
    plt.ylabel("Intensity", labelpad=-4)
    plt.xlabel(r"$q \ (\pi/a)$")
    plt.legend()
    #
    # 读取
    file = sio.loadmat(fname_p4s)
    qpi_ls_p4s = file["qpi_dm110"]
    # 子图
    plt.subplot(243)
    plt.pcolormesh(q_ls, q_ls, qpi_ls_p4s, cmap="RdBu_r", clim=(-8, 8))
    plt.axis("scaled")
    plot_arrow()
    plt.xticks(xtick0)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.text(-3.6, 3.2, r"$+40$ meV")
    plt.colorbar(location="left", shrink=0.7, aspect=14, pad=-0.1)
    #
    # 读取
    file = sio.loadmat(fname_m4s)
    qpi_ls_m4s = file["qpi_dm110"]
    # 子图
    plt.subplot(247)
    plt.pcolormesh(q_ls, q_ls, qpi_ls_m4s, cmap="RdBu_r", clim=(-8, 8))
    plt.axis("scaled")
    plot_arrow()
    plt.xticks(xtick0)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.text(-3.6, 3.2, r"$-40$ meV")
    plt.colorbar(location="left", shrink=0.7, aspect=14, pad=-0.1)
    #
    # 处理
    qpi_x = [qpi_ls_p4s[n_q // 2 + i, n_q // 2 + i] for i in range1d]
    qpi_y = [qpi_ls_p4s[n_q // 2 - i, n_q // 2 + i] for i in range1d]
    ylim0 = [-8, 8]
    # 子图
    plt.subplot(244)
    plt.plot(q_ls[n_q // 2 + range1d], qpi_x, c="r", label=r"$q_x$")
    plt.plot(q_ls[n_q // 2 + range1d], qpi_y, c="k", label=r"$q_y$")
    plt.axvspan(0.8, 1.2, color="0.9")
    plt.ylim(ylim0)
    plt.ylabel("Intensity", labelpad=-4)
    plt.xlabel(r"$q \ (\pi/a)$")
    plt.legend()
    #
    # 处理
    qpi_x = [qpi_ls_m4s[n_q // 2 + i, n_q // 2 + i] for i in range1d]
    qpi_y = [qpi_ls_m4s[n_q // 2 - i, n_q // 2 + i] for i in range1d]
    # 子图
    plt.subplot(248)
    plt.plot(q_ls[n_q // 2 + range1d], qpi_x, c="r", label=r"$q_x$")
    plt.plot(q_ls[n_q // 2 + range1d], qpi_y, c="k", label=r"$q_y$")
    plt.axvspan(0.8, 1.2, color="0.9")
    plt.ylim(ylim0)
    plt.ylabel("Intensity", labelpad=-4)
    plt.xlabel(r"$q \ (\pi/a)$")
    plt.legend()
    #
    oplt.number_subplots(offset=(-1.7, -0.5), in_layout=np.arange(8))
    plt.show()


def main_r1_topography():
    fname_3 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi(0,0,2,0),EΣ0.3.mat"
    fname_4 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi(0,0,0,2),EΣ0.3.mat"
    center = np.array((100, 100))
    clim_d = (0.5, 2)  # (-0.1, 0.1)
    atom_r = 0.4
    atom_r_k = 0.4
    cmap0 = "viridis"  # "Spectral_r" #"viridis"

    dos0_v = 2
    weight_v = 6
    pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 3-3, 5 * 2), dpi=90).set_layout_engine("compressed")

    # _
    plt.subplot(2, 3, (1, 4))
    plt.xticks([])
    plt.yticks([])
    #
    # 读取
    file = sio.loadmat(fname_3)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    # DoS_V
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    x1_ls, y1_ls = [np.linspace(*(center + (-50, 50)), 801)] * 2
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    ldos_0_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, +1], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    ln_ldos_ls = -np.log1p(ldos_0_ls / dos0_v)
    # 子图
    plt.subplot(232)
    plt.pcolormesh(x1_ls, y1_ls, ln_ldos_ls, cmap=cmap0, clim=None)
    plt.axis("scaled")
    plt.xlim(center + (-9, 9))
    plt.ylim(center + (-9, 9))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(" ")
    # plt.colorbar()
    # signal_K
    gd_k = lambda x: gaussian_delta(x, sigma=atom_r_k)
    alt_k_ls = np.ones((*(ldos_diff_ls.shape[:2]), 1))
    pos_atom_k = np.asarray([[-0.25, 0.25]])
    alt_k_ls = qpi.transform_ldos_orbit2position_in_Cartesian(alt_k_ls, gd_k, pos_atom_k, x0_ls, y0_ls, x1_ls, y1_ls)
    alt_tot = alt_k_ls + weight_v * alt_k_ls * ln_ldos_ls
    # 子图
    plt.subplot(233)
    plt.pcolormesh(x1_ls, y1_ls, alt_tot, cmap=cmap0, clim=clim_d)
    plt.axis("scaled")
    plt.xlim(center + (-9, 9))
    plt.ylim(center + (-9, 9))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(" ")
    # plt.colorbar()
    #
    # 读取
    file = sio.loadmat(fname_4)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    # DoS_V
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    x1_ls, y1_ls = [np.linspace(*(center + (-50, 50)), 801)] * 2
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    ldos_0_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, +1], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    ln_ldos_ls = -np.log1p(ldos_0_ls / dos0_v)
    # 子图
    plt.subplot(235)
    plt.pcolormesh(x1_ls, y1_ls, ln_ldos_ls, cmap=cmap0, clim=None)
    plt.axis("scaled")
    plt.xlim(center + (-9, 9))
    plt.ylim(center + (-9, 9))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(" ")
    # plt.colorbar()
    # signal_K
    gd_k = lambda x: gaussian_delta(x, sigma=atom_r_k)
    alt_k_ls = np.ones((*(ldos_diff_ls.shape[:2]), 1))
    pos_atom_k = np.asarray([[-0.25, 0.25]])
    alt_k_ls = qpi.transform_ldos_orbit2position_in_Cartesian(alt_k_ls, gd_k, pos_atom_k, x0_ls, y0_ls, x1_ls, y1_ls)
    alt_tot = alt_k_ls + weight_v * alt_k_ls * ln_ldos_ls
    # 子图
    plt.subplot(236)
    plt.pcolormesh(x1_ls, y1_ls, alt_tot, cmap=cmap0, clim=clim_d)
    plt.axis("scaled")
    plt.xlim(center + (-9, 9))
    plt.ylim(center + (-9, 9))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(" ")
    # plt.colorbar()
    #
    oplt.number_subplots()
    plt.show()


def main_r2_diff_2sdw():
    fname_fs0 = "./data/FmSurf_τzσz_μ0.07.mat"
    fname_fs9 = "./data/FmSurf_τzσ0_μ0.07.mat"
    fname_ld0 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi0.2,E0.07.mat"
    fname_ld9 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,SDWσz0,Vi0.2,E0.07.mat"

    clim_fs = (-8, 8)
    cmap_fs = "RdBu"
    #
    center = np.array((100, 100))
    clim_d = (-0.1, 0.1)
    cmap_d = "Spectral_r"
    atom_r = 0.4
    #
    clim_q = (0, 8)

    # 图
    oplt.set_rc_origin()
    plt.figure(figsize=(6*3, 5*3), dpi=60).set_layout_engine("compressed")
    #
    plt.subplot(331)
    # 读取
    file = sio.loadmat(fname_fs0)
    kx_ls0, ky_ls0, dos_ls0 = file["kx_ls"], file["ky_ls"], file["dos_ls"]
    # 作图
    plt.pcolormesh(kx_ls0 / np.pi, ky_ls0 / np.pi, dos_ls0, cmap=cmap_fs, clim=clim_fs)
    plt.axis("scaled")
    plt.yticks([-1, 0, 1])
    plt.xlabel(r"$k_x/\pi$")
    plt.ylabel(r"$k_y/\pi$")
    # plt.colorbar()
    #
    plt.subplot(332)
    # 读取
    file = sio.loadmat(fname_fs9)
    kx_ls9, ky_ls9, dos_ls9 = file["kx_ls"], file["ky_ls"], file["dos_ls"]
    # 作图
    plt.pcolormesh(kx_ls9 / np.pi, ky_ls9 / np.pi, dos_ls9, cmap=cmap_fs, clim=clim_fs)
    plt.axis("scaled")
    plt.yticks([-1, 0, 1])
    plt.xlabel(r"$k_x/\pi$")
    plt.ylabel(r"$k_y/\pi$")
    plt.colorbar()
    #
    plt.subplot(333)
    # 作图
    plt.pcolormesh(kx_ls0 / np.pi, ky_ls0 / np.pi, dos_ls0 - dos_ls9, cmap=cmap_fs)
    plt.axis("scaled")
    plt.yticks([-1, 0, 1])
    plt.xlabel(r"$k_x/\pi$")
    plt.ylabel(r"$k_y/\pi$")
    plt.colorbar()
    #
    plt.subplot(334)
    # 读取
    file = sio.loadmat(fname_ld0)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    # DoS_0(x1,x2)
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    x1_ls, y1_ls = [np.linspace(*(center + (-50, 50)), 401)] * 2
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)
    ldos_0_ls0 = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, +1], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # 作图
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_0_ls0), cmap=cmap_d, clim=clim_d)
    plt.axis("scaled")
    # plt.plot(*pos_c0.T, "k-", lw=1)
    # plt.plot(*pos_c1.T, "r--", lw=1)
    plt.xlim(center + (-15, 15))
    plt.ylim(center + (-15, 15))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #
    plt.subplot(335)
    # 读取
    file = sio.loadmat(fname_ld9)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    # DoS_0(x1,x2)
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    x1_ls, y1_ls = [np.linspace(*(center + (-50, 50)), 401)] * 2
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)
    ldos_0_ls9 = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, +1], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # 作图
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_0_ls9), cmap=cmap_d, clim=clim_d)
    plt.axis("scaled")
    # plt.plot(*pos_c0.T, "k-", lw=1)
    # plt.plot(*pos_c1.T, "r--", lw=1)
    plt.xlim(center + (-15, 15))
    plt.ylim(center + (-15, 15))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.colorbar()
    #
    plt.subplot(336)
    # 作图
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_0_ls0-ldos_0_ls9), cmap=cmap_d, clim=clim_d)
    plt.axis("scaled")
    # plt.plot(*pos_c0.T, "k-", lw=1)
    # plt.plot(*pos_c1.T, "r--", lw=1)
    plt.xlim(center + (-15, 15))
    plt.ylim(center + (-15, 15))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.colorbar()
    #
    plt.subplot(337)
    # QPI
    qpi_0_ls0 = qpi.calculate_qpi_from_ldos_2d(ldos_0_ls0)
    n_q = qpi_0_ls0.shape[0]
    q_ls = np.linspace(-4, 4, n_q)
    # 作图
    plt.pcolormesh(q_ls, q_ls,np.abs(qpi_0_ls0), cmap="cividis", clim=clim_q)
    plt.axis("scaled")
    plt.xticks([-2, 0, 2])
    plt.yticks([-2, 0, 2])
    plt.tick_params(which="both", color="0.7")
    plt.xlabel(r"$q_1 \ (\pi/a')$")
    plt.ylabel(r"$q_2 \ (\pi/a')$")
    # plt.colorbar()
    #
    plt.subplot(338)
    # QPI
    qpi_0_ls9 = qpi.calculate_qpi_from_ldos_2d(ldos_0_ls9)
    # 作图
    plt.pcolormesh(q_ls, q_ls, np.abs(qpi_0_ls9), cmap="cividis", clim=clim_q)
    plt.axis("scaled")
    plt.xticks([-2, 0, 2])
    plt.yticks([-2, 0, 2])
    plt.tick_params(which="both", color="0.7")
    plt.xlabel(r"$q_1 \ (\pi/a')$")
    plt.ylabel(r"$q_2 \ (\pi/a')$")
    plt.colorbar()
    #
    plt.subplot(339)
    # QPI
    qpi_0_ls = np.abs(qpi_0_ls0) - np.abs(qpi_0_ls9)
    # qpi_0_ls = qpi.calculate_qpi_from_ldos_2d(ldos_0_ls0-ldos_0_ls9)
    # 作图
    plt.pcolormesh(q_ls, q_ls, np.abs(qpi_0_ls), cmap="cividis", clim=None)
    plt.axis("scaled")
    plt.xticks([-2, 0, 2])
    plt.yticks([-2, 0, 2])
    plt.tick_params(which="both", color="0.7")
    plt.xlabel(r"$q_1 \ (\pi/a')$")
    plt.ylabel(r"$q_2 \ (\pi/a')$")
    plt.colorbar()
    #
    oplt.number_subplots()
    plt.show()

def main_r2_2_2mpoint():
    fname_0 = "./data/E(k)_M.mat"
    fname_9 = "./data/E(k)_M'.mat"

    # 图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 2, 5 * 2), dpi=90)
    #
    plt.subplot(221)
    # 读取
    file = sio.loadmat(fname_0)
    k_sym_ls0, eng_ls0, spin_ls0 = file["k_path"][:, 0], file["eng_ls"], file["spin_ls"]
    k_sym_ls0 = k_sym_ls0[800:]
    eng_ls0 = eng_ls0[800:]
    # 作图
    plt.plot(k_sym_ls0[::-1], eng_ls0[::-1, :4], c="k")
    plt.axhline(0, ls=":", lw=1, color="tab:gray")
    plt.xticks([k_sym_ls0[-1], np.pi * 2], labels=["M", r"$\Gamma$"])
    plt.ylabel(r"$E$ (eV)")
    plt.xlabel(" ")
    #
    plt.subplot(222)
    # 读取
    file = sio.loadmat(fname_9)
    k_sym_ls9, eng_ls9, spin_ls9 = file["k_path"][:, 0], file["eng_ls"], file["spin_ls"]
    k_sym_ls9 = k_sym_ls9[800:]
    eng_ls9 = eng_ls9[800:]
    # 作图
    plt.plot(k_sym_ls9[::-1], eng_ls9[::-1, :4], c="k")
    plt.axhline(0, ls=":", lw=1, color="tab:gray")
    plt.xticks([k_sym_ls9[-1], np.pi * 2], labels=["M'", r"$\Gamma$"])
    plt.ylabel(r"$E$ (eV)")
    plt.xlabel(" ")
    #
    plt.subplot(223)
    # 处理
    eng_diff0 = eng_ls0[::-1, 2] - eng_ls0[::-1, 1]
    eng_diff9 = eng_ls9[::-1, 2] - eng_ls9[::-1, 1]
    # 作图
    plt.plot(k_sym_ls0[::-1]/np.pi-2, eng_diff0, label=r"$\Gamma$-M")
    plt.plot(k_sym_ls9[::-1]/np.pi-2, eng_diff9, "--", label=r"$\Gamma$-M'")
    plt.ylabel("gap (eV)")
    plt.xlabel(r"$|k|$")
    plt.legend()
    #
    plt.subplot(224)
    # 作图
    plt.plot(k_sym_ls0[::-1]/np.pi-2, eng_diff0 - eng_diff9, "k")
    plt.ylabel("gap difference (eV)")
    plt.xlabel(r"$|k|$")
    #
    oplt.number_subplots(offset=(-1.4, +0.5))
    plt.show()


def main_r3_so():
    fname_fs0 = "./data/FmSurf_KV2Se2O-SDW-kk_E0.07.mat"
    fname_ld0 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi0.2,E0.07.mat"
    fname_fs1 = "./data/FmSurf_KV2Se2O-SDW-kk_so0.1,E0.07.mat"
    fname_ld1 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,so0.1,Vi0.2,E0.07.mat"

    # 图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 4, 5 * 2+0.5), dpi=60).set_layout_engine("compressed")
    #
    plt.subplot(241)
    # 读取
    file = sio.loadmat(fname_fs0)
    kx_ls0, ky_ls0, dos_ls0 = file["kx_ls"], file["ky_ls"], file["dos_ls"]
    # 作图
    plt.pcolormesh(kx_ls0 / np.pi, ky_ls0 / np.pi, dos_ls0, cmap="RdBu", clim=(-8, 8))
    plt.axis("scaled")
    plt.yticks([-1, 0, 1])
    plt.xlabel(r"$k_x/\pi$")
    plt.ylabel(r"$k_y/\pi$")
    plt.colorbar()
    #
    plt.subplot(242)
    # 读取
    file = sio.loadmat(fname_ld0)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    # DoS_0(x1,x2)
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    center = np.array((100, 100))
    x1_ls, y1_ls = [np.linspace(*(center + (-50, 50)), 400)] * 2
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)
    # pos_atom = np.repeat([[0, 0], [0.5, 0], [0.5, -0.5], [0, -0.5]], 2, axis=0)
    ldos_0_ls0 = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, +1], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # 作图
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_0_ls0), cmap="Spectral_r", clim=(-0.1, 0.1))
    plt.axis("scaled")
    # plt.plot(*pos_c0.T, "k-", lw=1)
    # plt.plot(*pos_c1.T, "r--", lw=1)
    plt.xlim(center + (-15, 15))
    plt.ylim(center + (-15, 15))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.colorbar()
    #
    plt.subplot(243)
    # QPI
    qpi_0_ls0 = qpi.calculate_qpi_from_ldos_2d(ldos_0_ls0)
    n_q = qpi_0_ls0.shape[0]
    q_ls = np.linspace(-4, 4, n_q)
    # 作图
    plt.pcolormesh(q_ls, q_ls, np.abs(qpi_0_ls0), cmap="cividis", clim=(0, 8))
    plt.axis("scaled")
    plt.xticks([-2, 0, 2])
    plt.yticks([-2, 0, 2])
    plt.tick_params(which="both", color="0.7")
    plt.xlabel(r"$q_1 \ (\pi/a')$")
    plt.ylabel(r"$q_2 \ (\pi/a')$")
    plt.colorbar()
    #
    plt.subplot(244)
    # R90
    qpi_0_ls0 = qpi_0_ls0[1:, 1:]
    qpi_9_ls0 = rotate90_mtx(qpi_0_ls0)
    # 作图
    plt.pcolormesh(np.abs(qpi_0_ls0) - np.abs(qpi_9_ls0), cmap="RdBu_r", clim=(-4, 4))
    plt.axis("scaled")
    plt.xticks([])
    plt.yticks([])
    plt.ylabel(" ")
    plt.colorbar()
    #
    plt.subplot(245)
    # 读取
    file = sio.loadmat(fname_fs1)
    kx_ls0, ky_ls0, dos_ls0 = file["kx_ls"], file["ky_ls"], file["dos_ls"]
    # 作图
    plt.pcolormesh(kx_ls0 / np.pi, ky_ls0 / np.pi, dos_ls0, cmap="RdBu", clim=(-8, 8))
    plt.axis("scaled")
    plt.yticks([-1, 0, 1])
    plt.xlabel(r"$k_x/\pi$")
    plt.ylabel(r"$k_y/\pi$")
    plt.colorbar()
    #
    plt.subplot(246)
    # 读取
    file = sio.loadmat(fname_ld1)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    # DoS_0(x1,x2)
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    center = np.array((100, 100))
    x1_ls, y1_ls = [np.linspace(*(center + (-50, 50)), 400)] * 2
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)
    # pos_atom = np.repeat([[0, 0], [0.5, 0], [0.5, -0.5], [0, -0.5]], 2, axis=0)
    ldos_0_ls0 = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, +1], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # 作图
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_0_ls0), cmap="Spectral_r", clim=(-0.1, 0.1))
    plt.axis("scaled")
    # plt.plot(*pos_c0.T, "k-", lw=1)
    # plt.plot(*pos_c1.T, "r--", lw=1)
    plt.xlim(center + (-15, 15))
    plt.ylim(center + (-15, 15))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.colorbar()
    #
    plt.subplot(247)
    # QPI
    qpi_0_ls0 = qpi.calculate_qpi_from_ldos_2d(ldos_0_ls0)
    n_q = qpi_0_ls0.shape[0]
    q_ls = np.linspace(-4, 4, n_q)
    # 作图
    plt.pcolormesh(q_ls, q_ls, np.abs(qpi_0_ls0), cmap="cividis", clim=(0, 8))
    plt.axis("scaled")
    plt.xticks([-2, 0, 2])
    plt.yticks([-2, 0, 2])
    plt.tick_params(which="both", color="0.7")
    plt.xlabel(r"$q_1 \ (\pi/a')$")
    plt.ylabel(r"$q_2 \ (\pi/a')$")
    plt.colorbar()
    #
    plt.subplot(248)
    # R90
    qpi_0_ls0 = qpi_0_ls0[1:, 1:]
    qpi_9_ls0 = rotate90_mtx(qpi_0_ls0)
    # 作图
    plt.pcolormesh(np.abs(qpi_0_ls0) - np.abs(qpi_9_ls0), cmap="RdBu_r", clim=(-4, 4))
    plt.axis("scaled")
    plt.xticks([])
    plt.yticks([])
    plt.ylabel(" ")
    plt.colorbar()
    #
    oplt.number_subplots()
    plt.show()


def main_r4_0sdw():
    fname_s2 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi0.2,E0.07.mat"
    # 0 SDW
    fname_s0 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,SDW0,Vi0.2,E0.07.mat"
    #
    center = np.array((100, 100))
    clim_d = (-0.1, 0.1)
    atom_r = 0.4
    cmap0 = "Spectral_r"

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 3, 5 * 2+0.8), dpi=100).set_layout_engine("compressed")
    #
    plt.subplot(231)
    # 读取
    file = sio.loadmat(fname_s2)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    pos_imp_ = (0.25, 0.25)
    # DoS_0(x1,x2)
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    x1_ls, y1_ls = [np.linspace(*(center + (-50, 50)), 401)] * 2
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)
    ldos_0_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, +1], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # cut line
    ldos_o = (ldos_diff_ls * np.tile([1, +1], 4)).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, pos_atom)
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 8, 81)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    pos_c2 = np.transpose([-x_cut, +x_cut]) + pos_imp
    pos_c3 = np.transpose([-x_cut, -x_cut]) + pos_imp
    ldos_c0_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_c1_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    ldos_c2_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c2)
    ldos_c3_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c3)
    # 子图
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_0_ls), cmap=cmap0, clim=clim_d)
    plt.axis("scaled")
    plt.plot(*pos_c0.T, "k-", lw=1)
    plt.plot(*pos_c1.T, "r--", lw=1)
    # plt.plot(*pos_c2.T, c="0.7", lw=1)
    # plt.plot(*pos_c3.T, c="0.7", lw=1)
    plt.xlim(center + (-15, 15))
    plt.ylim(center + (-15, 15))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    # plt.colorbar()
    #
    # DoS_z(x),(y)
    plt.subplot(232)
    # 子图
    plt.axhline(0.0, ls="-.", lw=1, color="0.7")
    plt.plot(x_cut * SR2, ldos_c0_ls, "k-", label=r"$x$")
    plt.plot(x_cut * SR2, ldos_c1_ls, "r--", label=r"$y$")
    # plt.plot(x_cut * SR2, ldos_c2_ls, "-", label=r"$-x$")
    # plt.plot(x_cut * SR2, ldos_c3_ls, "--", label=r"$-y$")
    plt.xlabel(r"Distance")
    plt.ylabel(r"$\delta$ LDoS")
    plt.legend(loc="lower right")
    #
    # QPI
    plt.subplot(233)
    # fft
    qpi_0_ls = qpi.calculate_qpi_from_ldos_2d(ldos_0_ls)
    n_q = qpi_0_ls.shape[0]
    q_ls = np.linspace(-4, 4, n_q)
    # 子图
    plt.pcolormesh(q_ls, q_ls, np.abs(qpi_0_ls), cmap="cividis", clim=(0, 8))
    plt.axis("scaled")
    plt.xticks([-2, 0, 2])
    plt.yticks([-2, 0, 2])
    plt.tick_params(which="both", color="0.7")
    plt.xlabel(r"$q_1 \ (\pi/a')$")
    plt.ylabel(r"$q_2 \ (\pi/a')$")
    # plt.colorbar()
    #
    plt.subplot(234)
    # 读取
    file = sio.loadmat(fname_s0)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    pos_imp_ = (0.25, 0.25)
    # DoS_0(x1,x2)
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    x1_ls, y1_ls = [np.linspace(*(center + (-50, 50)), 401)] * 2
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)
    ldos_0_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, +1], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # cut line
    ldos_o = (ldos_diff_ls * np.tile([1, +1], 4)).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, pos_atom)
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 8, 81)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    pos_c2 = np.transpose([-x_cut, +x_cut]) + pos_imp
    pos_c3 = np.transpose([-x_cut, -x_cut]) + pos_imp
    ldos_c0_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_c1_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    ldos_c2_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c2)
    ldos_c3_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c3)
    # 子图
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_0_ls), cmap=cmap0, clim=clim_d)
    plt.axis("scaled")
    plt.plot(*pos_c0.T, "k-", lw=1)
    plt.plot(*pos_c1.T, "r--", lw=1)
    # plt.plot(*pos_c2.T, c="0.7", lw=1)
    # plt.plot(*pos_c3.T, c="0.7", lw=1)
    plt.xlim(center + (-15, 15))
    plt.ylim(center + (-15, 15))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    # plt.colorbar()
    #
    # DoS_z(x),(y)
    plt.subplot(235)
    # 子图
    plt.axhline(0.0, ls="-.", lw=1, color="0.7")
    plt.plot(x_cut * SR2, ldos_c0_ls, "k-", label=r"$x$")
    plt.plot(x_cut * SR2, ldos_c1_ls, "r--", label=r"$y$")
    # plt.plot(x_cut * SR2, ldos_c2_ls, "-", label=r"$-x$")
    # plt.plot(x_cut * SR2, ldos_c3_ls, "--", label=r"$-y$")
    plt.xlabel(r"Distance")
    plt.ylabel(r"$\delta$ LDoS")
    plt.legend(loc="lower right")
    #
    # QPI
    plt.subplot(236)
    # fft
    qpi_0_ls = qpi.calculate_qpi_from_ldos_2d(ldos_0_ls)
    n_q = qpi_0_ls.shape[0]
    q_ls = np.linspace(-4, 4, n_q)
    # 子图
    plt.pcolormesh(q_ls, q_ls, np.abs(qpi_0_ls), cmap="cividis", clim=(0, 8))
    plt.axis("scaled")
    plt.xticks([-2, 0, 2])
    plt.yticks([-2, 0, 2])
    plt.tick_params(which="both", color="0.7")
    plt.xlabel(r"$q_1 \ (\pi/a')$")
    plt.ylabel(r"$q_2 \ (\pi/a')$")
    # plt.colorbar()
    #
    oplt.number_subplots()
    plt.show()


def main_r5_mapimp():
    # mag imp
    fname_all = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi(0,0,0.2,0.2)sz,E0.07.mat"
    fname_c4t = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi(0,0,0.2,-0.2)sz,E0.07.mat"
    fname_3 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi(0,0,0.2sz,0),E0.07.mat"
    fname_4 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi(0,0,0,-0.2sz),E0.07.mat"
    #
    center = np.array((100, 100))
    clim_d = (-0.1, 0.1)
    atom_r = 0.4
    cmap0 = "Spectral_r"

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 4, 5 * 2+0.5), dpi=90).set_layout_engine("compressed")

    # 读取
    file = sio.loadmat(fname_all)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    pos_imp_ = (0.25, 0.25)
    pos_imp_ = [(-0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.25, -0.25)][2]
    #
    # DoS_0(x1,x2)
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    x1_ls, y1_ls = [np.linspace(*(center + (-50, 50)), 401)] * 2
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)
    ldos_0_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, +1], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # cut line
    ldos_o = (ldos_diff_ls * np.tile([1, +1], 4)).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, pos_atom)
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 8, 81)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    pos_c2 = np.transpose([-x_cut, +x_cut]) + pos_imp
    pos_c3 = np.transpose([-x_cut, -x_cut]) + pos_imp
    ldos_c0_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_c1_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    ldos_c2_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c2)
    ldos_c3_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c3)
    # 子图
    plt.subplot(241)
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_0_ls), cmap=cmap0, clim=clim_d)
    plt.axis("scaled")
    plt.plot(*pos_c0.T, "k-", lw=1)
    plt.plot(*pos_c1.T, "r--", lw=1)
    # plt.plot(*pos_c2.T, c="0.7", lw=1)
    # plt.plot(*pos_c3.T, c="0.7", lw=1)
    plt.xlim(center + (-15, 15))
    plt.ylim(center + (-15, 15))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    # plt.colorbar()
    #
    # DoS_z(x),(y)
    plt.subplot(245)
    plt.axhline(0.0, ls="-.", lw=1, color="0.7")
    plt.plot(x_cut * SR2, ldos_c0_ls, "k-", label=r"$x$")
    plt.plot(x_cut * SR2, ldos_c1_ls, "r--", label=r"$y$")
    # plt.plot(x_cut * SR2, ldos_c2_ls, "-", label=r"$-x$")
    # plt.plot(x_cut * SR2, ldos_c3_ls, "--", label=r"$-y$")
    plt.xlabel(r"Distance")
    plt.ylabel(r"$\delta$ LDoS")
    plt.legend(loc="lower right")
    #
    # 读取
    file = sio.loadmat(fname_c4t)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    pos_imp_ = (0.25, 0.25)
    pos_imp_ = [(-0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.25, -0.25)][2]
    #
    # DoS_0(x1,x2)
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    x1_ls, y1_ls = [np.linspace(*(center + (-50, 50)), 401)] * 2
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    # pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)
    ldos_0_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, +1], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # cut line
    ldos_o = (ldos_diff_ls * np.tile([1, +1], 4)).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, pos_atom)
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 8, 81)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    pos_c2 = np.transpose([-x_cut, +x_cut]) + pos_imp
    pos_c3 = np.transpose([-x_cut, -x_cut]) + pos_imp
    ldos_c0_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_c1_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    ldos_c2_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c2)
    ldos_c3_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c3)
    # 子图
    plt.subplot(242)
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_0_ls), cmap=cmap0, clim=clim_d)
    plt.axis("scaled")
    plt.plot(*pos_c0.T, "k-", lw=1)
    plt.plot(*pos_c1.T, "r--", lw=1)
    # plt.plot(*pos_c2.T, c="0.7", lw=1)
    # plt.plot(*pos_c3.T, c="0.7", lw=1)
    plt.xlim(center + (-15, 15))
    plt.ylim(center + (-15, 15))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    # plt.colorbar()
    #
    # DoS_z(x),(y)
    plt.subplot(246)
    plt.axhline(0.0, ls="-.", lw=1, color="0.7")
    plt.plot(x_cut * SR2, ldos_c0_ls, "k-", label=r"$x$")
    plt.plot(x_cut * SR2, ldos_c1_ls, "r--", label=r"$y$")
    # plt.plot(x_cut * SR2, ldos_c2_ls, "-", label=r"$-x$")
    # plt.plot(x_cut * SR2, ldos_c3_ls, "--", label=r"$-y$")
    # plt.yticks([-0.06, -0.03, 0, 0.03])
    plt.xlabel(r"Distance")
    # plt.ylabel(r"$\delta LDoS_0$")
    plt.legend(loc="lower right")
    #
    # 读取
    file = sio.loadmat(fname_3)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    pos_imp_ = [(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 0)][2]
    #
    # DoS_0(x1,x2)
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    x1_ls, y1_ls = [np.linspace(*(center + (-50, 50)), 401)] * 2
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    # pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)
    ldos_0_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, +1], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # cut line
    ldos_o = (ldos_diff_ls * np.tile([1, +1], 4)).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, pos_atom)
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 8, 81)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    pos_c2 = np.transpose([-x_cut, +x_cut]) + pos_imp
    pos_c3 = np.transpose([-x_cut, -x_cut]) + pos_imp
    ldos_c0_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_c1_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    ldos_c2_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c2)
    ldos_c3_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c3)
    # 子图
    plt.subplot(243)
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_0_ls), cmap=cmap0, clim=clim_d)
    plt.axis("scaled")
    plt.plot(*pos_c0.T, "k-", lw=1)
    plt.plot(*pos_c1.T, "r--", lw=1)
    # plt.plot(*pos_c2.T, c="0.7", lw=1)
    # plt.plot(*pos_c3.T, c="0.7", lw=1)
    plt.xlim(center + (-15, 15))
    plt.ylim(center + (-15, 15))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    # plt.colorbar()
    #
    # DoS_z(x),(y)
    plt.subplot(247)
    plt.axhline(0.0, ls="-.", lw=1, color="0.7")
    plt.plot(x_cut * SR2, ldos_c0_ls, "k-", label=r"$x$")
    plt.plot(x_cut * SR2, ldos_c1_ls, "r--", label=r"$y$")
    # plt.plot(x_cut * SR2, ldos_c2_ls, "-", label=r"$-x$")
    # plt.plot(x_cut * SR2, ldos_c3_ls, "--", label=r"$-y$")
    plt.yticks([-0.06, -0.03, 0, 0.03])
    plt.xlabel(r"Distance")
    plt.ylabel(r"$\delta LDoS_0$")
    plt.legend(loc="lower right")
    #
    # 读取
    file = sio.loadmat(fname_4)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    pos_imp_ = [(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 0)][3]
    #
    # DoS_0(x1,x2)
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    x1_ls, y1_ls = [np.linspace(*(center + (-50, 50)), 401)] * 2
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    # pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)
    ldos_0_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, +1], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # cut line
    ldos_o = (ldos_diff_ls * np.tile([1, +1], 4)).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, pos_atom)
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 8, 81)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    pos_c2 = np.transpose([-x_cut, +x_cut]) + pos_imp
    pos_c3 = np.transpose([-x_cut, -x_cut]) + pos_imp
    ldos_c0_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_c1_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    ldos_c2_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c2)
    ldos_c3_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c3)
    # 子图
    plt.subplot(244)
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_0_ls), cmap=cmap0, clim=clim_d)
    plt.axis("scaled")
    plt.plot(*pos_c0.T, "k-", lw=1)
    plt.plot(*pos_c1.T, "r--", lw=1)
    # plt.plot(*pos_c2.T, c="0.7", lw=1)
    # plt.plot(*pos_c3.T, c="0.7", lw=1)
    plt.xlim(center + (-15, 15))
    plt.ylim(center + (-15, 15))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.colorbar()
    #
    # DoS_z(x),(y)
    plt.subplot(248)
    plt.axhline(0.0, ls="-.", lw=1, color="0.7")
    plt.plot(x_cut * SR2, ldos_c0_ls, "k-", label=r"$x$")
    plt.plot(x_cut * SR2, ldos_c1_ls, "r--", label=r"$y$")
    # plt.plot(x_cut * SR2, ldos_c2_ls, "-", label=r"$-x$")
    # plt.plot(x_cut * SR2, ldos_c3_ls, "--", label=r"$-y$")
    plt.yticks([-0.06, -0.03, 0, 0.03])
    plt.xlabel(r"Distance")
    # plt.ylabel(r"$\delta LDoS_0$")
    plt.legend(loc="lower right")
    #
    oplt.number_subplots()
    plt.show()


def main_r6_mag_qpi():
    fname_q3 = "./data/" + "QPI_KV2Se2O-SDW_(t1)0.03,Vi-0.2,E0.07.mat"
    fname_q6 = "./data/" + "QPI_KV2Se2O-SDW_(t1)0.03,Vi-0.2σzsz,E0.07.mat"

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 2, 5 + 0.5), dpi=90).set_layout_engine("compressed", wspace=0.06)
    #
    # QPI
    # 读取
    file = sio.loadmat(fname_q3)
    qpi_0_ls = file["qpi_0_ls"]
    n_q = qpi_0_ls.shape[0]
    q_ls = np.linspace(-4, 4, n_q)
    # 子图
    plt.subplot(121)
    plt.pcolormesh(q_ls, q_ls, np.abs(qpi_0_ls), cmap="cividis", clim=(0, 8))
    plt.axis("scaled")
    plt.xticks([-2, 0, 2])
    plt.yticks([-2, 0, 2])
    plt.tick_params(which="both", color="0.7")
    plt.xlabel(r"$q_1 \ (\pi/a')$")
    plt.ylabel(r"$q_2 \ (\pi/a')$")
    # plt.colorbar()
    #
    # 读取
    file = sio.loadmat(fname_q6)
    qpi_0_ls = file["qpi_0_ls"]
    # 子图
    plt.subplot(122)
    plt.pcolormesh(q_ls, q_ls, np.abs(qpi_0_ls), cmap="cividis", clim=(0, 8))
    plt.axis("scaled")
    plt.xticks([-2, 0, 2])
    plt.yticks([-2, 0, 2])
    plt.tick_params(which="both", color="0.7")
    plt.xlabel(r"$q_1 \ (\pi/a')$")
    plt.ylabel(r"$q_2 \ (\pi/a')$")
    plt.colorbar(pad=0.06)
    #
    oplt.number_subplots()
    plt.show()


def main_r7_ldos_pi_v1():
    fname0 = "./data/" + "DoS0_KV2Se2O-SDW_(t1)0.03,E0.04.mat"
    fname = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi(0,0,-0.2,-0.2),E0.04.mat"
    pos_imp_ = [(-0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.25, -0.25)][2]
    # pos_imp_ = [(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 0)][2]
    # pos_imp_ = (0.25, 0.25)
    center = np.array((100, 100))
    cmap_d = ["Spectral_r", "viridis", "cividis"][2]
    clim_d = None
    clim_q = (0, 8)
    atom_r = 0.6
    rt_k = 0.5
    x_lim0 = (0,9.9)

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 2, 5 * 2), dpi=60).set_layout_engine(hspace=0.06, wspace=0.06)
    # 读取
    dos0 = sio.loadmat(fname0)["dos0"][0]
    file = sio.loadmat(fname)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    ldos_diff_ls = (ldos_diff_ls) * 1 + dos0 * 1
    #
    # DoS_0(x1,x2)
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    x1_ls, y1_ls = [np.linspace(*(center + (-50, 50)), 401)] * 2
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)
    ldos_0_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, +1], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # cut line
    ldos_o_up = (ldos_diff_ls * np.tile([1, 0], 4)).flatten()
    ldos_o_dn = (ldos_diff_ls * np.tile([0, 1], 4)).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, pos_atom)
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 5, 81)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    ldos_c0_ls_up = qpi.transform_ldos_orbit2position_iso(ldos_o_up, gd, pos0, pos_c0)
    ldos_c0_ls_dn = qpi.transform_ldos_orbit2position_iso(ldos_o_dn, gd, pos0, pos_c0)
    ldos_c0_ls = ldos_c0_ls_up + ldos_c0_ls_dn
    ldos_c1_ls_up = qpi.transform_ldos_orbit2position_iso(ldos_o_up, gd, pos0, pos_c1)
    ldos_c1_ls_dn = qpi.transform_ldos_orbit2position_iso(ldos_o_dn, gd, pos0, pos_c1)
    ldos_c1_ls = ldos_c1_ls_up + ldos_c1_ls_dn
    # K-latt
    ldos_k_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        np.ones((*ldos_diff_ls.shape[:2], 2)) * [1, +1], gd, np.asarray([[-0.25, 0.25]] * 2), x0_ls, y0_ls, x1_ls, y1_ls
    )
    ldos_0_ls = ldos_0_ls * 1 + rt_k * ldos_k_ls
    ldos_o = (np.ones((*ldos_diff_ls.shape[:2], 2)) * [1, +1]).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, np.asarray([[-0.25, 0.25]] * 2))
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 5, 81)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    ldos_k_c0 = rt_k * qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_k_c1 = rt_k * qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    # ldos_c0_ls = ldos_c0_ls * 1 + ldos_k_c0
    # ldos_c1_ls = ldos_c1_ls * 1 + ldos_k_c1
    # 子图
    plt.subplot(221)
    # plt.plot(x_cut * 2, ldos_c0_ls_up, "-", label=r"$x$")
    # plt.plot(x_cut * 2, ldos_c0_ls_dn, "-", label=r"$x$")
    # plt.plot(x_cut * 2, ldos_c1_ls_up, "--", label=r"$y$")
    # plt.plot(x_cut * 2, ldos_c1_ls_dn, "--", label=r"$y$")
    plt.plot(x_cut * 2, ldos_c0_ls, "-", label=r"$x$")
    plt.plot(x_cut * 2, ldos_c1_ls, "C3--", label=r"$y$")
    plt.yticks([4.65,4.67,4.69])
    plt.xlim(x_lim0)
    ylim1 = plt.ylim()
    plt.xlabel(r"Distance")
    plt.ylabel(r"LDoS of V atoms")
    plt.legend()
    # --- 输出子图1数据 ---
    data_sub1 = np.column_stack((x_cut * 2, ldos_c0_ls, ldos_c1_ls))
    np.savetxt("./to/subplot_221_V_atoms.txt", data_sub1, header="Distance ldos_v(x) ldos_v(y)", fmt="%.6e")
    #
    # DoS_K
    plt.subplot(223)
    plt.plot(x_cut * 2, ldos_k_c0, "k", label=r"$x$")
    # plt.plot(x_cut * 2, ldos_k_c1, "--", label=r"$y$")
    plt.yticks([2.24,2.26,2.28])
    plt.xlim(x_lim0)
    plt.ylim(ylim1-np.average(ylim1)+2.26)
    # plt.yticks([2.25,2.26,2.27])
    plt.xlabel(r"Distance")
    plt.ylabel(r"LDoS of K atoms")
    # plt.legend()
    # --- 输出子图2数据 ---
    data_sub2 = np.column_stack((x_cut * 2, ldos_k_c0))
    np.savetxt("./to/subplot_223_K_atoms.txt", data_sub2, header="Distance ldos_k(x)", fmt="%.6e")
    #
    # DoS_z(x),(y)
    ldos_c0_ls = ldos_c0_ls * 1 + ldos_k_c0
    ldos_c1_ls = ldos_c1_ls * 1 + ldos_k_c1
    plt.subplot(222)
    # plt.axhline(0.0, ls="-.", lw=1, color="0.7")
    # ldos_z_ls1 = [ldos_0_ls[200 + i, 200 + i] for i in range(51)]
    # ldos_z_ls2 = [ldos_0_ls[200 + i, 200 - i] for i in range(51)]
    # plt.plot(x1_ls[200:251] - x_imp, ldos_z_ls1, ".-", label=r"$x$")
    # plt.plot(x1_ls[200:251] - y_imp, ldos_z_ls2, ".-", label=r"$y$")
    plt.plot(x_cut * 2, ldos_c0_ls, "-", label=r"$x$")
    plt.plot(x_cut * 2, ldos_c1_ls, "C3--", label=r"$y$")
    # plt.plot(x_cut * SR2, ldos_c2_ls, "-", label=r"$-x$")
    # plt.plot(x_cut * SR2, ldos_c3_ls, "--", label=r"$-y$")
    plt.yticks([6.91,6.93,6.95])
    plt.xlim(x_lim0)
    plt.xlabel(r"Distance")
    plt.ylabel(r"total LDoS")
    plt.legend()  
    # --- 输出子图3数据 ---
    data_sub3 = np.column_stack((x_cut * 2, ldos_c0_ls, ldos_c1_ls))
    np.savetxt(
        "./to/subplot_222_total_LDoS.txt", data_sub3, header="Distance ldos_tot(x) ldos_tot(y)", fmt="%.6e"
    )
    #
    # DoS_z(x1,x2)
    # x0_ls = x_ls[0]
    # y0_ls = y_ls[:, 0]
    # x1_ls, y1_ls = [np.linspace(100 - 50, 100 + 50, 401)] * 2
    # atom_r = 0.4
    # gd = lambda x: gaussian_delta(x, sigma=atom_r)
    pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)
    ldos_z_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, 0.2], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # cut line
    ldos_o = (ldos_diff_ls * np.tile([1, 0.2], 4)).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, pos_atom)
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 5, 81)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    pos_c2 = np.transpose([-x_cut, +x_cut]) + pos_imp
    pos_c3 = np.transpose([-x_cut, -x_cut]) + pos_imp
    ldos_c0_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_c1_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    ldos_c2_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c2)
    ldos_c3_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c3)
    # K-latt
    ldos_k_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        np.ones((*ldos_diff_ls.shape[:2], 2)) * [1, 0.2],
        gd,
        np.asarray([[-0.25, 0.25], [-0.25, 0.25]]),
        x0_ls,
        y0_ls,
        x1_ls,
        y1_ls,
    )
    ldos_z_ls = ldos_z_ls * 1 + rt_k * ldos_k_ls
    ldos_o = (np.ones((*ldos_diff_ls.shape[:2], 2)) * [1, 0.2]).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, np.asarray([[-0.25, 0.25]] * 2))
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 5, 81)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    ldos_c0_ls = ldos_c0_ls * 1 + rt_k * qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_c1_ls = ldos_c1_ls * 1 + rt_k * qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    # 子图
    # DoS_z(x),(y)
    # plt.subplot(224)
    # plt.plot(x_cut * 2, ldos_c0_ls, "-", label=r"$x$")
    # plt.plot(x_cut * 2, ldos_c1_ls, "C3--", label=r"$y$")
    # # plt.plot(x_cut * 2, ldos_c2_ls, "-", label=r"$-x$")
    # # plt.plot(x_cut * 2, ldos_c3_ls, "--", label=r"$-y$")
    # plt.yticks([4.1,4.15,4.2])
    # plt.xlim(x_lim0)
    # plt.ylim([4.06,4.22])
    # plt.xlabel(r"Distance")
    # plt.ylabel(r"spin-resolved LDOS")
    # # plt.legend(loc="upper right")
    # plt.legend()
    plt.subplot(224)
    # 创建双Y轴，ax1 为左侧Y轴，ax2 为右侧Y轴
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    # 左侧Y轴绘制 x 数据 (使用默认蓝色)
    line1, = ax1.plot(x_cut * 2, ldos_c0_ls, "-", color="C0", label=r"$x$")
    # 右侧Y轴绘制 y 数据 (使用红色)
    line2, = ax2.plot(x_cut * 2, ldos_c1_ls, "--", color="C3", label=r"$y$")
    # --- 左侧Y轴设置 ---
    ax1.set_xlim(x_lim0)
    ax1.set_ylim([4.095, 4.175])
    # ax1.set_yticks([4.1, 4.15, 4.2])
    ax1.set_xlabel(r"Distance")
    ax1.set_ylabel(r"spin-resolved LDOS")
    # 设置左侧Y轴刻度和标签颜色与曲线一致
    ax1.tick_params(axis='y', colors='C0')
    # --- 右侧Y轴设置 ---
    # 根据数据范围自行调整右侧Y轴的ylim，这里给一个示例范围，您可以按需修改
    ax2.set_ylim([4.145, 4.225]) 
    # 设置右侧Y轴刻度和标签颜色与曲线一致
    ax2.tick_params(axis='y', colors='C3')
    # --- 图例合并 ---
    # 因为分属两个不同的Axes，直接legend会生成两个，这里将线条合并到一个图例中
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")
    # --- 输出子图4数据 ---
    data_sub4 = np.column_stack((x_cut * 2, ldos_c0_ls, ldos_c1_ls))
    np.savetxt(
        "./to/subplot_224_spin_resolved_LDOS.txt",
        data_sub4,
        header="Distance ldos_up1dn0.2(x) ldos_up1dn0.2(y)",
        fmt="%.6e",
    )
    #
    oplt.number_subplots(plt.gcf().axes[:4], in_layout=np.arange(1))
    plt.savefig("./to/fig_re_ldos_pi.pdf", dpi=300)
    plt.savefig("./to/fig_re_ldos_pi.png", dpi=300)
    plt.show()


def main_r7_ldos_pi():
    fname0 = "./data/" + "DoS0_KV2Se2O-SDW_(t1)0.03,E0.07.mat"
    fname = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi(0,-0.2,-0.2,0),E0.07.mat"
    pos_imp_ = [(-0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.25, -0.25)][1]
    # pos_imp_ = [(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 0)][2]
    # pos_imp_ = (0.25, 0.25)
    center = np.array((100, 100))
    cmap_d = ["Spectral_r", "viridis", "cividis"][2]
    clim_d = None
    clim_q = (0, 8)
    atom_r = 0.6
    rt_k = 0.8
    x_lim0 = (0, 4)

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 2, 5 * 2+1), dpi=90).set_layout_engine("compressed", hspace=0.06, wspace=0.03)
    # 读取
    dos0 = sio.loadmat(fname0)["dos0"][0]
    file = sio.loadmat(fname)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    ldos_diff_ls = (ldos_diff_ls) * 1 + dos0 * 1
    #
    # DoS_0(x1,x2)
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    x1_ls, y1_ls = [np.linspace(*(center + (-50, 50)), 401)] * 2
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)
    ldos_0_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, +1], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # cut line
    ldos_o_up = (ldos_diff_ls * np.tile([1, 0], 4)).flatten()
    ldos_o_dn = (ldos_diff_ls * np.tile([0, 1], 4)).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, pos_atom)
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 5, 101)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    ldos_c0_ls_up = qpi.transform_ldos_orbit2position_iso(ldos_o_up, gd, pos0, pos_c0)
    ldos_c0_ls_dn = qpi.transform_ldos_orbit2position_iso(ldos_o_dn, gd, pos0, pos_c0)
    ldos_c0_ls = ldos_c0_ls_up + ldos_c0_ls_dn
    ldos_c1_ls_up = qpi.transform_ldos_orbit2position_iso(ldos_o_up, gd, pos0, pos_c1)
    ldos_c1_ls_dn = qpi.transform_ldos_orbit2position_iso(ldos_o_dn, gd, pos0, pos_c1)
    ldos_c1_ls = ldos_c1_ls_up + ldos_c1_ls_dn
    # K-latt
    ldos_k_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        np.ones((*ldos_diff_ls.shape[:2], 2)) * [1, +1], gd, np.asarray([[-0.25, 0.25]] * 2), x0_ls, y0_ls, x1_ls, y1_ls
    )
    ldos_0_ls = ldos_0_ls * 1 + rt_k * ldos_k_ls
    ldos_o = (np.ones((*ldos_diff_ls.shape[:2], 2)) * [1, +1]).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, np.asarray([[-0.25, 0.25]] * 2))
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 5, 101)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    ldos_k_c0 = rt_k * qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_k_c1 = rt_k * qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    ldos_c0_ls = ldos_c0_ls * 1 + ldos_k_c0
    ldos_c1_ls = ldos_c1_ls * 1 + ldos_k_c1
    # 子图
    plt.subplot(221)
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_0_ls), cmap=cmap_d, clim=clim_d)
    plt.axis("scaled")
    plt.plot(*pos_c0.T, c="C3", ls="--")
    plt.plot(*pos_c1.T, c="C0")
    plt.xticks([92,100,108])
    plt.yticks([92, 100, 108])
    plt.xticks([])
    plt.yticks([])
    plt.xlim(center + (-9.9, 9.9))
    plt.ylim(center + (-9.9, 9.9))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #
    # DoS_z(x),(y)
    plt.subplot(222)
    # plt.axhline(0.0, ls="-.", lw=1, color="0.7")
    # ldos_z_ls1 = [ldos_0_ls[200 + i, 200 + i] for i in range(51)]
    # ldos_z_ls2 = [ldos_0_ls[200 + i, 200 - i] for i in range(51)]
    # plt.plot(x1_ls[200:251] - x_imp, ldos_z_ls1, ".-", label=r"$x$")
    # plt.plot(x1_ls[200:251] - y_imp, ldos_z_ls2, ".-", label=r"$y$")
    plt.plot(x_cut * SR2*0.404*SR2, ldos_c0_ls, "C3--", label=r"$x$")
    plt.plot(x_cut * SR2*0.404*SR2, ldos_c1_ls, "-", label=r"$y$")
    # plt.plot(x_cut * SR2, ldos_c2_ls, "-", label=r"$-x$")
    # plt.plot(x_cut * SR2, ldos_c3_ls, "--", label=r"$-y$")
    plt.axvline(3.1/SR2*0.404*SR2, ls=(0, (5, 3)), lw=1, color="0.7")
    plt.axvline(5.0/SR2*0.404*SR2, ls=(0, (5, 3)), lw=1, color="0.7")
    plt.axvline(6.9/SR2*0.404*SR2, ls=(0, (5, 3)), lw=1, color="0.7")
    plt.xticks(np.arange(0, 5, 2))
    plt.yticks([7.8, 7.85, 7.9])
    plt.xlim(x_lim0)
    plt.xlabel(r"Distance (nm)")
    plt.ylabel(r"LDoS")
    plt.legend()
    # --- 输出子图3数据 ---
    data_sub3 = np.column_stack((x_cut * 2, ldos_c0_ls, ldos_c1_ls))
    np.savetxt("./to/subplot_222_total_LDoS.txt", data_sub3, header="Distance ldos_tot(x) ldos_tot(y)", fmt="%.6e")
    #
    # DoS_z(x1,x2)
    # x0_ls = x_ls[0]
    # y0_ls = y_ls[:, 0]
    # x1_ls, y1_ls = [np.linspace(100 - 50, 100 + 50, 401)] * 2
    # atom_r = 0.4
    # gd = lambda x: gaussian_delta(x, sigma=atom_r)
    pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)
    ldos_z_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, 0.5], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # cut line
    ldos_o = (ldos_diff_ls * np.tile([1, 0.5], 4)).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, pos_atom)
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 5, 101)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    pos_c2 = np.transpose([-x_cut, +x_cut]) + pos_imp
    pos_c3 = np.transpose([-x_cut, -x_cut]) + pos_imp
    ldos_c0_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_c1_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    ldos_c2_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c2)
    ldos_c3_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c3)
    # K-latt
    rt_k=0.2
    ldos_k_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        np.ones((*ldos_diff_ls.shape[:2], 2)) * [1, 0.5],
        gd,
        np.asarray([[-0.25, 0.25], [-0.25, 0.25]]),
        x0_ls,
        y0_ls,
        x1_ls,
        y1_ls,
    )
    ldos_z_ls = ldos_z_ls * 1 + rt_k * ldos_k_ls
    ldos_o = (np.ones((*ldos_diff_ls.shape[:2], 2)) * [1, 0.5]).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, np.asarray([[-0.25, 0.25]] * 2))
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 5, 101)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    ldos_c0_ls = ldos_c0_ls * 1 + rt_k * qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_c1_ls = ldos_c1_ls * 1 + rt_k * qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    # 子图
    plt.subplot(223)
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_z_ls), cmap=cmap_d, clim=clim_d)
    plt.axis("scaled")
    plt.plot(*pos_c0.T, c="C3", ls="--")
    plt.plot(*pos_c1.T, c="C0")
    plt.xticks([92,100,108])
    plt.yticks([92,100,108])
    plt.xticks([])
    plt.yticks([])
    plt.xlim(center + (-9.9, 9.9))
    plt.ylim(center + (-9.9, 9.9))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    # 子图
    # DoS_z(x),(y)
    plt.subplot(224)
    plt.plot(x_cut * SR2*0.404*SR2, ldos_c0_ls, "C3--", label=r"$x$")
    plt.plot(x_cut * SR2*0.404*SR2, ldos_c1_ls, "-", label=r"$y$")
    # plt.plot(x_cut * 2, ldos_c2_ls, "-", label=r"$-x$")
    # plt.plot(x_cut * 2, ldos_c3_ls, "--", label=r"$-y$")
    plt.axvline(2.6/SR2*0.404*SR2, ls=(0, (5, 3)), lw=1, color="0.7")
    plt.axvline(4.8/SR2*0.404*SR2, ls=(0, (5, 3)), lw=1, color="0.7")
    plt.axvline(7.0/SR2*0.404*SR2, ls=(0, (5, 3)), lw=1, color="0.7")
    plt.xticks(np.arange(0, 5, 2))
    plt.yticks([3.8,3.85,3.9])
    plt.xlim(x_lim0)
    # plt.ylim([4.06,4.22])
    plt.xlabel(r"Distance (nm)")
    plt.ylabel(r"LDOS")
    plt.legend(loc="center right")
    # plt.subplot(224)
    # # 创建双Y轴，ax1 为左侧Y轴，ax2 为右侧Y轴
    # ax1 = plt.gca()
    # ax2 = ax1.twinx()
    # # 左侧Y轴绘制 x 数据 (使用默认蓝色)
    # (line1,) = ax1.plot(x_cut * 2, ldos_c0_ls, "-", color="C0", label=r"$x$")
    # # 右侧Y轴绘制 y 数据 (使用红色)
    # (line2,) = ax2.plot(x_cut * 2, ldos_c1_ls, "--", color="C3", label=r"$y$")
    # # --- 左侧Y轴设置 ---
    # ax1.set_xlim(x_lim0)
    # # ax1.set_ylim([4.095, 4.175])
    # # ax1.set_yticks([4.1, 4.15, 4.2])
    # ax1.set_xlabel(r"Distance")
    # ax1.set_ylabel(r"spin-resolved LDOS")
    # # 设置左侧Y轴刻度和标签颜色与曲线一致
    # ax1.tick_params(axis="y", colors="C0")
    # # --- 右侧Y轴设置 ---
    # # 根据数据范围自行调整右侧Y轴的ylim，这里给一个示例范围，您可以按需修改
    # # ax2.set_ylim([4.145, 4.225])
    # # 设置右侧Y轴刻度和标签颜色与曲线一致
    # ax2.tick_params(axis="y", colors="C3")
    # # --- 图例合并 ---
    # # 因为分属两个不同的Axes，直接legend会生成两个，这里将线条合并到一个图例中
    # lines = [line1, line2]
    # labels = [l.get_label() for l in lines]
    # ax1.legend(lines, labels, loc="best")
    # --- 输出子图4数据 ---
    data_sub4 = np.column_stack((x_cut * 2, ldos_c0_ls, ldos_c1_ls))
    np.savetxt(
        "./to/subplot_224_spin_resolved_LDOS.txt",
        data_sub4,
        header="Distance ldos_up1dn0.2(x) ldos_up1dn0.2(y)",
        fmt="%.6e",
    )
    #
    oplt.number_subplots(plt.gcf().axes[:4], in_layout=np.arange(1))
    plt.savefig("./to/fig_re_ldos_pi.pdf", dpi=300)
    plt.savefig("./to/fig_re_ldos_pi.png", dpi=300)
    plt.show()

def main_r8_qpi_anisotropy_v1():
    fname_0p = "./data/QPI_KV2Se2O-SDW_(t1)0.03,Vi0.2,E0.04.mat"
    fname_0m = "./data/QPI_KV2Se2O-SDW_(t1)0.03,Vi0.2,E-0.04.mat"
    fname_1p = "./data/QPI_KV2Se2O-SDW_(t1)0.03,Vi(0.1,0.2,0.1,0.2),E0.04.mat"
    fname_1m = "./data/QPI_KV2Se2O-SDW_(t1)0.03,Vi(0.1,0.2,0.1,0.2),E-0.04.mat"
    fname_2p = "./data/QPI_KV2Se2O-SDW_(t1)0.03,M0,Vi0.2,E0.04.mat"
    fname_2m = "./data/QPI_KV2Se2O-SDW_(t1)0.03,M0,Vi0.2,E-0.04.mat"
    fname_0a = "./data/QPI_AnIso(E)_KV2Se2O-SDW_(t1)0.03,Vi0.2.mat"
    fname_1a = "./data/QPI_AnIso(E)_KV2Se2O-SDW_(t1)0.03,Vi(0.1,0.2,0.1,0.2).mat"
    fname_2a = "./data/QPI_AnIso(E)_KV2Se2O-SDW_(t1)0.03,M0,Vi0.2.mat"
    fname_0r = "./data/QPI_q-Rq(E)_KV2Se2O-SDW_(t1)0.03,Vi0.2.mat"
    fname_1r = "./data/QPI_q-Rq(E)_KV2Se2O-SDW_(t1)0.03,Vi(0.1,0.2,0.1,0.2).mat"
    fname_2r = "./data/QPI_q-Rq(E)_KV2Se2O-SDW_(t1)0.03,M0,Vi0.2.mat"

    n_q = 400
    q_ls = np.linspace(-4, 4, n_q)

    def plot_qpi(qpi_ls, text):
        plt.pcolormesh(q_ls, q_ls, np.abs(qpi_ls), cmap="cividis", clim=(0, 8))
        plt.axis("scaled")
        plt.xticks([-2, 0, 2])
        plt.yticks([-2, 0, 2])
        plt.tick_params(which="both", color="0.7")
        plt.xlabel(r"$q_1 \ (\pi/a')$")
        plt.ylabel(r"$q_2 \ (\pi/a')$")
        plt.text(-3.6, 3.2, text, c="w")

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 4, 5 * 2+2.4), dpi=60).set_layout_engine("compressed", hspace=0.12)
    #
    plt.subplot(241)
    # 读取
    file = sio.loadmat(fname_0p)
    qpi_ls = file["qpi_s_ls"]
    # 子图
    plot_qpi(qpi_ls, r"$+40$ meV")
    #
    plt.subplot(242)
    # 读取
    file = sio.loadmat(fname_0m)
    qpi_ls = file["qpi_s_ls"]
    # 子图
    plot_qpi(qpi_ls, r"$-40$ meV")
    #
    plt.subplot(243)
    # 读取
    file = sio.loadmat(fname_1p)
    qpi_ls = file["qpi_0_ls"]
    # 子图
    plot_qpi(qpi_ls, r"$+40$ meV")
    #
    plt.subplot(244)
    # 读取
    file = sio.loadmat(fname_1m)
    qpi_ls = file["qpi_0_ls"]
    # 子图
    plot_qpi(qpi_ls, r"$-40$ meV")
    #
    plt.subplot(245)
    # 读取
    file = sio.loadmat(fname_2p)
    qpi_ls = file["qpi_0_ls"]
    # 子图
    plot_qpi(qpi_ls, r"$+40$ meV")
    #
    plt.subplot(246)
    # 读取
    file = sio.loadmat(fname_2m)
    qpi_ls = file["qpi_0_ls"]
    # 子图
    plot_qpi(qpi_ls, r"$-40$ meV")
    #
    plt.subplot(2, 4, 7)
    # 读取
    file = sio.loadmat(fname_0r)
    eng_ls, aniso_ls0 = file["eng_ls"][0], file["aniso_ls"][0]
    file = sio.loadmat(fname_1r)
    eng_ls, aniso_ls1 = file["eng_ls"][0], file["aniso_ls"][0]
    file = sio.loadmat(fname_2r)
    eng_ls, aniso_ls2 = file["eng_ls"][0], file["aniso_ls"][0]
    # 子图
    plt.plot(eng_ls, aniso_ls0, "o-k", label="spin-sensitive")
    plt.plot(eng_ls, aniso_ls1, "v-C3", label="orbital-imbalanced")
    plt.plot(eng_ls, aniso_ls2, "s-C0", label="no AM order")
    # plt.ylim([0.0, None])
    plt.xticks(np.arange(-40, 41, 20))
    plt.yticks([0,0.2,0.4])
    plt.xlabel(r"$E$ (meV)")
    plt.ylabel(r"Norm$|I(q)-I(R_{\pi/2}q)|$")
    plt.legend()
    #
    plt.subplot(2,4,8)
    # 读取
    file = sio.loadmat(fname_0a)
    eng_ls, aniso_ls0 = file["eng_ls"][0], file["aniso_ls"][0]
    file = sio.loadmat(fname_1a)
    eng_ls, aniso_ls1 = file["eng_ls"][0], file["aniso_ls"][0]
    file = sio.loadmat(fname_2a)
    eng_ls, aniso_ls2 = file["eng_ls"][0], file["aniso_ls"][0]
    # 子图
    plt.axhline(0.0, ls="-.", lw=1, color="0.7")
    plt.plot(eng_ls, aniso_ls0, "o-k", label="spin-sensitive")
    plt.plot(eng_ls, aniso_ls1, "v-C3", label="orbital-imbalanced")
    # plt.plot(eng_ls, aniso_ls2, "s-C0", label="no AM order")
    # plt.xlim([-50,50])
    plt.xticks(np.arange(-40,41,20))
    plt.xlabel(r"$E$ (meV)")
    plt.ylabel(r"$A$", labelpad=-10)
    plt.legend()
    #
    oplt.number_subplots(plt.gcf().axes[::2]+[plt.gcf().axes[-1]])
    plt.show()


def main_r8_qpi_anisotropy():
    fname_0p = "./data/QPI_KV2Se2O-SDW_(t1)0.03,Vi0.2,E0.04.mat"
    fname_0m = "./data/QPI_KV2Se2O-SDW_(t1)0.03,Vi0.2,E-0.04.mat"
    fname_1p = "./data/QPI_KV2Se2O-SDW_(t1)0.03,Vi(0.1,0.2,0.1,0.2),E0.04.mat"
    fname_1m = "./data/QPI_KV2Se2O-SDW_(t1)0.03,Vi(0.1,0.2,0.1,0.2),E-0.04.mat"
    fname_2p = "./data/QPI_KV2Se2O-SDW_(t1)0.03,M0,Vi0.2,E0.04.mat"
    fname_2m = "./data/QPI_KV2Se2O-SDW_(t1)0.03,M0,Vi0.2,E-0.04.mat"
    fname_0a = "./data/QPI_AnIso(E)_KV2Se2O-SDW_(t1)0.03,Vi0.2.mat"
    fname_1a = "./data/QPI_AnIso(E)_KV2Se2O-SDW_(t1)0.03,Vi(0.1,0.2,0.1,0.2).mat"
    fname_2a = "./data/QPI_AnIso(E)_KV2Se2O-SDW_(t1)0.03,M0,Vi0.2.mat"
    fname_2r = "./data/QPI_q-Rq(E)_KV2Se2O-SDW_(t1)0.03,M0,Vi0.2_q(2,1).mat"
    fname_2rs = "./data/QPI_q-Rq(E)_KV2Se2O-SDW_(t1)0.03,M0,Vi0.2_q(2,1)_s.mat"
    fname_2r = "./data/QPI_q-Rq(E)_0.mat"
    fname_2r = "./data/QPI_q-Rq(E)_s.mat"

    n_q = 400
    q_ls = np.linspace(-4, 4, n_q)

    def plot_qpi(qpi_ls, text):
        plt.pcolormesh(q_ls, q_ls, np.abs(qpi_ls), cmap="cividis", clim=(0, 8))
        plt.axis("scaled")
        plt.xticks([-2, 0, 2])
        plt.yticks([-2, 0, 2])
        plt.tick_params(which="both", color="0.7")
        plt.xlabel(r"$q_1 \ (\pi/a')$")
        plt.ylabel(r"$q_2 \ (\pi/a')$")
        plt.text(-3.6, 3.2, text, c="w")

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 4, 5 * 2+1), dpi=60).set_layout_engine("compressed", hspace=0.09)
    #
    plt.subplot(241)
    # 读取
    file = sio.loadmat(fname_0p)
    qpi_ls = file["qpi_s_ls"]
    # 子图
    plot_qpi(qpi_ls, r"$+40$ meV")
    #
    plt.subplot(242)
    # 读取
    file = sio.loadmat(fname_0m)
    qpi_ls = file["qpi_s_ls"]
    # 子图
    plot_qpi(qpi_ls, r"$-40$ meV")
    #
    plt.subplot(245)
    # 读取
    file = sio.loadmat(fname_1p)
    qpi_ls = file["qpi_0_ls"]
    # 子图
    plot_qpi(qpi_ls, r"$+40$ meV")
    #
    plt.subplot(246)
    # 读取
    file = sio.loadmat(fname_1m)
    qpi_ls = file["qpi_0_ls"]
    # 子图
    plot_qpi(qpi_ls, r"$-40$ meV")
    #
    plt.subplot(2, 4, (3,8))
    # 读取
    file = sio.loadmat(fname_0a)
    eng_ls, aniso_ls0 = file["eng_ls"][0], file["aniso_ls"][0]
    file = sio.loadmat(fname_1a)
    eng_ls, aniso_ls1 = file["eng_ls"][0], file["aniso_ls"][0]
    file = sio.loadmat(fname_2a)
    eng_ls, aniso_ls2 = file["eng_ls"][0], file["aniso_ls"][0]
    # 子图
    plt.axhline(0.0, ls="-.", lw=1, color="0.7")
    plt.plot(eng_ls, aniso_ls0, "o-k", label="spin-sensitive")
    plt.plot(eng_ls, aniso_ls1, "v-C3", label="orbital-imbalanced impurities")
    # plt.plot(eng_ls, aniso_ls2, "s-C0", label="no AM order")
    # plt.xlim([-50,50])
    plt.xticks(np.arange(-40, 41, 20))
    plt.xlabel(r"$E$ (meV)")
    plt.ylabel(r"$A$", labelpad=-10)
    plt.legend()
    #
    oplt.number_subplots(plt.gcf().axes[::2])
    plt.show()

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 3, 5 * 1+1), dpi=60).set_layout_engine("compressed", hspace=0.09)
    #
    plt.subplot(131)
    # 读取
    file = sio.loadmat(fname_2p)
    qpi_ls = file["qpi_0_ls"]
    # 子图
    plot_qpi(qpi_ls, r"$+40$ meV")
    #
    plt.subplot(132)
    # 读取
    file = sio.loadmat(fname_2m)
    qpi_ls = file["qpi_0_ls"]
    # 子图
    plot_qpi(qpi_ls, r"$-40$ meV")
    #
    plt.subplot(133)
    # 读取
    file = sio.loadmat(fname_2r)
    eng_ls, aniso_ls2 = file["eng_ls"][0], file["aniso_ls"][0]
    file = sio.loadmat(fname_2rs)
    eng_ls, aniso_ls2s = file["eng_ls"][0], file["aniso_ls"][0]
    # 子图
    plt.axhline(0.0, ls="-.", lw=1, color="0.7")
    # plt.plot(eng_ls, aniso_ls0, "o-k", label="spin-sensitive")
    # plt.plot(eng_ls, aniso_ls1, "v-C3", label="orbital-imbalanced")
    plt.plot(eng_ls, aniso_ls2, "s-C0", label="spin-insensitive")
    plt.plot(eng_ls, aniso_ls2s, "s--C1", label="spin-sensitive")
    plt.ylim([-0.12, None])
    plt.xticks(np.arange(-40, 41, 20))
    plt.yticks(np.arange(-0.1, 0.11,0.1))
    plt.xlabel(r"$E$ (meV)")
    plt.ylabel(r"$A'$", labelpad=-10)
    plt.legend()
    #
    oplt.number_subplots(plt.gcf().axes[::2])
    plt.show()


def main_r9_conclu():
    fname_togra = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi(0,0,2,0),EΣ0.3.mat"
    fname_dos = "./data/" + "DoS0_KV2Se2O-SDW_(t1)0.03,E0.07.mat"
    fname_ldos = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi(0,-0.2,-0.2,0),E0.07.mat"
    fname_0a = "./data/QPI_AnIso(E)_KV2Se2O-SDW_(t1)0.03,Vi0.2.mat"
    fname_1a = "./data/QPI_AnIso(E)_KV2Se2O-SDW_(t1)0.03,Vi(0.1,0.2,0.1,0.2).mat"
    fname_2a = "./data/QPI_AnIso(E)_KV2Se2O-SDW_(t1)0.03,M0,Vi0.2.mat"

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 4, 5 * 1+0.8), dpi=90)
    #
    plt.subplot(141)
    center = np.array((100, 100))
    clim_d = (0.5, 2)  # (-0.1, 0.1)
    atom_r = 0.4
    atom_r_k = 0.4
    cmap0 = "viridis"  # "Spectral_r" #"viridis"
    dos0_v = 2
    weight_v = 6
    pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)
    # 读取
    file = sio.loadmat(fname_togra)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    # DoS_V
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    x1_ls, y1_ls = [np.linspace(*(center + (-50, 50)), 801)] * 2
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    ldos_0_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, +1], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    ln_ldos_ls = -np.log1p(ldos_0_ls / dos0_v)
    # signal_K
    gd_k = lambda x: gaussian_delta(x, sigma=atom_r_k)
    alt_k_ls = np.ones((*(ldos_diff_ls.shape[:2]), 1))
    pos_atom_k = np.asarray([[-0.25, 0.25]])
    alt_k_ls = qpi.transform_ldos_orbit2position_in_Cartesian(alt_k_ls, gd_k, pos_atom_k, x0_ls, y0_ls, x1_ls, y1_ls)
    alt_tot = alt_k_ls + weight_v * alt_k_ls * ln_ldos_ls
    # 子图
    plt.pcolormesh(x1_ls, y1_ls, alt_tot, cmap=cmap0, clim=clim_d)
    plt.axis("scaled")
    plt.xlim(center + (-9, 9))
    plt.ylim(center + (-9, 9))
    plt.xticks([])
    plt.yticks([])
    # plt.xlabel(r"$x_1$")
    # plt.ylabel(r"$x_2$")
    plt.title(" ")
    #
    plt.subplot(143)
    pos_imp_ = [(-0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.25, -0.25)][1]
    # pos_imp_ = [(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 0)][2]
    # pos_imp_ = (0.25, 0.25)
    center = np.array((100, 100))
    clim_d = None
    atom_r = 0.6
    rt_k = 0.8
    x_lim0 = (0, 4)
    # 读取
    dos0 = sio.loadmat(fname_dos)["dos0"][0]
    file = sio.loadmat(fname_ldos)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    ldos_diff_ls = (ldos_diff_ls) * 1 + dos0 * 1
    #
    # DoS_0(x1,x2)
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    x1_ls, y1_ls = [np.linspace(*(center + (-50, 50)), 401)] * 2
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)
    ldos_0_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, +1], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # cut line
    ldos_o_up = (ldos_diff_ls * np.tile([1, 0], 4)).flatten()
    ldos_o_dn = (ldos_diff_ls * np.tile([0, 1], 4)).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, pos_atom)
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 5, 101)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    ldos_c0_ls_up = qpi.transform_ldos_orbit2position_iso(ldos_o_up, gd, pos0, pos_c0)
    ldos_c0_ls_dn = qpi.transform_ldos_orbit2position_iso(ldos_o_dn, gd, pos0, pos_c0)
    ldos_c0_ls = ldos_c0_ls_up + ldos_c0_ls_dn
    ldos_c1_ls_up = qpi.transform_ldos_orbit2position_iso(ldos_o_up, gd, pos0, pos_c1)
    ldos_c1_ls_dn = qpi.transform_ldos_orbit2position_iso(ldos_o_dn, gd, pos0, pos_c1)
    ldos_c1_ls = ldos_c1_ls_up + ldos_c1_ls_dn
    # K-latt
    ldos_k_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        np.ones((*ldos_diff_ls.shape[:2], 2)) * [1, +1], gd, np.asarray([[-0.25, 0.25]] * 2), x0_ls, y0_ls, x1_ls, y1_ls
    )
    ldos_0_ls = ldos_0_ls * 1 + rt_k * ldos_k_ls
    ldos_o = (np.ones((*ldos_diff_ls.shape[:2], 2)) * [1, +1]).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, np.asarray([[-0.25, 0.25]] * 2))
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 5, 101)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    ldos_k_c0 = rt_k * qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_k_c1 = rt_k * qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    ldos_c0_ls = ldos_c0_ls * 1 + ldos_k_c0
    ldos_c1_ls = ldos_c1_ls * 1 + ldos_k_c1
    #
    plt.plot(x_cut * SR2 * 0.404 * SR2, ldos_c0_ls, "C3--", label=r"$x$")
    plt.plot(x_cut * SR2 * 0.404 * SR2, ldos_c1_ls, "-", label=r"$y$")
    # plt.plot(x_cut * SR2, ldos_c2_ls, "-", label=r"$-x$")
    # plt.plot(x_cut * SR2, ldos_c3_ls, "--", label=r"$-y$")
    plt.axvline(3.1 / SR2 * 0.404 * SR2, ls=(0, (5, 3)), lw=1, color="0.7")
    plt.axvline(5.0 / SR2 * 0.404 * SR2, ls=(0, (5, 3)), lw=1, color="0.7")
    plt.axvline(6.9 / SR2 * 0.404 * SR2, ls=(0, (5, 3)), lw=1, color="0.7")
    plt.xticks(np.arange(0, 5, 2))
    plt.yticks([7.8, 7.85, 7.9])
    plt.xlim(x_lim0)
    plt.xlabel(r"Distance (nm)")
    plt.ylabel(r"LDoS")
    plt.legend()
    #
    plt.subplot(142)
    #
    pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)
    ldos_z_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, 0.5], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # cut line
    ldos_o = (ldos_diff_ls * np.tile([1, 0.5], 4)).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, pos_atom)
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 5, 101)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    pos_c2 = np.transpose([-x_cut, +x_cut]) + pos_imp
    pos_c3 = np.transpose([-x_cut, -x_cut]) + pos_imp
    ldos_c0_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_c1_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    ldos_c2_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c2)
    ldos_c3_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c3)
    # K-latt
    rt_k = 0.2
    ldos_k_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        np.ones((*ldos_diff_ls.shape[:2], 2)) * [1, 0.5],
        gd,
        np.asarray([[-0.25, 0.25], [-0.25, 0.25]]),
        x0_ls,
        y0_ls,
        x1_ls,
        y1_ls,
    )
    ldos_z_ls = ldos_z_ls * 1 + rt_k * ldos_k_ls
    ldos_o = (np.ones((*ldos_diff_ls.shape[:2], 2)) * [1, 0.5]).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, np.asarray([[-0.25, 0.25]] * 2))
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 5, 101)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    ldos_c0_ls = ldos_c0_ls * 1 + rt_k * qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_c1_ls = ldos_c1_ls * 1 + rt_k * qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    #
    plt.plot(x_cut * SR2 * 0.404 * SR2, ldos_c0_ls, "C3--", label=r"$x$")
    plt.plot(x_cut * SR2 * 0.404 * SR2, ldos_c1_ls, "-", label=r"$y$")
    # plt.plot(x_cut * 2, ldos_c2_ls, "-", label=r"$-x$")
    # plt.plot(x_cut * 2, ldos_c3_ls, "--", label=r"$-y$")
    plt.axvline(2.6 / SR2 * 0.404 * SR2, ls=(0, (5, 3)), lw=1, color="0.7")
    plt.axvline(4.8 / SR2 * 0.404 * SR2, ls=(0, (5, 3)), lw=1, color="0.7")
    plt.axvline(7.0 / SR2 * 0.404 * SR2, ls=(0, (5, 3)), lw=1, color="0.7")
    plt.xticks(np.arange(0, 5, 2))
    plt.yticks([3.8, 3.85, 3.9])
    plt.xlim(x_lim0)
    # plt.ylim([4.06,4.22])
    plt.xlabel(r"Distance (nm)")
    plt.ylabel(r"LDOS")
    plt.legend(loc="center right")
    #
    plt.subplot(144)
    # 读取
    file = sio.loadmat(fname_0a)
    eng_ls, aniso_ls0 = file["eng_ls"][0], file["aniso_ls"][0]
    file = sio.loadmat(fname_1a)
    eng_ls, aniso_ls1 = file["eng_ls"][0], file["aniso_ls"][0]
    file = sio.loadmat(fname_2a)
    eng_ls, aniso_ls2 = file["eng_ls"][0], file["aniso_ls"][0]
    # 子图
    plt.axhline(0.0, ls="-.", lw=1, color="0.7")
    plt.plot(eng_ls, aniso_ls0, "o-k", label="spin-sensitive")
    plt.plot(eng_ls, aniso_ls1, "v-C3", label="orbital-imbalanced")
    # plt.plot(eng_ls, aniso_ls2, "s-C0", label="no AM order")
    # plt.xlim([-50,50])
    plt.xticks(np.arange(-40,41,20))
    plt.xlabel(r"$E$ (meV)")
    plt.ylabel(r"$A$", labelpad=-10)
    plt.legend()
    #
    plt.show()


def main_r10_ldos():
    fname_all = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi0.2,E0.07.mat"
    fname_3 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi(0,0,0.2,0),E0.07.mat"
    fname_4 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi(0,0,0,0.2),E0.07.mat"
    #
    center = np.array((100, 100))
    clim_d = (-0.1, 0.1)
    atom_r = 0.4
    cmap0 = "Spectral_r"

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 3, 5 * 2), dpi=100).set_layout_engine("compressed")

    # 读取
    file = sio.loadmat(fname_all)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    pos_imp_ = (0.25, 0.25)
    #
    # DoS_0(x1,x2)
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    x1_ls, y1_ls = [np.linspace(*(center + (-50, 50)), 401)] * 2
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)
    ldos_0_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, 0.5], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # cut line
    ldos_o = (ldos_diff_ls * np.tile([1, 0.5], 4)).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, pos_atom)
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 8, 81)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    pos_c2 = np.transpose([-x_cut, +x_cut]) + pos_imp
    pos_c3 = np.transpose([-x_cut, -x_cut]) + pos_imp
    ldos_c0_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_c1_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    ldos_c2_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c2)
    ldos_c3_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c3)
    # 子图
    plt.subplot(231)
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_0_ls), cmap=cmap0, clim=clim_d)
    plt.axis("scaled")
    plt.plot(*pos_c0.T, "k-", lw=1)
    plt.plot(*pos_c1.T, "r--", lw=1)
    # plt.plot(*pos_c2.T, c="0.7", lw=1)
    # plt.plot(*pos_c3.T, c="0.7", lw=1)
    plt.xlim(center + (-15, 15))
    plt.ylim(center + (-15, 15))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    # plt.colorbar()
    #
    # DoS_z(x),(y)
    plt.subplot(234)
    plt.axhline(0.0, ls="-.", lw=1, color="0.7")
    plt.plot(x_cut * SR2, ldos_c0_ls, "k-", label=r"$x$")
    plt.plot(x_cut * SR2, ldos_c1_ls, "r--", label=r"$y$")
    # plt.plot(x_cut * SR2, ldos_c2_ls, "-", label=r"$-x$")
    # plt.plot(x_cut * SR2, ldos_c3_ls, "--", label=r"$-y$")
    plt.xlabel(r"Distance")
    plt.ylabel(r"$\delta$ LDoS")
    plt.legend(loc="lower right")
    #
    # 读取
    file = sio.loadmat(fname_3)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    pos_imp_ = [(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 0)][2]
    #
    # DoS_0(x1,x2)
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    x1_ls, y1_ls = [np.linspace(*(center + (-50, 50)), 401)] * 2
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    # pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)
    ldos_0_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, 0.5], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # cut line
    ldos_o = (ldos_diff_ls * np.tile([1, 0.5], 4)).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, pos_atom)
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 8, 81)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    pos_c2 = np.transpose([-x_cut, +x_cut]) + pos_imp
    pos_c3 = np.transpose([-x_cut, -x_cut]) + pos_imp
    ldos_c0_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_c1_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    ldos_c2_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c2)
    ldos_c3_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c3)
    # 子图
    plt.subplot(232)
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_0_ls), cmap=cmap0, clim=clim_d)
    plt.axis("scaled")
    plt.plot(*pos_c0.T, "k-", lw=1)
    plt.plot(*pos_c1.T, "r--", lw=1)
    # plt.plot(*pos_c2.T, c="0.7", lw=1)
    # plt.plot(*pos_c3.T, c="0.7", lw=1)
    plt.xlim(center + (-15, 15))
    plt.ylim(center + (-15, 15))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    # plt.colorbar()
    #
    # DoS_z(x),(y)
    plt.subplot(235)
    plt.axhline(0.0, ls="-.", lw=1, color="0.7")
    plt.plot(x_cut * SR2, ldos_c0_ls, "k-", label=r"$x$")
    plt.plot(x_cut * SR2, ldos_c1_ls, "r--", label=r"$y$")
    # plt.plot(x_cut * SR2, ldos_c2_ls, "-", label=r"$-x$")
    # plt.plot(x_cut * SR2, ldos_c3_ls, "--", label=r"$-y$")
    plt.yticks([-0.06, -0.03, 0, 0.03])
    plt.xlabel(r"Distance")
    # plt.ylabel(r"$\delta LDoS_0$")
    plt.legend(loc="lower right")
    #
    # 读取
    file = sio.loadmat(fname_4)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    pos_imp_ = [(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 0)][3]
    #
    # DoS_0(x1,x2)
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    x1_ls, y1_ls = [np.linspace(*(center + (-50, 50)), 401)] * 2
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    # pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)
    ldos_0_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, 0.5], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # cut line
    ldos_o = (ldos_diff_ls * np.tile([1, 0.5], 4)).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, pos_atom)
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 8, 81)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    pos_c2 = np.transpose([-x_cut, +x_cut]) + pos_imp
    pos_c3 = np.transpose([-x_cut, -x_cut]) + pos_imp
    ldos_c0_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_c1_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    ldos_c2_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c2)
    ldos_c3_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c3)
    # 子图
    plt.subplot(233)
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_0_ls), cmap=cmap0, clim=clim_d)
    plt.axis("scaled")
    plt.plot(*pos_c0.T, "k-", lw=1)
    plt.plot(*pos_c1.T, "r--", lw=1)
    # plt.plot(*pos_c2.T, c="0.7", lw=1)
    # plt.plot(*pos_c3.T, c="0.7", lw=1)
    plt.xlim(center + (-15, 15))
    plt.ylim(center + (-15, 15))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.colorbar()
    #
    # DoS_z(x),(y)
    plt.subplot(236)
    plt.axhline(0.0, ls="-.", lw=1, color="0.7")
    plt.plot(x_cut * SR2, ldos_c0_ls, "k-", label=r"$x$")
    plt.plot(x_cut * SR2, ldos_c1_ls, "r--", label=r"$y$")
    # plt.plot(x_cut * SR2, ldos_c2_ls, "-", label=r"$-x$")
    # plt.plot(x_cut * SR2, ldos_c3_ls, "--", label=r"$-y$")
    plt.yticks([-0.06, -0.03, 0, 0.03])
    plt.xlabel(r"Distance")
    # plt.ylabel(r"$\delta LDoS_0$")
    plt.legend(loc="lower right")
    #
    oplt.number_subplots()
    plt.show()


def main_dos():
    fname_disp0 = "./data/E(k)_KV2Se2O.mat"
    fname_disp = "./data/E(k).mat"
    fname_dos0 = "./data/DoS_KV2Se2O.mat"
    fname_dos = "./data/DoS.mat"

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 2, 5 * 2), dpi=120)
    #
    # 读取
    file = sio.loadmat(fname_disp0)
    k_sym_ls0, eng_ls0 = file["k_path"][:, 0], file["eng_ls"]
    file = sio.loadmat(fname_disp)
    k_sym_ls, eng_ls = file["k_path"][:, 0], file["eng_ls"]
    #
    plt.subplot(221)
    plt.plot(k_sym_ls0[:], eng_ls0[:], c="k")
    plt.plot(k_sym_ls[:], eng_ls[:], c="C0")
    plt.axhline(0, ls=":", lw=1, color="tab:gray")
    plt.yticks([-0.4, 0, 0.4])
    plt.axvline(np.pi, color="k")
    plt.axvline(np.pi * 2, color="k")
    plt.xticks([0, np.pi, np.pi * 2, k_sym_ls[-1]], labels=[r"$\Gamma$", "X", "M", r"$\Gamma$"])
    plt.ylim(-0.7, 0.7)
    plt.ylabel(r"$E$ (meV)")
    #
    # 读取
    file = sio.loadmat(fname_dos0)
    eng_ls0, dos_ls0 = file["eng_ls"][0], file["dos_ls"]
    file = sio.loadmat(fname_dos)
    eng_ls, dos_ls = file["eng_ls"][0], file["dos_ls"]  # [0]
    #
    plt.subplot(223)
    plt.plot(eng_ls0, dos_ls0[:, 0], "k")
    plt.plot(eng_ls, dos_ls[:, 0])
    plt.plot(eng_ls, dos_ls[:, 3])
    plt.plot(eng_ls, dos_ls[:, 4])
    plt.plot(eng_ls, dos_ls[:, 7])
    # plt.ylim(bottom=0)
    plt.xlabel(r"$E$")
    plt.ylabel(r"DoS")
    #
    plt.subplot(224)
    plt.plot(eng_ls0, np.sum(dos_ls0, axis=-1), "k")
    plt.plot(eng_ls, np.sum(dos_ls, axis=-1))
    # plt.ylim(bottom=0)
    plt.xlabel(r"$E$")
    plt.ylabel(r"DoS")
    #
    plt.show()


""" ================================================================================
    RUN
================================================================================ """
if __name__ == "__main__":
    # main_qpi_disp()
    # main_qpi_disp_plot()
    # main_qpi_d90()
    # main_s1_before_fold()
    # main_s2_folded()
    # main_s3_ldos()
    # main_s4_qpi()
    # main_s5_d90()
    # main_s5_dm110()
    # main_r1_topography()
    # main_r2_diff_2sdw()
    # main_r2_2_2mpoint()
    # main_r3_so()
    # main_r4_0sdw()
    # main_r5_mapimp()
    # main_r6_mag_qpi()
    # main_r7_ldos_pi()
    main_r8_qpi_anisotropy()
    # main_r9_conclu()
    # main_r10_ldos()
    # main_dos()
