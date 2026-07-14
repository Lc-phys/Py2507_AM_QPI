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

SR2 = np.sqrt(2)


def main_ldos_diff_x_y_imp1uc_plot_uc_at():
    fname = "./data/" + "δLDoS(x,y).mat"

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 2, 5 * 2 - 1), dpi=100).set_layout_engine("compressed")

    # 读取
    file = sio.loadmat(fname)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    opr = [1, +1] * (ldos_diff_ls.shape[-1] // 2)
    ldos_diff_ls_uc = np.sum(ldos_diff_ls * opr, axis=-1)
    # 子图
    plt.subplot(221)
    plt.pcolormesh(x_ls, y_ls, np.real(ldos_diff_ls_uc), cmap="Spectral_r", clim=(-0.2, 0.2))
    plt.axis("scaled")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.colorbar()
    # 处理
    qpi_ls = qpi.calculate_qpi_from_ldos_2d(ldos_diff_ls_uc)
    qx_ls, qy_ls = qpi.get_q_ls(qpi_ls.shape)
    # 子图
    plt.subplot(223)
    plt.pcolormesh(qx_ls / np.pi, qy_ls / np.pi, np.abs(qpi_ls), cmap="viridis", clim=(0, 8))
    plt.axis("scaled")
    plt.xlabel(r"$q_x/\pi$")
    plt.ylabel(r"$q_y/\pi$")
    plt.colorbar()
    #
    # # IFFT(QPI)
    # inv_qpi_ls= np.fft.ifft2((qpi_ls))
    # # 子图
    # plt.subplot(223)
    # plt.pcolormesh(np.real(inv_qpi_ls), cmap="Spectral_r", clim=(-0.2, 0.2))
    # plt.axis("scaled")
    # plt.xlabel(r"$x$")
    # plt.ylabel(r"$y$")
    # plt.colorbar()
    # #
    # # periodic QPI
    # qpi_ls = np.block([[qpi_ls, qpi_ls, qpi_ls], [qpi_ls, qpi_ls, qpi_ls], [qpi_ls, qpi_ls, qpi_ls]])
    # # 子图
    # plt.subplot(224)
    # plt.pcolormesh(np.abs(qpi_ls), cmap="viridis", clim=(0, 8))
    # plt.axis("scaled")
    # plt.xlabel(r"$q_x$")
    # plt.ylabel(r"$q_y$")
    # plt.colorbar()
    #
    # atom DoS
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    x1_ls, y1_ls = [np.linspace(100 - 50, 100 + 50, 401)] * 2
    atom_r = 0.4
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    pos_atom = np.repeat([[0, 0], [0.5, 0], [0.5, -0.5], [0, -0.5]], 2, axis=0)
    ldos_a_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, -1], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # 子图
    plt.subplot(222)
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_a_ls), cmap="Spectral_r", clim=(-0.1, 0.1))
    plt.axis("scaled")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.colorbar()
    #
    # atom DoS QPI
    qpi_ls = qpi.calculate_qpi_from_ldos_2d(ldos_a_ls)
    # 子图
    plt.subplot(224)
    plt.pcolormesh(np.abs(qpi_ls), cmap="viridis", clim=(0, 10))
    plt.axis("scaled")
    plt.xlabel(r"$q_x$")
    plt.ylabel(r"$q_y$")
    plt.colorbar()
    #
    plt.show()


def main_ldos_diff_x_y_imp1uc():
    # model0 = Model_AM_xy((200, 200))
    # model0.t_hop = [1.0, -0.3]
    # model0.am = 1.6
    # model0 = Model_KV2Se2O_xy((200, 200))
    model0 = Model_KV2Se2O_SDW_xy_rev((201, 201))
    model0.set_parameter()
    model0.m0[0] = -0.
    # model0.m0[1] = -0.0
    # model0.t1 = 0.03
    # model0.so = 0.
    model0.mu = -np.sqrt(model0.m0[0] ** 2 + 4 * model0.t1**2)
    model0.mu = -0.

    eng = 0.04
    eps = 5e-3
    n_orb = model0.n_orbit
    # opr = np.eye(n_orb)
    # opr = np.diag([1, 1] * (model0.n_orbit // 2))
    x_ls, y_ls = model0.lattice.tag_ls.T
    n_s = model0.lattice.n_site

    model0.v_imp = 0.0
    hmtn0 = model0.get_Hamiltonian()
    model0.v_imp = 0.2
    # hmtn1 = model0.get_Hamiltonian()

    # 格林函数
    idx_imp = model0.lattice.tag2idx(model0.tag_imp)
    gfr = ih.calculate_Green_function(hmtn0, eng, +eps, r0_ls=[idx_imp], n_orb=n_orb)
    gf_r0 = np.reshape(gfr.toarray(), (n_s, n_orb, n_orb))
    gfa = ih.calculate_Green_function(hmtn0, eng, -eps, r0_ls=[idx_imp], n_orb=n_orb)
    gf_0r = np.reshape(gfa.toarray(), (n_s, n_orb, n_orb))
    gf_0r = np.transpose(gf_0r, (0, 2, 1)).conj()
    gf_00 = gf_r0[idx_imp]

    # 杂质Hamiltonian
    imp_mtx = model0.v_imp * np.kron(np.diag([1,1,1,1]), SIGMA_0)  # ([1] * 2 + [0] * 6)
    # imp_mtx = model0.v_imp * np.kron(np.diag([1, 0]), np.kron(SIGMA_X, SIGMA_X))
    # LDOS差值
    gf_diff_r = qpi.calculate_greenfunc_difference_from_impurity(imp_mtx, gf_00, gf_r0, gf_0r)
    ldos_diff_ls = -np.imag([np.diag(gf) for gf in gf_diff_r]) / np.pi  # DoS(r,o)
    print("δLDoS Calculation COMPLETE.")

    # 保存
    x_ls, y_ls = [np.reshape(arr, (model0.n_y, model0.n_x)) for arr in [x_ls, y_ls]]
    ldos_diff_ls = np.reshape(ldos_diff_ls, (model0.n_y, model0.n_x, n_orb))
    sio.savemat("./data/" + "δLDoS(x,y).mat", {"x_ls": x_ls, "y_ls": y_ls, "ldos_diff_ls": ldos_diff_ls})


def main_ldos_diff_x_y():
    # model0 = Model_AM_xy((200, 200))
    # model0.t_hop = [1.0, -0.3]
    # model0.am = 1.6
    # model0 = Model_KV2Se2O_xy((200, 200))
    model0 = Model_KV2Se2O_SDW_xy((200, 200))
    model0.set_parameter()
    # model0.m0[0] = 0
    model0.m0[1] = -0.02
    # model0.t1 = -0.2
    model0.mu = -np.sqrt(model0.m0[0] ** 2 + 4 * model0.t1**2)

    mu = -0.08
    eps = 5e-3
    n_orb = model0.n_orbit
    # opr = np.eye(n_orb)
    # opr = np.diag([1, 1] * (model0.n_orbit // 2))
    x_ls, y_ls = model0.lattice.tag_ls.T
    n_s = model0.lattice.n_site

    model0.v_imp = 0.0
    hmtn0 = model0.get_Hamiltonian()

    # 杂质Hamiltonian
    v_imp_ls = [0.2] + [0.1] * 4
    imp_ctr = np.asarray([model0.n_x // 2, model0.n_y // 2])
    idx_imp_ls = model0.lattice.tag2idx(imp_ctr + [(0, 0), (-1, 0), (0, -1)])
    imp_mtx = np.diag(
        np.kron([v_imp_ls[0], v_imp_ls[2], 0, v_imp_ls[1]] + [0, 0, 0, v_imp_ls[3]] + [0, v_imp_ls[4], 0, 0], [1, 1])
    )
    # 次近邻
    v_imp_ls = [0.2, 0.05, 0.03]
    idx_imp_ls = model0.lattice.tag2idx(imp_ctr + [(0, 0), (-1, 0), (-1, -1), (0, -1)])
    imp_mtx = np.diag(
        np.kron(
            [v_imp_ls[0], v_imp_ls[1], v_imp_ls[2], v_imp_ls[1]]
            + [0, 0, v_imp_ls[2], v_imp_ls[1]]
            + [0, 0, v_imp_ls[2], 0]
            + [0, v_imp_ls[1], v_imp_ls[2], 0],
            [1, 1],
        )
    )

    # 格林函数
    # idx_imp = model0.lattice.tag2idx(model0.tag_imp)
    gfr = ih.calculate_Green_function(hmtn0, mu, +eps, r0_ls=idx_imp_ls, n_orb=n_orb)
    gf_r0 = np.reshape(gfr.toarray(), (n_s, n_orb, -1))
    gfa = ih.calculate_Green_function(hmtn0, mu, -eps, r0_ls=idx_imp_ls, n_orb=n_orb)
    gf_0r = np.reshape(gfa.toarray(), (n_s, n_orb, -1))
    gf_0r = np.transpose(gf_0r, (0, 2, 1)).conj()
    gf_00 = np.concatenate(gf_r0[idx_imp_ls], axis=0)
    # gf_00 = ih.calculate_Green_function(hmtn0, mu, eps, r1_ls=idx_imp_ls, r0_ls=idx_imp_ls, n_orb=n_orb)

    # LDOS差值
    gf_diff_r = qpi.calculate_greenfunc_difference_from_impurity(imp_mtx, gf_00, gf_r0, gf_0r)
    ldos_diff_ls = -np.imag([np.diag(gf) for gf in gf_diff_r]) / np.pi  # DoS(r,o)
    print("δLDoS Calculation COMPLETE.")

    # 保存
    x_ls, y_ls = [np.reshape(arr, (model0.n_y, model0.n_x)) for arr in [x_ls, y_ls]]
    ldos_diff_ls = np.reshape(ldos_diff_ls, (model0.n_y, model0.n_x, n_orb))
    sio.savemat("./data/" + "δLDoS(x,y).mat", {"x_ls": x_ls, "y_ls": y_ls, "ldos_diff_ls": ldos_diff_ls})


def main_ldos_diff_x_y_plot_0z():
    fname = "./data/" + "δLDoS(x,y).mat"
    pos_imp_ = [(0.25, 0.25), (0.75, -0.25), (0.25, -0.75), (-0.25, -0.25)][0]
    # pos_imp_ = [(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 0)][0]
    pos_imp_ = (0.25, -0.25)
    center = np.array((100, 100))
    clim_d = (-0.1, 0.1)
    clim_q = (0, 8)
    atom_r = 0.4

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 3, 5 * 2 - 1), dpi=100).set_layout_engine("compressed")

    # 读取
    file = sio.loadmat(fname)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    #
    # DoS_0(x1,x2)
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    x1_ls, y1_ls = [np.linspace(*(center + (-50, 50)), 401)] * 2
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    pos_atom = np.repeat([[0, 0], [0.5, 0], [0.5, -0.5], [0, -0.5]], 2, axis=0)
    ldos_0_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, +1], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # cut line
    ldos_o = (ldos_diff_ls * np.tile([1, +1], 4)).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, pos_atom)
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 8, 81)
    pos_c0 = np.transpose([+x_cut, +x_cut]) + pos_imp
    pos_c1 = np.transpose([-x_cut, +x_cut]) + pos_imp
    pos_c2 = np.transpose([-x_cut, -x_cut]) + pos_imp
    pos_c3 = np.transpose([+x_cut, -x_cut]) + pos_imp
    ldos_c0_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_c1_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    ldos_c2_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c2)
    ldos_c3_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c3)
    # 子图
    plt.subplot(231)
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_0_ls), cmap="Spectral_r", clim=clim_d)
    plt.axis("scaled")
    plt.plot(*pos_c0.T, c="0.7", lw=1)
    plt.plot(*pos_c1.T, c="0.7", lw=1)
    plt.plot(*pos_c2.T, c="0.7", lw=1)
    plt.plot(*pos_c3.T, c="0.7", lw=1)
    plt.xlim(center + (-15, 15))
    plt.ylim(center + (-15, 15))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.colorbar()
    #
    # DoS_z(x),(y)
    plt.subplot(232)
    plt.axhline(0.0, ls="-.", lw=1, color="0.7")
    # ldos_z_ls1 = [ldos_0_ls[200 + i, 200 + i] for i in range(51)]
    # ldos_z_ls2 = [ldos_0_ls[200 + i, 200 - i] for i in range(51)]
    # plt.plot(x1_ls[200:251] - x_imp, ldos_z_ls1, ".-", label=r"$x$")
    # plt.plot(x1_ls[200:251] - y_imp, ldos_z_ls2, ".-", label=r"$y$")
    plt.plot(x_cut * SR2, ldos_c0_ls, "-", label=r"$x$")
    plt.plot(x_cut * SR2, ldos_c1_ls, "--", label=r"$y$")
    plt.plot(x_cut * SR2, ldos_c2_ls, "-", label=r"$-x$")
    plt.plot(x_cut * SR2, ldos_c3_ls, "--", label=r"$-y$")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$\delta LDoS_0$")
    plt.legend()
    #
    # QPI
    qpi_0_ls = qpi.calculate_qpi_from_ldos_2d(ldos_0_ls)
    sio.savemat("./data/QPI.mat", {"qpi_0_ls": qpi_0_ls})
    # 子图
    plt.subplot(233)
    plt.pcolormesh(np.abs(qpi_0_ls), cmap="cividis", clim=clim_q)
    # plt.pcolormesh(np.imag(qpi_0_ls), cmap="coolwarm", clim=(-10, 10))
    plt.axis("scaled")
    plt.xlabel(r"$q_1$")
    plt.ylabel(r"$q_2$")
    plt.colorbar()
    #
    # DoS_z(x1,x2)
    # x0_ls = x_ls[0]
    # y0_ls = y_ls[:, 0]
    # x1_ls, y1_ls = [np.linspace(100 - 50, 100 + 50, 401)] * 2
    # atom_r = 0.4
    # gd = lambda x: gaussian_delta(x, sigma=atom_r)
    pos_atom = np.repeat([[0, 0], [0.5, 0], [0.5, -0.5], [0, -0.5]], 2, axis=0)
    ldos_z_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile([1, 0.5], 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # cut line
    ldos_o = (ldos_diff_ls * np.tile([1, 0.5], 4)).flatten()
    ldos_c0_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_c1_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    ldos_c2_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c2)
    ldos_c3_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c3)
    # 子图
    plt.subplot(234)
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_z_ls), cmap="Spectral_r", clim=clim_d)
    plt.axis("scaled")
    plt.plot(*pos_c0.T, c="0.7", lw=1)
    plt.plot(*pos_c1.T, c="0.7", lw=1)
    plt.plot(*pos_c2.T, c="0.7", lw=1)
    plt.plot(*pos_c3.T, c="0.7", lw=1)
    plt.xlim(center + (-15, 15))
    plt.ylim(center + (-15, 15))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.colorbar()
    #
    # DoS_z(x),(y)
    plt.subplot(235)
    plt.axhline(0.0, ls="-.", lw=1, color="0.7")
    # ldos_z_ls1 = [ldos_z_ls[200 + i, 200 + i] for i in range(51)]
    # ldos_z_ls2 = [ldos_z_ls[200 + i, 200 - i] for i in range(51)]
    # plt.plot(x1_ls[200:251] - x_imp, ldos_z_ls1, ".-", label=r"$x$")
    # plt.plot(x1_ls[200:251] - y_imp, ldos_z_ls2, ".-", label=r"$y$")
    plt.plot(x_cut * SR2, ldos_c0_ls, "-", label=r"$x$")
    plt.plot(x_cut * SR2, ldos_c1_ls, "--", label=r"$y$")
    plt.plot(x_cut * SR2, ldos_c2_ls, "-", label=r"$-x$")
    plt.plot(x_cut * SR2, ldos_c3_ls, "--", label=r"$-y$")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$\delta LDoS_z$")
    plt.legend(loc="upper right")
    #
    # QPI
    qpi_z_ls = qpi.calculate_qpi_from_ldos_2d(ldos_z_ls)
    # 子图
    plt.subplot(236)
    plt.pcolormesh(np.abs(qpi_z_ls), cmap="cividis", clim=clim_q)
    # plt.pcolormesh(np.imag(qpi_z_ls), cmap="coolwarm", clim=(-10, 10))
    plt.axis("scaled")
    plt.xlabel(r"$q_1$")
    plt.ylabel(r"$q_2$")
    plt.colorbar()
    #
    plt.show()


def main_ldos_diff_x_y_plot_0z_rev():
    fname = "./data/" + "δLDoS(x,y).mat"
    pos_imp_ = [(-0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.25, -0.25)][1]
    # pos_imp_ = [(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 0)][2]
    pos_imp_ = (0.25, 0.25)
    center = np.array((100, 100))
    clim_d = (-0.1, 0.1)
    clim_q = (0, 8)
    atom_r = 0.4
    spin_sens = [[1, 0.5],[0.5, 1]][0]

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 3, 5 * 2 - 1), dpi=100).set_layout_engine("compressed")

    # 读取
    file = sio.loadmat(fname)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    #
    # DoS_0(x1,x2)
    x0_ls = x_ls[0]
    y0_ls = y_ls[:, 0]
    x1_ls, y1_ls = [np.linspace(*(center + (-50, 50)), 400)] * 2
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
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_0_ls), cmap="Spectral_r", clim=clim_d)
    plt.axis("scaled")
    plt.plot(*pos_c0.T, c="0.7", lw=1)
    plt.plot(*pos_c1.T, c="0.7", lw=1)
    plt.plot(*pos_c2.T, c="0.7", lw=1)
    plt.plot(*pos_c3.T, c="0.7", lw=1)
    plt.xlim(center + (-15, 15))
    plt.ylim(center + (-15, 15))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.colorbar()
    #
    # DoS_z(x),(y)
    plt.subplot(232)
    plt.axhline(0.0, ls="-.", lw=1, color="0.7")
    # ldos_z_ls1 = [ldos_0_ls[200 + i, 200 + i] for i in range(51)]
    # ldos_z_ls2 = [ldos_0_ls[200 + i, 200 - i] for i in range(51)]
    # plt.plot(x1_ls[200:251] - x_imp, ldos_z_ls1, ".-", label=r"$x$")
    # plt.plot(x1_ls[200:251] - y_imp, ldos_z_ls2, ".-", label=r"$y$")
    plt.plot(x_cut * SR2, ldos_c0_ls, "-", label=r"$x$")
    plt.plot(x_cut * SR2, ldos_c1_ls, "--", label=r"$y$")
    plt.plot(x_cut * SR2, ldos_c2_ls, "-", label=r"$-x$")
    plt.plot(x_cut * SR2, ldos_c3_ls, "--", label=r"$-y$")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$\delta LDoS_0$")
    plt.legend()
    #
    # QPI
    qpi_0_ls = qpi.calculate_qpi_from_ldos_2d(ldos_0_ls)
    # 子图
    plt.subplot(233)
    plt.pcolormesh(np.abs(qpi_0_ls), cmap="cividis", clim=clim_q)
    # plt.pcolormesh(np.imag(qpi_0_ls), cmap="coolwarm", clim=(-10, 10))
    plt.axis("scaled")
    plt.xlabel(r"$q_1$")
    plt.ylabel(r"$q_2$")
    plt.colorbar()
    #
    # DoS_z(x1,x2)
    # x0_ls = x_ls[0]
    # y0_ls = y_ls[:, 0]
    # x1_ls, y1_ls = [np.linspace(100 - 50, 100 + 50, 401)] * 2
    # atom_r = 0.4
    # gd = lambda x: gaussian_delta(x, sigma=atom_r)
    pos_atom = np.repeat([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], 2, axis=0)
    ldos_z_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        ldos_diff_ls * np.tile(spin_sens, 4), gd, pos_atom, x0_ls, y0_ls, x1_ls, y1_ls
    )
    # cut line
    ldos_o = (ldos_diff_ls * np.tile(spin_sens, 4)).flatten()
    ldos_c0_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_c1_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    ldos_c2_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c2)
    ldos_c3_ls = qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c3)
    # 子图
    plt.subplot(234)
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_z_ls), cmap="Spectral_r", clim=clim_d)
    plt.axis("scaled")
    plt.plot(*pos_c0.T, c="0.7", lw=1)
    plt.plot(*pos_c1.T, c="0.7", lw=1)
    plt.plot(*pos_c2.T, c="0.7", lw=1)
    plt.plot(*pos_c3.T, c="0.7", lw=1)
    plt.xlim(center + (-15, 15))
    plt.ylim(center + (-15, 15))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.colorbar()
    #
    # DoS_z(x),(y)
    plt.subplot(235)
    plt.axhline(0.0, ls="-.", lw=1, color="0.7")
    # ldos_z_ls1 = [ldos_z_ls[200 + i, 200 + i] for i in range(51)]
    # ldos_z_ls2 = [ldos_z_ls[200 + i, 200 - i] for i in range(51)]
    # plt.plot(x1_ls[200:251] - x_imp, ldos_z_ls1, ".-", label=r"$x$")
    # plt.plot(x1_ls[200:251] - y_imp, ldos_z_ls2, ".-", label=r"$y$")
    plt.plot(x_cut * SR2, ldos_c0_ls, "-", label=r"$x$")
    plt.plot(x_cut * SR2, ldos_c1_ls, "--", label=r"$y$")
    plt.plot(x_cut * SR2, ldos_c2_ls, "-", label=r"$-x$")
    plt.plot(x_cut * SR2, ldos_c3_ls, "--", label=r"$-y$")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$\delta LDoS_z$")
    plt.legend(loc="upper right")
    #
    # QPI
    qpi_z_ls = qpi.calculate_qpi_from_ldos_2d(ldos_z_ls)
    sio.savemat("./data/QPI.mat", {"qpi_0_ls": qpi_0_ls, "qpi_s_ls": qpi_z_ls})
    # 子图
    plt.subplot(236)
    plt.pcolormesh(np.abs(qpi_z_ls), cmap="cividis", clim=clim_q)
    # plt.pcolormesh(np.imag(qpi_z_ls), cmap="coolwarm", clim=(-10, 10))
    plt.axis("scaled")
    plt.xlabel(r"$q_1$")
    plt.ylabel(r"$q_2$")
    plt.colorbar()
    #
    plt.show()


def main_full_ldos_x_y_plot_0z_rev():
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
    rt_k = 0.2

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 3, 5 * 2 - 1), dpi=100).set_layout_engine("compressed")
    # 读取
    dos0 = sio.loadmat(fname0)["dos0"][0]
    file = sio.loadmat(fname)
    x_ls, y_ls, ldos_diff_ls = file["x_ls"], file["y_ls"], file["ldos_diff_ls"]
    ldos_diff_ls = (ldos_diff_ls)*1+dos0*1
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
    # K-latt
    ldos_k_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
        np.ones((*ldos_diff_ls.shape[:2], 2)) * [1, +1],
        gd,
        np.asarray([[-0.25, 0.25]] * 2),
        x0_ls,
        y0_ls,
        x1_ls,
        y1_ls
    )
    ldos_0_ls = ldos_0_ls * 1 + rt_k * ldos_k_ls
    ldos_o = (np.ones((*ldos_diff_ls.shape[:2], 2)) * [1, +1]).flatten()
    pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    pos0 = qpi.get_pos_atom(pos0, np.asarray([[-0.25, 0.25]] * 2))
    pos_imp = center + pos_imp_
    x_cut = np.linspace(0, 8, 81)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    ldos_c0_ls = ldos_c0_ls * 1 + rt_k * qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_c1_ls = ldos_c1_ls * 1 + rt_k * qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    # 子图
    plt.subplot(231)
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_0_ls), cmap=cmap_d, clim=clim_d)
    plt.axis("scaled")
    plt.plot(*pos_c0.T, c="0.7", lw=1)
    plt.plot(*pos_c1.T, c="0.7", lw=1)
    plt.plot(*pos_c2.T, c="0.7", lw=1)
    plt.plot(*pos_c3.T, c="0.7", lw=1)
    plt.xlim(center + (-15, 15))
    plt.ylim(center + (-15, 15))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.colorbar()
    #
    # DoS_z(x),(y)
    plt.subplot(232)
    # plt.axhline(0.0, ls="-.", lw=1, color="0.7")
    # ldos_z_ls1 = [ldos_0_ls[200 + i, 200 + i] for i in range(51)]
    # ldos_z_ls2 = [ldos_0_ls[200 + i, 200 - i] for i in range(51)]
    # plt.plot(x1_ls[200:251] - x_imp, ldos_z_ls1, ".-", label=r"$x$")
    # plt.plot(x1_ls[200:251] - y_imp, ldos_z_ls2, ".-", label=r"$y$")
    plt.plot(x_cut * 2, ldos_c0_ls, "-", label=r"$x$")
    plt.plot(x_cut * 2, ldos_c1_ls, "--", label=r"$y$")
    # plt.plot(x_cut * SR2, ldos_c2_ls, "-", label=r"$-x$")
    # plt.plot(x_cut * SR2, ldos_c3_ls, "--", label=r"$-y$")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$\delta LDoS_0$")
    plt.legend()
    #
    # QPI
    qpi_0_ls = qpi.calculate_qpi_from_ldos_2d(ldos_0_ls)
    sio.savemat("./data/QPI.mat", {"qpi_0_ls": qpi_0_ls})
    # 子图
    plt.subplot(233)
    plt.pcolormesh(np.abs(qpi_0_ls), cmap="cividis", clim=clim_q)
    # plt.pcolormesh(np.imag(qpi_0_ls), cmap="coolwarm", clim=(-10, 10))
    plt.axis("scaled")
    plt.xlabel(r"$q_1$")
    plt.ylabel(r"$q_2$")
    plt.colorbar()
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
    x_cut = np.linspace(0, 8, 81)
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
    x_cut = np.linspace(0, 8, 81)
    pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    ldos_c0_ls = ldos_c0_ls * 1 + rt_k * qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c0)
    ldos_c1_ls = ldos_c1_ls * 1 + rt_k * qpi.transform_ldos_orbit2position_iso(ldos_o, gd, pos0, pos_c1)
    # 子图
    plt.subplot(234)
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_z_ls), cmap=cmap_d, clim=clim_d)
    plt.axis("scaled")
    plt.plot(*pos_c0.T, c="0.7", lw=1)
    plt.plot(*pos_c1.T, c="0.7", lw=1)
    plt.plot(*pos_c2.T, c="0.7", lw=1)
    plt.plot(*pos_c3.T, c="0.7", lw=1)
    plt.xlim(center + (-15, 15))
    plt.ylim(center + (-15, 15))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.colorbar()
    #
    # DoS_z(x),(y)
    plt.subplot(235)
    # plt.axhline(0.0, ls="-.", lw=1, color="0.7")
    # ldos_z_ls1 = [ldos_z_ls[200 + i, 200 + i] for i in range(51)]
    # ldos_z_ls2 = [ldos_z_ls[200 + i, 200 - i] for i in range(51)]
    # plt.plot(x1_ls[200:251] - x_imp, ldos_z_ls1, ".-", label=r"$x$")
    # plt.plot(x1_ls[200:251] - y_imp, ldos_z_ls2, ".-", label=r"$y$")
    plt.plot(x_cut * 2, ldos_c0_ls, "-", label=r"$x$")
    plt.plot(x_cut * 2, ldos_c1_ls, "--", label=r"$y$")
    # plt.plot(x_cut * SR2, ldos_c2_ls, "-", label=r"$-x$")
    # plt.plot(x_cut * SR2, ldos_c3_ls, "--", label=r"$-y$")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$\delta LDoS_z$")
    plt.legend(loc="upper right")
    #
    # QPI
    qpi_z_ls = qpi.calculate_qpi_from_ldos_2d(ldos_z_ls)
    # 子图
    plt.subplot(236)
    plt.pcolormesh(np.abs(qpi_z_ls), cmap="cividis", clim=clim_q)
    # plt.pcolormesh(np.imag(qpi_z_ls), cmap="coolwarm", clim=(-10, 10))
    plt.axis("scaled")
    plt.xlabel(r"$q_1$")
    plt.ylabel(r"$q_2$")
    plt.colorbar()
    #
    plt.show()


def main_qpi_mu_sdw():
    # model0 = Model_KV2Se2O_xy((200, 200))
    model0 = Model_KV2Se2O_SDW_xy((200, 200))
    model0.set_parameter()
    # model0.m0 = [-0.9, -0.02]
    # model0.t1 = 0.03
    # model0.mu = -np.sqrt(model0.m0[0] ** 2 + 4 * model0.t1**2)
    # model0.mu = -0.9

    mu_ls = np.linspace(-0.1, 0.1, 41)
    v_imp = -0.2
    eps = 5e-3
    x_ls = np.arange(-61, 62) + model0.n_x // 2
    y_ls = np.arange(-60, 61) + model0.n_y // 2

    # 杂质Hamiltonian
    imp_mtx = v_imp * np.diag(np.kron([1, 1, 1, 1], [1, 1]))  # ([1] * 2 + [0] * 6)
    # imp_mtx = model0.v_imp * np.kron(np.diag([1, 0]), np.kron(SIGMA_X, SIGMA_X))

    # 格林函数输入
    # model0.v_imp = 0.0
    hmtn0 = model0.get_Hamiltonian()
    n_orb = model0.n_orbit
    tag_imp = np.asarray((model0.n_x // 2, model0.n_y // 2))
    idx_imp = model0.lattice.tag2idx(tag_imp)
    idx_prb = model0.lattice.tag2idx(np.transpose(np.reshape(np.meshgrid(x_ls, y_ls), (2, -1))))

    # QPI输入
    atom_pos = np.kron([[0, 0], [0.5, 0], [0.5, -0.5], [0, -0.5]], np.ones((2, 1)))
    atom_r = 0.4
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    rez = 400
    x1_ls = np.linspace(*(tag_imp + (-50, 50)), rez)
    y1_ls = np.linspace(*(tag_imp + (-50, 50)), rez)

    def qpi_of_mu(bias):

        # 格林函数
        gfr = ih.calculate_Green_function(hmtn0, bias, +eps, r1_ls=idx_prb, r0_ls=[idx_imp], n_orb=n_orb)
        gf_r0 = np.reshape(gfr.toarray(), (len(idx_prb), n_orb, n_orb))
        gfa = ih.calculate_Green_function(hmtn0, bias, -eps, r1_ls=idx_prb, r0_ls=[idx_imp], n_orb=n_orb)
        gf_0r = np.reshape(gfa.toarray(), (len(idx_prb), n_orb, n_orb))  # G^A(r,0)
        gf_0r = np.transpose(gf_0r, (0, 2, 1)).conj()  # G^R(0,r)
        gf_00 = gf_r0[np.argwhere(idx_prb == idx_imp)[0][0]]

        # LDOS差值
        gf_diff_r = qpi.calculate_greenfunc_difference_from_impurity(imp_mtx, gf_00, gf_r0, gf_0r)
        ldos_diff_ls = -np.imag([np.diag(gf) for gf in gf_diff_r]) / np.pi  # DoS(r,o)
        ldos_diff_ls = np.reshape(ldos_diff_ls, (len(y_ls), len(x_ls), n_orb))

        # QPI
        ldos_0_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
            ldos_diff_ls * np.tile([1, +1], 4), gd, atom_pos, x_ls, y_ls, x1_ls, y1_ls
        )
        qpi_0_ls = qpi.calculate_qpi_from_ldos_2d(ldos_0_ls)
        qpi_x = [qpi_0_ls[rez // 2 - i, rez // 2 + i] for i in np.arange(rez // 2)]

        return qpi_x

    # scan mu
    para_ls = np.reshape(mu_ls, (-1, 1))
    if 200 < mp.cpu_count():
        n_cpu = mp.cpu_count()//2  # 并行
        print("Using mp.Pool(", n_cpu, ")...")
        with mp.Pool(n_cpu) as process_pool:
            app_ls = [process_pool.apply_async(qpi_of_mu, arg) for arg in para_ls]
            process_pool.close()
            res_ls = [app.get() for app in tqdm(app_ls)]
    else:
        res_ls = [qpi_of_mu(mu) for mu in tqdm(mu_ls)]  # 串行
    qpi_x_ls = np.asarray(res_ls)

    # 保存
    sio.savemat("./data/" + "QPI_x(E).mat", {"eng_ls": mu_ls, "qpi_x_ls": qpi_x_ls})


def main_qpi_mu_sdw_plot():
    fname = "./data/" + "QPI_x(E).mat"

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6, 5), dpi=90)
    #
    # 读取
    file = sio.loadmat(fname)
    eng_ls = file["eng_ls"][0]
    qpi_x_ls = file["qpi_x_ls"]
    # 子图
    plt.pcolormesh(np.linspace(0, 4, qpi_x_ls.shape[1]), eng_ls, np.abs(qpi_x_ls), clim=(0, 8), cmap="viridis")
    plt.xlim([0.5, 1.5])
    plt.xlabel(r"$q_x/\pi$")
    plt.ylabel(r"$E$")
    plt.colorbar()
    #
    plt.show()


def main_qpi_mu():
    model0 = Model_KV2Se2O_xy((200, 200))
    model0.set_parameter()
    # model0.m0 = -0.9
    # model0.t1 = 0.03
    # model0.mu = -np.sqrt(model0.m0[0] ** 2 + 4 * model0.t1**2)
    # model0.mu = -0.9

    mu_ls = np.linspace(-0.1, 0.1, 41)
    v_imp = 0.2
    eps = 5e-3
    x_ls = np.arange(-61, 62) + model0.n_x // 2
    y_ls = np.arange(-60, 61) + model0.n_y // 2

    # 杂质Hamiltonian
    imp_mtx = v_imp * np.diag(np.kron([1, 1], [1, 1]))  # ([1] * 2 + [0] * 6)
    # imp_mtx = model0.v_imp * np.kron(np.diag([1, 0]), np.kron(SIGMA_X, SIGMA_X))

    # 格林函数输入
    model0.v_imp = 0.0
    hmtn0 = model0.get_Hamiltonian()
    n_orb = model0.n_orbit
    tag_imp = np.asarray((model0.n_x // 2, model0.n_y // 2))
    idx_imp = model0.lattice.tag2idx(tag_imp)
    idx_prb = model0.lattice.tag2idx(np.transpose(np.reshape(np.meshgrid(x_ls, y_ls), (2, -1))))

    # QPI输入
    atom_pos = np.kron([[0, 0], [-0.5, 0.5]], np.ones((2, 1)))
    atom_r = 0.4
    gd = lambda x: gaussian_delta(x, sigma=atom_r)
    rez = 400
    x1_ls = np.linspace(*(tag_imp + (-50, 50)), rez)
    y1_ls = np.linspace(*(tag_imp + (-50, 50)), rez)

    def qpi_of_mu(bias):

        # 格林函数
        gfr = ih.calculate_Green_function(hmtn0, bias, +eps, r1_ls=idx_prb, r0_ls=[idx_imp], n_orb=n_orb)
        gf_r0 = np.reshape(gfr.toarray(), (len(idx_prb), n_orb, n_orb))
        gfa = ih.calculate_Green_function(hmtn0, bias, -eps, r1_ls=idx_prb, r0_ls=[idx_imp], n_orb=n_orb)
        gf_0r = np.reshape(gfa.toarray(), (len(idx_prb), n_orb, n_orb))  # G^A(r,0)
        gf_0r = np.transpose(gf_0r, (0, 2, 1)).conj()  # G^R(0,r)
        gf_00 = gf_r0[np.argwhere(idx_prb == idx_imp)[0][0]]

        # LDOS差值
        gf_diff_r = qpi.calculate_greenfunc_difference_from_impurity(imp_mtx, gf_00, gf_r0, gf_0r)
        ldos_diff_ls = -np.imag([np.diag(gf) for gf in gf_diff_r]) / np.pi  # DoS(r,o)
        ldos_diff_ls = np.reshape(ldos_diff_ls, (len(y_ls), len(x_ls), n_orb))

        # QPI
        ldos_0_ls = qpi.transform_ldos_orbit2position_in_Cartesian(
            ldos_diff_ls * np.tile([1, +1], n_orb // 2), gd, atom_pos, x_ls, y_ls, x1_ls, y1_ls
        )
        qpi_0_ls = qpi.calculate_qpi_from_ldos_2d(ldos_0_ls)
        qpi_x = [qpi_0_ls[rez // 2 + i, rez // 2] for i in np.arange(rez // 2)]

        return qpi_x

    # scan mu
    para_ls = np.reshape(mu_ls, (-1, 1))
    if 200 < mp.cpu_count():
        n_cpu = mp.cpu_count()  # 并行
        print("Using mp.Pool(", n_cpu, ")...")
        with mp.Pool(n_cpu) as process_pool:
            app_ls = [process_pool.apply_async(qpi_of_mu, arg) for arg in para_ls]
            process_pool.close()
            res_ls = [app.get() for app in tqdm(app_ls)]
    else:
        res_ls = [qpi_of_mu(mu) for mu in tqdm(mu_ls)]  # 串行
    qpi_x_ls = np.asarray(res_ls)

    # 保存
    sio.savemat("./data/" + "QPI_x(E).mat", {"eng_ls": mu_ls, "qpi_x_ls": qpi_x_ls})


def main_qpi_mu_plot():
    fname = "./data/" + "QPI_x(E).mat"

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6, 5), dpi=90)
    #
    # 读取
    file = sio.loadmat(fname)
    eng_ls = file["eng_ls"][0]
    qpi_x_ls = file["qpi_x_ls"]
    # 子图
    plt.pcolormesh(np.linspace(0, 4, qpi_x_ls.shape[1]), eng_ls, np.abs(qpi_x_ls), clim=(0, 4), cmap="viridis")
    plt.xlim([0.5, 1.5])
    plt.xlabel(r"$q_x/\pi$")
    plt.ylabel(r"$E$")
    plt.colorbar()
    #-
    plt.show()

def main_qpi_anisotropy_v1():
    param_text = ["Vi0.2,", "Vi(0.1,0.2,0.1,0.2),", "M0,Vi0.2,"][2]
    fname_ls = [
        "./data/QPI_KV2Se2O-SDW_(t1)0.03,"+param_text+"E-0.04.mat",
        "./data/QPI_KV2Se2O-SDW_(t1)0.03,"+param_text+"E-0.03.mat",
        "./data/QPI_KV2Se2O-SDW_(t1)0.03,"+param_text+"E-0.02.mat",
        "./data/QPI_KV2Se2O-SDW_(t1)0.03,"+param_text+"E-0.01.mat",
        "./data/QPI_KV2Se2O-SDW_(t1)0.03,"+param_text+"E0.01.mat",
        "./data/QPI_KV2Se2O-SDW_(t1)0.03,"+param_text+"E0.02.mat",
        "./data/QPI_KV2Se2O-SDW_(t1)0.03,"+param_text+"E0.03.mat",
        "./data/QPI_KV2Se2O-SDW_(t1)0.03,"+param_text+"E0.04.mat",
    ]
    eng_ls=[-40,-30,-20,-10,10,20,30,40]
    clim_q= (0,8)

    # 读取
    file = sio.loadmat(fname_ls[5])
    qpi_ls = np.abs(file["qpi_s_ls"]) if param_text == "Vi0.2," else np.abs(file["qpi_0_ls"])
    n_q = qpi_ls.shape[0]
    q_ls = np.linspace(-4, 4, n_q)
    q1_ls, q2_ls = np.meshgrid(q_ls, q_ls)
    mask_y = (q1_ls + q2_ls > 1.65) & (q1_ls + q2_ls < 2.35) & (q1_ls - q2_ls > -1.6) & (q1_ls - q2_ls < 1.6)
    mask_x = (q1_ls - q2_ls > 1.65) & (q1_ls - q2_ls < 2.35) & (q1_ls + q2_ls > -1.6) & (q1_ls + q2_ls < 1.6)
    qpi_x_lobe = np.where(mask_x, qpi_ls, np.nan)
    qpi_y_lobe = np.where(mask_y, qpi_ls, np.nan)

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6*3, 5+3), dpi=90)
    # 子图
    plt.subplot(131)
    plt.pcolormesh(q1_ls, q2_ls, qpi_ls, cmap="cividis", clim=clim_q)
    plt.axis("scaled")
    # 子图
    plt.subplot(132)
    plt.pcolormesh(q1_ls, q2_ls, qpi_x_lobe, cmap="cividis", clim=clim_q)
    plt.axis("scaled")
    # 子图
    plt.subplot(133)
    plt.pcolormesh(q1_ls, q2_ls, qpi_y_lobe, cmap="cividis", clim=clim_q)
    plt.axis("scaled")
    #
    plt.show()

    inten_x = np.nansum(qpi_x_lobe)
    inten_y = np.nansum(qpi_y_lobe)
    aniso = (inten_x - inten_y) / (inten_x + inten_y)
    print(aniso)

    def aniso_of_fname(fname):
        # 读取
        file = sio.loadmat(fname)
        qpi_ls = np.abs(file["qpi_s_ls"]) if param_text == "Vi0.2," else np.abs(file["qpi_s_ls"])

        qpi_x_lobe = np.where(mask_x, qpi_ls, np.nan)
        qpi_y_lobe = np.where(mask_y, qpi_ls, np.nan)
        inten_x = np.nansum(qpi_x_lobe)
        inten_y = np.nansum(qpi_y_lobe)
        aniso = (inten_x - inten_y) / (inten_x + inten_y)
        return aniso

    # scan E
    para_ls = np.reshape(fname_ls, (-1, 1))
    if 20 < mp.cpu_count():
        n_cpu = mp.cpu_count()  # 并行
        print("Using mp.Pool(", n_cpu, ")...")
        with mp.Pool(n_cpu) as process_pool:
            app_ls = [process_pool.apply_async(aniso_of_fname, arg) for arg in para_ls]
            process_pool.close()
            res_ls = [app.get() for app in tqdm(app_ls)]
    else:
        res_ls = [aniso_of_fname(fn) for fn in tqdm(para_ls[:,0])]  # 串行
    aniso_ls = np.asarray(res_ls)

    # 保存
    sio.savemat("./data/" + "QPI_AnIso(E).mat", {"eng_ls": eng_ls, "aniso_ls": aniso_ls})

def main_qpi_anisotropy():
    param_text = ["Vi0.2,", "Vi(0.1,0.2,0.1,0.2),", "M0,Vi0.2,"][2]
    fname_ls = [
        "./data/QPI_KV2Se2O-SDW_(t1)0.03," + param_text + "E-0.04.mat",
        "./data/QPI_KV2Se2O-SDW_(t1)0.03," + param_text + "E-0.03.mat",
        "./data/QPI_KV2Se2O-SDW_(t1)0.03," + param_text + "E-0.02.mat",
        "./data/QPI_KV2Se2O-SDW_(t1)0.03," + param_text + "E-0.01.mat",
        "./data/QPI_KV2Se2O-SDW_(t1)0.03," + param_text + "E0.01.mat",
        "./data/QPI_KV2Se2O-SDW_(t1)0.03," + param_text + "E0.02.mat",
        "./data/QPI_KV2Se2O-SDW_(t1)0.03," + param_text + "E0.03.mat",
        "./data/QPI_KV2Se2O-SDW_(t1)0.03," + param_text + "E0.04.mat",
    ]
    eng_ls = [-40, -30, -20, -10, 10, 20, 30, 40]
    clim_q = (0, 8)

    # 读取
    file = sio.loadmat(fname_ls[0])
    qpi_ls = np.abs(file["qpi_s_ls"]) if param_text == "Vi0.2," else np.abs(file["qpi_0_ls"])
    n_q = qpi_ls.shape[0]
    q_ls = np.linspace(-4, 4, n_q)
    q1_ls, q2_ls = np.meshgrid(q_ls, q_ls)
    mask_x = (q1_ls - q2_ls > 1.65) & (q1_ls - q2_ls < 2.35) & (q1_ls + q2_ls > 0.4) & (q1_ls + q2_ls < 1.6)
    mask_x = (q1_ls - q2_ls > 1.7) & (q1_ls - q2_ls < 2.3) & (q1_ls + q2_ls > 0.7) & (q1_ls + q2_ls < 1.3)
    qpi_x_lobe = np.where(mask_x, qpi_ls, np.nan)
    qpi_x_lobe_r90 = np.transpose(qpi_x_lobe)[:, np.arange(0, 0 - n_q, -1) % n_q]

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 3, 5 + 3), dpi=90)
    # 子图
    plt.subplot(131)
    plt.pcolormesh(q1_ls, q2_ls, qpi_ls, cmap="cividis", clim=clim_q)
    plt.axis("scaled")
    # 子图
    plt.subplot(132)
    plt.pcolormesh(q1_ls, q2_ls, qpi_x_lobe, cmap="cividis", clim=clim_q)
    plt.axis("scaled")
    # 子图
    plt.subplot(133)
    plt.pcolormesh(q1_ls, q2_ls, qpi_x_lobe_r90-qpi_ls, cmap="cividis", clim=clim_q)
    plt.axis("scaled")
    #
    plt.show()

    aniso = (qpi_x_lobe_r90 - qpi_ls) / (qpi_x_lobe_r90 + qpi_ls)
    aniso = np.nansum(aniso) / np.count_nonzero(~np.isnan(aniso))
    print(aniso)

    def aniso_of_fname(fname):
        # 读取
        file = sio.loadmat(fname)
        qpi_ls = np.abs(file["qpi_s_ls"]) if param_text == "Vi0.2," else np.abs(file["qpi_s_ls"])

        qpi_x_lobe = np.where(mask_x, qpi_ls, np.nan)
        qpi_x_lobe_r90 = np.transpose(qpi_x_lobe)[:, np.arange(0, 0 - n_q, -1) % n_q]
        # aniso = (qpi_x_lobe_r90 - qpi_ls) / (qpi_x_lobe_r90 + qpi_ls)
        # aniso = np.nansum(np.abs(aniso)) / np.count_nonzero(~np.isnan(aniso))
        diff = np.nansum(np.real((qpi_x_lobe_r90 - qpi_ls)))
        sum_ = np.nansum((qpi_x_lobe_r90 + qpi_ls))
        aniso = diff / sum_
        return aniso

    # scan E
    para_ls = np.reshape(fname_ls, (-1, 1))
    if 20 < mp.cpu_count():
        n_cpu = mp.cpu_count()  # 并行
        print("Using mp.Pool(", n_cpu, ")...")
        with mp.Pool(n_cpu) as process_pool:
            app_ls = [process_pool.apply_async(aniso_of_fname, arg) for arg in para_ls]
            process_pool.close()
            res_ls = [app.get() for app in tqdm(app_ls)]
    else:
        res_ls = [aniso_of_fname(fn) for fn in tqdm(para_ls[:, 0])]  # 串行
    aniso_ls = np.asarray(res_ls)

    # 保存
    sio.savemat("./data/" + "QPI_q-Rq(E).mat", {"eng_ls": eng_ls, "aniso_ls": aniso_ls})


def main_topography_diff():
    # model0 = Model_AM_xy((200, 200))
    # model0.t_hop = [1.0, -0.3]
    # model0.am = 1.6
    # model0 = Model_KV2Se2O_xy((200, 200))
    model0 = Model_KV2Se2O_SDW_xy_rev((201, 201))
    model0.set_parameter()
    # model0.m0[0] = -0.9
    # model0.m0[1] = -0.0
    # model0.t1 = 0.03
    model0.mu = -np.sqrt(model0.m0[0] ** 2 + 4 * model0.t1**2)
    model0.mu = -0.9

    eng_ls = np.linspace(0, 0.3, 301)
    eps = 5e-3
    n_orb = model0.n_orbit
    # opr = np.eye(n_orb)
    # opr = np.diag([1, 1] * (model0.n_orbit // 2))
    x_ls, y_ls = model0.lattice.tag_ls.T
    n_s = model0.lattice.n_site

    model0.v_imp = 0.0
    hmtn0 = model0.get_Hamiltonian()
    model0.v_imp = 0.4
    # hmtn1 = model0.get_Hamiltonian()

    def ldos_of_eng(eng):
        # 格林函数
        idx_imp = model0.lattice.tag2idx(model0.tag_imp)
        gfr = ih.calculate_Green_function(hmtn0, eng, +eps, r0_ls=[idx_imp], n_orb=n_orb)
        gf_r0 = np.reshape(gfr.toarray(), (n_s, n_orb, n_orb))
        gfa = ih.calculate_Green_function(hmtn0, eng, -eps, r0_ls=[idx_imp], n_orb=n_orb)
        gf_0r = np.reshape(gfa.toarray(), (n_s, n_orb, n_orb))
        gf_0r = np.transpose(gf_0r, (0, 2, 1)).conj()
        gf_00 = gf_r0[idx_imp]

        # 杂质Hamiltonian
        imp_mtx = model0.v_imp * np.diag(np.kron([0, 1, 0, 0], [1, 1]))  # ([1] * 2 + [0] * 6)
        # imp_mtx = model0.v_imp * np.kron(np.diag([1, 0]), np.kron(SIGMA_X, SIGMA_X))
        # LDOS差值
        gf_diff_r = qpi.calculate_greenfunc_difference_from_impurity(imp_mtx, gf_00, gf_r0, gf_0r)
        ldos_diff_ls = -np.imag([np.diag(gf) for gf in gf_diff_r]) / np.pi  # DoS(r,o)
        return ldos_diff_ls

    # scan bias
    if 40 < mp.cpu_count():
        n_cpu = mp.cpu_count()  # // 8  # 并行
        print("Using mp.Pool(", n_cpu, ")...")
        with mp.Pool(n_cpu) as process_pool:
            app_ls = [process_pool.apply_async(ldos_of_eng, arg) for arg in zip(eng_ls)]
            process_pool.close()
            res_ls = [app.get() for app in tqdm(app_ls)]
    else:
        res_ls = [ldos_of_eng(e_) for e_ in tqdm(eng_ls)]  # 串行
    ldos_diff_ls = np.average(res_ls, axis=0)

    # 保存
    x_ls, y_ls = [np.reshape(arr, (model0.n_y, model0.n_x)) for arr in [x_ls, y_ls]]
    ldos_diff_ls = np.reshape(ldos_diff_ls, (model0.n_y, model0.n_x, n_orb))
    sio.savemat("./data/" + "δLDoS(x,y).mat", {"x_ls": x_ls, "y_ls": y_ls, "ldos_diff_ls": ldos_diff_ls})


def main_topography_plot():
    fname_3 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi(0,0,2,0),EΣ0.3.mat"
    # fname_4 = "./data/" + "δLDoS(x,y)_KV2Se2O-SDW_(t1)0.03,Vi(0,0,0,2),EΣ0.3.mat"
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
    plt.figure(figsize=(6 * 2, 5 * 2), dpi=100).set_layout_engine("compressed")

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
    plt.subplot(221)
    plt.pcolormesh(x1_ls, y1_ls, ln_ldos_ls, cmap=cmap0, clim=None)
    plt.axis("scaled")
    plt.xlim(center + (-9, 9))
    plt.ylim(center + (-9, 9))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.colorbar()
    # signal_K
    gd_k = lambda x: gaussian_delta(x, sigma=atom_r_k)
    alt_k_ls = np.ones((*(ldos_diff_ls.shape[:2]), 1))
    pos_atom_k = np.asarray([[-0.25, 0.25]])
    alt_k_ls = qpi.transform_ldos_orbit2position_in_Cartesian(alt_k_ls, gd_k, pos_atom_k, x0_ls, y0_ls, x1_ls, y1_ls)
    alt_tot = alt_k_ls * (1 + weight_v * ln_ldos_ls)
    # cut line
    # pos0 = np.transpose([x_ls.flatten(), y_ls.flatten()])
    # pos0 = qpi.get_pos_atom(pos0, pos_atom)
    # pos_imp = center + pos_imp_
    # x_cut = np.linspace(0, 8, 81)
    # pos_c0 = np.transpose([+x_cut, -x_cut]) + pos_imp
    # pos_c1 = np.transpose([+x_cut, +x_cut]) + pos_imp
    # pos_c2 = np.transpose([-x_cut, +x_cut]) + pos_imp
    # pos_c3 = np.transpose([-x_cut, -x_cut]) + pos_imp
    # 子图
    plt.subplot(223)
    plt.pcolormesh(x1_ls, y1_ls, alt_tot, cmap=cmap0, clim=clim_d)
    plt.axis("scaled")
    plt.xlim(center + (-9, 9))
    plt.ylim(center + (-9, 9))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.colorbar()
    # 子图
    plt.subplot(222)
    plt.pcolormesh(x1_ls, y1_ls, np.real(ldos_0_ls), cmap=cmap0, clim=clim_d)
    plt.axis("scaled")
    plt.xlim(center + (-15, 15))
    plt.ylim(center + (-15, 15))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.colorbar()
    #
    oplt.number_subplots()
    plt.show()


""" ================================================================================
    Not Main
================================================================================ """


def convolve_matrix_2d(mtx0, func_cvv, x0_ls, y0_ls, x1_ls, y1_ls):
    """
    二维空间矩阵变换
    """
    # mtx0 = np.asarray(mtx0).flatten()

    x1_ls, x0_ls = np.meshgrid(x1_ls, x0_ls, indexing="ij")
    cvv_x = func_cvv(x1_ls - x0_ls)
    y1_ls, y0_ls = np.meshgrid(y1_ls, y0_ls, indexing="ij")
    cvv_y = func_cvv(y1_ls - y0_ls)

    mtx1 = cvv_y @ mtx0 @ cvv_x.T
    # mtx1 = np.reshape(mtx1, (len(y1_ls), len(x1_ls)))
    return mtx1


def gaussian_delta(x, mu=0, sigma=1):
    """
    高斯δ函数
    """
    delta = np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    return delta


""" ================================================================================
    RUN
================================================================================ """
if __name__ == "__main__":
    # main_ldos_diff_x_y_plot_uc_at()
    # main_ldos_diff_x_y_imp1uc()
    # main_ldos_diff_x_y()
    # main_ldos_diff_x_y_plot_0z()
    # main_ldos_diff_x_y_plot_0z_rev()
    # main_full_ldos_x_y_plot_0z_rev()
    #
    # main_qpi_mu_sdw()
    # main_qpi_mu_sdw_plot()
    # main_qpi_mu()
    # main_qpi_mu_plot()
    #
    main_qpi_anisotropy()
    #
    # main_topography_diff()
    # main_topography_plot()
    #
