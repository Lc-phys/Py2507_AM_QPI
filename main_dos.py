from copy import copy
import numpy as np
from scipy import io as sio
from pathos import multiprocessing as mp
from tqdm import tqdm
from matplotlib import pyplot as plt

from model.AlterMag import *
from task import inverse_Hamiltonian as ih
from task import origin_style_plot as oplt


def main_dispersion():
    model0 = Model_KV2Se2O_kk()
    model0 = Model_KV2Se2O_SDW_kk()
    model0.set_parameter()
    model0.m0[1] = -0.02  # [-0.9, -0.02]  # [0]
    # model0.t1 = 0.03
    model0.mu = -0.9

    # disp
    # opr = np.kron(np.diag([1,1,0,0]),SIGMA_0)
    opr = np.kron(np.eye(model0.n_orbit // 2), SIGMA_Z)
    # opr = np.kron(SIGMA_Z, SIGMA_00)

    # (ks,kx,ky)
    rez = 401
    k_path_GX = np.linspace((0, 0, 0), (np.pi, np.pi, 0), rez)
    k_path_XM = np.linspace((np.pi, np.pi, 0), (np.pi * 2, np.pi, np.pi), rez)[1:]
    k_path_MG = np.linspace((np.pi * 2, np.pi, np.pi), (np.pi * (2 + np.sqrt(2)), 0, 0), round(rez * np.sqrt(2)))[1:]
    # k_path_GX = np.linspace((0, 0, 0), (np.pi, -np.pi, 0), rez)
    # k_path_XM = np.linspace((np.pi, -np.pi, 0), (np.pi * 2, -np.pi, np.pi), rez)[1:]
    # k_path_MG = np.linspace((np.pi * 2, -np.pi, np.pi), (np.pi * (2 + np.sqrt(2)), 0, 0), round(rez * np.sqrt(2)))[1:]
    k_path = np.concatenate((k_path_GX, k_path_XM, k_path_MG))
    #
    # k_path_MY = np.linspace((0, np.pi, np.pi), (np.pi, 0, np.pi), rez)
    # k_path_YG = np.linspace((np.pi, 0, np.pi), (np.pi * 2, 0, 0), rez)[1:]
    # k_path_GX = np.linspace((np.pi * 2, 0, 0), (np.pi * 3, np.pi, 0), rez)[1:]
    # k_path_XM = np.linspace((np.pi * 3, np.pi, 0), (np.pi * 4, np.pi, np.pi), rez)[1:]
    # k_path_MG = np.linspace((np.pi * 4, np.pi, np.pi), (np.pi * (4 + np.sqrt(2)), 0, 0), round(rez * np.sqrt(2)))[1:]
    # k_path = np.concatenate((k_path_MY, k_path_YG, k_path_GX, k_path_XM, k_path_MG))
    #
    # k_path_GX = np.linspace((0, 0, 0), (np.pi, np.pi, 0), rez)
    # k_path_XM = np.linspace((np.pi, np.pi, 0), (np.pi * 2, np.pi, np.pi), rez)[1:]
    # k_path_MY = np.linspace((np.pi * 2, np.pi, np.pi), (np.pi * 3, 0, np.pi), rez)[1:]
    # k_path_YG = np.linspace((np.pi * 3, 0, np.pi), (np.pi * 4, 0, 0), rez)[1:]
    # k_path_GM = np.linspace((np.pi * 4, 0, 0), (np.pi * (4 + np.sqrt(2)), np.pi, np.pi), round(rez * np.sqrt(2)))[1:]
    # k_path = np.concatenate((k_path_GX, k_path_XM, k_path_MY, k_path_YG, k_path_GM))
    #
    # k_path_XG = np.linspace((0, -np.pi, 0), (np.pi, 0, 0), rez)
    # k_path_GX = np.linspace((np.pi, 0, 0), (np.pi * 2, np.pi, 0), rez)[1:]
    # k_path = np.concatenate((k_path_XG, k_path_GX))

    # E(k)
    def eng_of_k(k_):
        kx, ky = k_
        k_ = kx + ky, -kx + ky
        hmtn = model0.get_Hamiltonian(k_)

        eng, wf = np.linalg.eigh(hmtn)
        expt = np.real(np.diag(wf.conj().T @ opr @ wf))
        return eng, expt

    eng_ls, spin_ls = np.transpose([eng_of_k(k_) for k_ in tqdm(k_path[:, 1:])], (1, 0, 2))

    # 保存
    sio.savemat("./data/E(k).mat", {"k_path": k_path, "eng_ls": eng_ls, "spin_ls": spin_ls})


def main_dispersion_plot():
    fname = "./data/E(k).mat"

    oplt.set_rc_origin()
    plt.figure(figsize=(6, 5), dpi=120)

    # 读取
    file = sio.loadmat(fname)
    k_sym_ls, eng_ls, spin_ls = file["k_path"][:, 0], file["eng_ls"] * 1e3, file["spin_ls"]
    # 作图
    for ib in range(eng_ls.shape[1]):
        oplt.colored_line(k_sym_ls[:], eng_ls[:, ib], spin_ls[:, ib], plt.gca(), cmap="coolwarm", clim=(-1, 1))
    # plt.plot(k_sym_ls[:], eng_ls[:], c="0.3")
    plt.axhline(0, ls=":", lw=1, color="tab:gray")
    plt.yticks([-400, 0, 400])
    plt.axvline(np.pi, color="k")
    plt.axvline(np.pi * 2, color="k")
    plt.xticks([0, np.pi, np.pi * 2, k_sym_ls[-1]], labels=[r"$\Gamma$", "X", "M", r"$\Gamma$"])
    # plt.axvline(np.pi * 3, color="k")
    # plt.axvline(np.pi * 4, color="k")
    # plt.xticks(
    #     [0, np.pi, np.pi * 2, np.pi * 3, np.pi * 4, k_sym_ls[-1]],
    #     labels=[r"$\Gamma$", "X", "M", "Y", r"$\Gamma$", "M"],
    # )
    plt.ylim(-700, 700)
    plt.ylabel(r"$E$ (meV)")
    #
    plt.show()


def main_dispersion_diff_plot():
    fname_0 = "./data/E(k)_τzσz.mat"
    fname_9 = "./data/E(k)_τzσ0.mat"
    fname_0 = "./data/E(k)_M.mat"
    fname_9 = "./data/E(k)_M'.mat"

    oplt.set_rc_origin()
    plt.figure(figsize=(6, 5), dpi=120)

    # 读取
    file = sio.loadmat(fname_0)
    k_sym_ls0, eng_ls0, spin_ls0 = file["k_path"][:, 0], file["eng_ls"], file["spin_ls"]
    file = sio.loadmat(fname_9)
    k_sym_ls9, eng_ls9, spin_ls9 = file["k_path"][:, 0], file["eng_ls"], file["spin_ls"]
    # 作图
    # for ib in range(eng_ls0.shape[1]):
    #     oplt.colored_line(
    #         k_sym_ls0[:],
    #         eng_ls0[:, ib] - eng_ls9[:, ib],
    #         spin_ls0[:, ib],
    #         plt.gca(),
    #         cmap="coolwarm",
    #         clim=(-1, 1)
    #     )
    plt.plot(k_sym_ls0[:], eng_ls0[:, :4] - eng_ls9[:, :4], c="0.3")
    plt.axhline(0, ls=":", lw=1, color="tab:gray")
    # plt.yticks([-0.4, 0, 0.4])
    plt.axvline(np.pi, color="k")
    plt.axvline(np.pi * 2, color="k")
    plt.xticks([0, np.pi, np.pi * 2, k_sym_ls0[-1]], labels=[r"$\Gamma$", "X", "M", r"$\Gamma$"])
    # plt.axvline(np.pi * 3, color="k")
    # plt.axvline(np.pi * 4, color="k")
    # plt.xticks(
    #     [0, np.pi, np.pi * 2, np.pi * 3, np.pi * 4, k_sym_ls0[-1]],
    #     labels=[r"$\Gamma$", "X", "M", "Y", r"$\Gamma$", "M"],
    # )
    # plt.ylim(-0.7, 0.7)
    plt.ylabel(r"$E$ (meV)")
    #
    plt.show()


def main_dispersion_Gf():
    # model0 = Model_KV2Se2O_kk()
    model0 = Model_KV2Se2O_SDW_kk()
    # model0 = Model_KV2Se2O_dxy_kk()
    model0.set_parameter()
    # model0.t2 = [0.29,0.17]
    # model0.t3=-0.03
    model0.m0[0] = -0.
    model0.m0[1] = -0.2
    # model0.t1 = -0.0
    model0.mu = -np.sqrt(model0.m0[0] ** 2 + 4 * model0.t1**2)

    # disp
    eng_ls = np.linspace(-2, 2, 201)
    eps = 5e-3
    opr = np.diag([1, -1] * (model0.n_orbit // 2))
    # opr = np.eye(model0.n_orbit)

    # (ks,kx,ky)
    rez = 401
    # k_path_GX = np.linspace((0, 0, 0), (np.pi, np.pi, 0), rez)
    # k_path_XM = np.linspace((np.pi, np.pi, 0), (np.pi * 2, np.pi, np.pi), rez)[1:]
    # k_path_MG = np.linspace((np.pi * 2, np.pi, np.pi), (np.pi * (2 + np.sqrt(2)), 0, 0), round(rez * np.sqrt(2)))[1:]
    # k_path = np.concatenate((k_path_GX, k_path_XM, k_path_MG))
    k_path_MY = np.linspace((0, np.pi, np.pi), (np.pi, 0, np.pi), rez)
    k_path_YG = np.linspace((np.pi, 0, np.pi), (np.pi * 2, 0, 0), rez)[1:]
    k_path_GX = np.linspace((np.pi * 2, 0, 0), (np.pi * 3, np.pi, 0), rez)[1:]
    k_path_XM = np.linspace((np.pi * 3, np.pi, 0), (np.pi * 4, np.pi, np.pi), rez)[1:]
    k_path_MG = np.linspace((np.pi * 4, np.pi, np.pi), (np.pi * (4 + np.sqrt(2)), 0, 0), round(rez * np.sqrt(2)))[1:]
    k_path = np.concatenate((k_path_MY, k_path_YG, k_path_GX, k_path_XM, k_path_MG))

    # DoS(E,k)
    def dos_of_eng_of_k(k_):
        model1 = copy(model0)
        kx, ky = k_
        k_ = kx + ky, -kx + ky
        hmtn = model1.get_Hamiltonian(k_)

        def dos_of_eng(eng):
            gfunc = ih.calculate_Green_function(hmtn, eng, eps)
            dos = -np.imag(np.trace(opr @ gfunc)) / np.pi
            return dos

        dos_ls = np.asarray([dos_of_eng(eng) for eng in eng_ls])
        return dos_ls

    dos_ls = np.asarray([dos_of_eng_of_k(k_) for k_ in tqdm(k_path[:, 1:])]).T
    # eng_ls, k_sym_ls = np.meshgrid(eng_ls, k_path[:, 0], indexing="ij")

    # 保存
    sio.savemat("./data/Disp.mat", {"k_path": k_path, "eng_ls": eng_ls, "dos_ls": dos_ls})


def main_dispersion_Gf_plot():
    fname = "./data/Disp.mat"

    # 读取
    file = sio.loadmat(fname)
    eng_ls, k_sym_ls, dos_ls = file["eng_ls"][0], file["k_path"][:, 0], file["dos_ls"]

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6, 4), dpi=120)
    #
    plt.pcolormesh(k_sym_ls, eng_ls, dos_ls, cmap="Spectral_r", clim=(-4, 4))
    plt.axvline(np.pi, color="k")
    plt.axvline(np.pi * 2, color="k")
    plt.xticks([0, np.pi, np.pi * 2, k_sym_ls[-1]], labels=[r"$\Gamma$", "X", "M", r"$\Gamma$"])
    plt.axvline(np.pi * 3, color="k")
    plt.axvline(np.pi * 4, color="k")
    plt.xticks(
        [0, np.pi, np.pi * 2, np.pi * 3, np.pi * 4, k_sym_ls[-1]], labels=["M", "Y", r"$\Gamma$", "X", "M", r"$\Gamma$"]
    )
    plt.ylabel(r"$E$")
    #
    plt.show()


def main_Fermi_surface():
    model0 = Model_KV2Se2O_kk()
    model0 = Model_KV2Se2O_SDW_kk()
    # model0 = Model_KV2Se2O_dxy_kk()
    model0.set_parameter()
    # model0.m0[0]*=-0
    # model0.m0[1] = -0.0
    # model0.t3 += 0.000
    # model0.t2[1] *= 1.2
    # model0.t1 = 0.03
    # model0.mu = -np.sqrt(model0.m0[0] ** 2 + 4 * model0.t1**2)
    model0.mu = -0.9
    # model0.so=0.1

    # FSurf
    mu = 0.05  # +0.000 * 4
    eps = 5e-3
    # opr = np.kron([1, 1, +1, +1], [1, -1])

    # (kx,ky)
    rez = 401
    kx_ls, ky_ls = np.linspace(-np.pi, np.pi, rez)[:-1], np.linspace(-np.pi, np.pi, rez)[:-1]
    kx_ls, ky_ls = np.meshgrid(kx_ls, ky_ls)
    k_ls = np.vstack((kx_ls.flatten(), ky_ls.flatten())).T

    # DoS(k)
    def dos_of_k(k_):
        model1 = copy(model0)
        kx, ky = k_
        k_ = kx + ky, -kx + ky
        # k_ = kx + ky, -kx + ky, 0
        # k_ = kx + np.pi, -kx + np.pi, ky
        hmtn = model1.get_Hamiltonian(k_)
        gfunc = ih.calculate_Green_function(hmtn, mu, eps)
        dos = -np.imag(np.diag(gfunc)) / np.pi
        return dos

    dos_ls = np.reshape([dos_of_k(k_) for k_ in tqdm(k_ls)], (*kx_ls.shape,-1))

    # 保存
    sio.savemat("./data/FmSurf.mat", {"kx_ls": kx_ls, "ky_ls": ky_ls, "dos_ls": dos_ls})


def main_Fermi_surface_plot():
    fname = "./data/FmSurf.mat"

    cmap = "RdBu"

    # 读取
    file = sio.loadmat(fname)
    kx_ls, ky_ls, dos_ls = file["kx_ls"], file["ky_ls"], file["dos_ls"]
    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6*2, 5), dpi=120).set_layout_engine("compressed")
    #
    plt.subplot(121)
    plt.pcolormesh(kx_ls / np.pi, ky_ls / np.pi, np.sum(dos_ls, axis=-1), cmap="viridis", clim=(0, 16))
    plt.axis("scaled")
    plt.yticks([-1, 0, 1])
    # plt.xticks([])
    # plt.yticks([])
    plt.xlabel(r"$k_x/\pi$")
    plt.ylabel(r"$k_y/\pi$")
    plt.colorbar()
    #
    plt.subplot(122)
    wt=np.kron([1, 1, 1, 1], [1, -1])
    plt.pcolormesh(kx_ls / np.pi, ky_ls / np.pi, np.sum(dos_ls*wt, axis=-1), cmap="RdBu", clim=(-8, 8))
    plt.axis("scaled")
    plt.yticks([-1, 0, 1])
    # plt.xticks([])
    # plt.yticks([])
    plt.xlabel(r"$k_x/\pi$")
    plt.ylabel(r"$k_y/\pi$")
    plt.colorbar()
    #
    plt.show()

    dos0 = np.average(dos_ls, axis=(0,1))
    print(np.sum(dos0))
    sio.savemat("./data/DoS0.mat", {"dos0": dos0})


def main_Fermi_surface_diff_plot():
    fname_0 = "./data/FmSurf_τzσz.mat"
    fname_9 = "./data/FmSurf_τzσ0.mat"

    cmap = "RdBu"

    # 读取
    file = sio.loadmat(fname_0)
    kx_ls0, ky_ls0, dos_ls0 = file["kx_ls"], file["ky_ls"], file["dos_ls"]
    file = sio.loadmat(fname_9)
    kx_ls9, ky_ls9, dos_ls9 = file["kx_ls"], file["ky_ls"], file["dos_ls"]
    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6, 5), dpi=120).set_layout_engine("compressed")
    #
    plt.pcolormesh(kx_ls0 / np.pi, ky_ls0 / np.pi, dos_ls0 - dos_ls9, cmap=cmap)
    plt.axis("scaled")
    plt.yticks([-1, 0, 1])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(r"$k_x/\pi$")
    plt.ylabel(r"$k_y/\pi$")
    plt.colorbar()
    #
    plt.show()


def main_Fermi_surface_parity():
    model0 = Model_KV2Se2O_kk()
    model0 = Model_KV2Se2O_SDW_kkk()
    # model0 = Model_KV2Se2O_dxy_kk()
    model0.set_parameter()
    model0.m0[0] =0.3
    # model0.m0[1] = -0.6
    # model0.t3 += 0.000
    # model0.t2[1] *= 1.2
    # model0.t1 = 0.03
    # model0.mu = -np.sqrt(model0.m0[0] ** 2 + 4 * model0.t1**2)
    model0.mu = -0.3
    model0.so=0.4
    model0.tz=0.1

    # FSurf
    mu = 0.07  # +0.000 * 4
    eps = 5e-3
    opr = np.kron([1, 1, +1, +1], [1, -1])

    # (kx,ky)
    rez = 401
    kx_ls, ky_ls = np.linspace(-np.pi, np.pi, rez)[:-1], np.linspace(-np.pi, np.pi, rez)[:-1]
    kx_ls, ky_ls = np.meshgrid(kx_ls, ky_ls)
    k_ls = np.vstack((kx_ls.flatten(), ky_ls.flatten())).T

    # DoS(k)
    def dos_of_k(k_):
        model1 = copy(model0)
        kx, ky = k_
        k_ = kx + ky, -kx + ky
        k_ = kx + np.pi/2, -kx + np.pi/2, ky
        hmtn = model1.get_Hamiltonian(k_)
        gfunc = ih.calculate_Green_function(hmtn, mu, eps)
        dos = -np.imag(np.trace(opr * gfunc)) / np.pi
        # ρ(-k)
        k_m = -kx - np.pi / 2, kx - np.pi / 2, -ky
        # k_m = -np.array(k_)
        hmtn_m = model1.get_Hamiltonian(k_m)
        gfunc_m = ih.calculate_Green_function(hmtn_m, mu, eps)
        dos_m = -np.imag(np.trace(opr * gfunc_m)) / np.pi
        return dos, dos_m

    dos_ls, dos_ls_mk = zip(*[dos_of_k(k_) for k_ in tqdm(k_ls)])
    dos_ls, dos_ls_mk = [np.reshape(arr, kx_ls.shape) for arr in [dos_ls, dos_ls_mk]]

    # 保存
    sio.savemat("./data/FmSurf_P.mat", {"kx_ls": kx_ls, "ky_ls": ky_ls, "dos_ls": dos_ls, "dos_ls_mk":dos_ls_mk})


def main_Fermi_surface_parity_plot():
    fname = "./data/FmSurf_P.mat"

    cmap = "RdBu_r"

    # 图
    oplt.set_rc_origin()
    plt.figure(figsize=(6*1, 5*2), dpi=120).set_layout_engine("compressed")
    #
    plt.subplot(211)
    # 读取
    file = sio.loadmat(fname)
    kx_ls, ky_ls, dos_ls, dos_ls_mk = file["kx_ls"], file["ky_ls"], file["dos_ls"], file["dos_ls_mk"]
    dos_pm = dos_ls * dos_ls_mk
    # 作图
    plt.pcolormesh(kx_ls / np.pi, ky_ls / np.pi, dos_pm, cmap=cmap, clim=(-64, 64))
    plt.axis("scaled")
    plt.yticks([-1, 0, 1])
    # plt.xticks([])
    # plt.yticks([])
    plt.xlabel(r"$k_x/\pi$")
    plt.ylabel(r"$k_y/\pi$")
    plt.colorbar()
    #
    plt.subplot(212)
    #
    dos_pp = dos_ls **2
    eta = np.sum(dos_pm) / np.sum(dos_pp)
    print("η = ", eta)
    print("2|ρ_o|^2 = ", np.sum(dos_pp) - np.sum(dos_pm))
    # 作图
    plt.pcolormesh(kx_ls / np.pi, ky_ls / np.pi, dos_pp, cmap=cmap, clim=(-64, 64))
    plt.axis("scaled")
    plt.yticks([-1, 0, 1])
    # plt.xticks([])
    # plt.yticks([])
    plt.xlabel(r"$k_x/\pi$")
    plt.ylabel(r"$k_y/\pi$")
    plt.colorbar()
    #
    oplt.number_subplots()
    plt.show()


def main_dos_Gf():
    # model0 = Model_KV2Se2O_kk()
    model0 = Model_KV2Se2O_SDW_kk()
    model0.set_parameter()
    # model0.t2 = [-0.29,0.1]
    # model0.t3=0
    model0.m0[1] = -0.02
    # model0.t1 = 0.0
    model0.mu = -np.sqrt(model0.m0[0] ** 2 + 4 * model0.t1**2)
    model0.mu = -0.9

    #
    eng_ls = np.linspace(-0.1, 0.5, 61)
    eps = 5e-3
    opr = np.diag([1, +1] * (model0.n_orbit // 2))
    # opr = np.eye(model0.n_orbit)

    # (kx,ky)
    rez = 401
    kx_ls, ky_ls = np.linspace(-np.pi, np.pi, rez)[:-1], np.linspace(-np.pi, np.pi, rez)[:-1]
    kx_ls, ky_ls = np.meshgrid(kx_ls, ky_ls)
    k_ls = np.vstack((kx_ls.flatten(), ky_ls.flatten())).T

    # DoS(E,k)
    def dos_of_eng_of_k(k_):
        model1 = copy(model0)
        kx, ky = k_
        k_ = kx + ky, -kx + ky
        hmtn = model1.get_Hamiltonian(k_)

        def dos_of_eng(eng):
            gfunc = ih.calculate_Green_function(hmtn, eng, eps)
            dos = -np.imag(np.diag(gfunc)) / np.pi
            return dos

        dos_ls = np.asarray([dos_of_eng(eng) for eng in eng_ls])
        return dos_ls

    # scan k
    if 20 < mp.cpu_count():
        n_task = mp.cpu_count()
        print("Using mp.Pool(", n_task, ")...")
        with mp.Pool(n_task) as process_pool:
            app_ls = [process_pool.apply_async(dos_of_eng_of_k, arg) for arg in zip(k_ls)]
            process_pool.close()
            res_ls = np.asarray([app.get() for app in tqdm(app_ls)])
    else:
        res_ls = np.asarray([dos_of_eng_of_k(k_) for k_ in tqdm(k_ls)])
    dos_ls = np.average(res_ls, axis=0)

    # 保存
    sio.savemat("./data/DoS.mat", {"eng_ls": eng_ls, "dos_ls": dos_ls})


def main_dos_Gf_plot():
    fname = "./data/DoS.mat"

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 2, 5 * 2), dpi=120)
    #
    # 读取
    file = sio.loadmat(fname)
    eng_ls, dos_ls = file["eng_ls"][0], file["dos_ls"]  # [0]
    #
    plt.subplot(221)
    plt.plot(eng_ls, dos_ls[:, 0])
    plt.plot(eng_ls, dos_ls[:, 3])
    plt.plot(eng_ls, dos_ls[:, 4])
    plt.plot(eng_ls, dos_ls[:, 7])
    # plt.ylim(bottom=0)
    plt.xlabel(r"$E$")
    plt.ylabel(r"DoS")
    #
    #
    plt.subplot(222)
    plt.plot(eng_ls, dos_ls[:, 1])
    plt.plot(eng_ls, dos_ls[:, 2])
    plt.plot(eng_ls, dos_ls[:, 5])
    plt.plot(eng_ls, dos_ls[:, 6])
    # plt.ylim(bottom=0)
    plt.xlabel(r"$E$")
    plt.ylabel(r"DoS")
    #
    plt.subplot(223)
    plt.plot(eng_ls, dos_ls[:, 0] - dos_ls[:, 4])
    # plt.ylim(bottom=0)
    plt.xlabel(r"$E$")
    plt.ylabel(r"DoS")
    #
    plt.subplot(224)
    plt.plot(eng_ls, np.sum(dos_ls, axis=-1))
    # plt.ylim(bottom=0)
    plt.xlabel(r"$E$")
    plt.ylabel(r"DoS")
    #
    plt.show()


def main_ldos_Gf():
    # model0 = Model_KV2Se2O_xy((200, 200))
    model0 = Model_KV2Se2O_SDW_xy((200, 200))
    model0 = Model_KV2Se2O_SDW_xy_itf((200, 200))
    model0 = Model_KV2Se2O_SDW_xy_rev((201, 201))
    model0.set_parameter()
    # model0.m0[1] = -0.02
    # model0.t1 = 0.0
    model0.mu = -np.sqrt(model0.m0[0] ** 2 + 4 * model0.t1**2)
    model0.mu = -0.9

    # model0.v_imp=-20e-3

    #
    eng_ls = np.linspace(-0.1, 0.5, 61)  # np.linspace(-7.4, 4.2, 116 * 2 + 1)
    eps = 5e-3
    # opr = np.eye(model0.n_orbit)

    #
    hmtn = model0.get_Hamiltonian()
    idx_p = model0.lattice.tag2idx((model0.n_x // 2, model0.n_y // 2))

    # DoS(E)
    def dos_of_eng(eng):
        gfunc = ih.calculate_Green_function(
            hmtn, eng, eps, r1_ls=[idx_p], r0_ls=[idx_p], n_orb=model0.n_orbit
        ).toarray()
        dos = -np.imag(np.diag(gfunc)) / np.pi
        return dos

    # scan k
    if 40 < mp.cpu_count():
        n_task = mp.cpu_count()
        print("Using mp.Pool(", n_task, ")...")
        with mp.Pool(n_task) as process_pool:
            app_ls = [process_pool.apply_async(dos_of_eng, arg) for arg in zip(eng_ls)]
            process_pool.close()
            res_ls = np.asarray([app.get() for app in tqdm(app_ls)])
    else:
        res_ls = np.asarray([dos_of_eng(e_) for e_ in tqdm(eng_ls)])
    dos_ls = res_ls

    # 保存
    sio.savemat("./data/LDoS.mat", {"eng_ls": eng_ls, "dos_ls": dos_ls})


def main_ldos_Gf_plot():
    fname = "./data/LDoS_KV2Se2O-SDW_SDW-0.02.mat"

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6, 5), dpi=120)
    #
    # 读取
    file = sio.loadmat(fname)
    eng_ls, dos_ls = file["eng_ls"][0], file["dos_ls"][0]
    plt.plot(eng_ls, dos_ls)
    # plt.ylim(bottom=0)
    plt.xlabel(r"$E$")
    plt.ylabel(r"DoS")
    #
    plt.show()


def main_ldos_Gf_imp():
    # model0 = Model_KV2Se2O_xy((200, 200))
    model0 = Model_KV2Se2O_SDW_xy((200, 200))
    model0.set_parameter()
    # model0.m0[1] = -0.02
    # model0.t1 = 0.0
    # model0.mu = -np.sqrt(model0.m0[0] ** 2 + 4 * model0.t1**2)

    # model0.v_imp = -20e-3

    #
    v_imp_ls = np.arange(-100e-3, 101e-3, 20e-3)
    eng_ls = np.linspace(-0.05, 0.05, 101)  # np.linspace(-7.4, 4.2, 116 * 2 + 1)
    eps = 1e-3
    # opr = np.eye(model0.n_orbit)

    #
    idx_p = model0.lattice.tag2idx((model0.n_x // 2, model0.n_y // 2))

    def dos_of_imp(v_):
        model1 = copy(model0)
        model1.v_imp = v_
        hmtn = model1.get_Hamiltonian()

        # DoS(E)
        def dos_of_eng(eng):
            gfunc = ih.calculate_Green_function(
                hmtn, eng, eps, r1_ls=[idx_p], r0_ls=[idx_p], n_orb=model1.n_orbit
            ).toarray()
            dos = -np.imag(np.trace(gfunc)) / np.pi
            return dos

        # scan k
        if 200 < mp.cpu_count():
            n_task = mp.cpu_count()
            print("Using mp.Pool(", n_task, ")...")
            with mp.Pool(n_task) as process_pool:
                app_ls = [process_pool.apply_async(dos_of_eng, arg) for arg in zip(eng_ls)]
                process_pool.close()
                res_ls = np.asarray([app.get() for app in (app_ls)])
        else:
            res_ls = np.asarray([dos_of_eng(e_) for e_ in (eng_ls)])
        dos_ls = res_ls
        return dos_ls

    dos_ls_imp = np.asarray([dos_of_imp(v_) for v_ in tqdm(v_imp_ls)])

    # 保存
    sio.savemat("./data/LDoS(V_imp).mat", {"v_imp_ls": v_imp_ls, "eng_ls": eng_ls, "dos_ls": dos_ls_imp})


def main_ldos_Gf_imp_plot():
    fname = "./data/" + "LDoS(V_imp)_KV2Se2O-SDW-xy_ε0.5e-3.mat"

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6, 5), dpi=120)
    #
    # 读取
    file = sio.loadmat(fname)
    v_imp_ls, eng_ls, dos_ls_imp = file["v_imp_ls"][0], file["eng_ls"][0], file["dos_ls"]
    # 子图
    plt.pcolormesh(eng_ls, v_imp_ls, dos_ls_imp)
    plt.xlabel(r"$E$")
    plt.ylabel(r"$V_{imp}$")
    plt.colorbar()
    #
    plt.show()


def main_ldos_imp_x_y():
    model0 = Model_AM_xy((100, 100))
    model0.am = 0.2

    mu = -1
    eps = 2e-2
    opr = np.diag([1, 1])
    x_ls, y_ls = model0.lattice.tag_ls.T
    iter_ls = np.arange(model0.lattice.n_site)

    hmtn0 = model0.get_Hamiltonian()
    model0.v_imp = 1.0
    hmtn1 = model0.get_Hamiltonian()

    def ldos_of_idx(idx_p):
        gfunc = ih.calculate_Green_function(
            hmtn0, mu, eps, r1_ls=[idx_p], r0_ls=[idx_p], n_orb=model0.n_orbit
        ).toarray()
        dos0 = -np.imag(np.trace(opr @ gfunc)) / np.pi
        gfunc = ih.calculate_Green_function(
            hmtn1, mu, eps, r1_ls=[idx_p], r0_ls=[idx_p], n_orb=model0.n_orbit
        ).toarray()
        dos1 = -np.imag(np.trace(opr @ gfunc)) / np.pi
        return dos0, dos1

    # scan r
    if 20 < mp.cpu_count():
        n_cpu = mp.cpu_count()
        print("Using mp.Pool(", n_cpu, ")...")
        with mp.Pool(n_cpu) as process_pool:
            app_ls = [process_pool.apply_async(ldos_of_idx, arg) for arg in np.reshape(iter_ls, (-1, 1))]
            process_pool.close()
            res_ls = np.transpose([app.get() for app in tqdm(app_ls)])
    else:
        res_ls = np.transpose([ldos_of_idx(ip) for ip in tqdm(iter_ls)])
    ldos0_ls, ldos1_ls = res_ls
    x_ls, y_ls, ldos0_ls, ldos1_ls = [
        np.reshape(arr, (model0.n_x, model0.n_y)) for arr in [x_ls, y_ls, ldos0_ls, ldos1_ls]
    ]

    # 保存
    sio.savemat(
        "./data/" + "LDoS_imp(x,y)_" + model0.name + ".mat",
        {"x_ls": x_ls, "y_ls": y_ls, "ldos0_ls": ldos0_ls, "ldos1_ls": ldos1_ls},
    )


def main_ldos_imp_x_y_plot():
    fname = "./data/" + "LDoS_imp(x,y)_AM-xy_t2(0.7),M0.2,Vi1,μ-1.mat"

    # 作图
    oplt.set_rc_origin()
    plt.figure(figsize=(6 * 2, 5 * 2), dpi=120).set_layout_engine("compressed")

    # 读取
    file = sio.loadmat(fname)
    x_ls, y_ls, ldos0_ls, ldos1_ls = file["x_ls"], file["y_ls"], file["ldos0_ls"], file["ldos1_ls"]
    ldos_diff_ls = ldos1_ls - ldos0_ls
    qpi_ls = np.fft.fft2(ldos_diff_ls)
    # 子图
    plt.subplot(221)
    plt.pcolormesh(x_ls, y_ls, ldos0_ls, cmap="viridis")  # , clim=(0, 2))
    plt.axis("scaled")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.colorbar()
    # 子图
    plt.subplot(222)
    plt.pcolormesh(x_ls, y_ls, ldos1_ls, cmap="viridis")  # , clim=(0, 2))
    plt.axis("scaled")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.colorbar()
    # 子图
    plt.subplot(223)
    plt.pcolormesh(x_ls, y_ls, np.abs(ldos_diff_ls), cmap="viridis")  # , clim=(0, 2))
    plt.axis("scaled")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.colorbar()
    # 子图
    plt.subplot(224)
    plt.pcolormesh(x_ls, y_ls, np.abs(qpi_ls), cmap="viridis")  # , clim=(0, 2))
    plt.axis("scaled")
    plt.colorbar()
    #
    plt.show()


""" ================================================================================
    RUN
================================================================================ """
if __name__ == "__main__":
    # main_dispersion()
    # main_dispersion_plot()
    # main_dispersion_diff_plot()
    main_dispersion_Gf()
    main_dispersion_Gf_plot()
    # main_Fermi_surface()
    # main_Fermi_surface_plot()
    # main_Fermi_surface_diff_plot()
    # main_Fermi_surface_parity()
    # main_Fermi_surface_parity_plot()

    # main_dos_Gf()
    # main_dos_Gf_plot()
    # main_ldos_Gf()
    # main_ldos_Gf_plot()
    # main_ldos_Gf_imp()
    # main_ldos_Gf_imp_plot()
    #
    # main_ldos_imp_x_y()
    # main_ldos_imp_x_y_plot()
