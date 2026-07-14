import numpy as np
from task import inverse_Hamiltonian as ih


""" ================================================================================
    LDoS(orbit)
================================================================================ """


def calculate_ldos_difference_from_impurity(hmtn, mu, eps, idx_imp, imp_mtx):
    """
    计算格林函数差值
    """

    n_orb = len(imp_mtx)
    n_s = len(hmtn) // n_orb
    # 格林函数 G(r,r0)
    gfr = ih.calculate_Green_function(hmtn, mu, +eps, r0_ls=[idx_imp], n_orb=n_orb)
    gf_r0 = np.reshape(gfr.toarray(), (n_s, n_orb, n_orb))  # G(r,o,o')
    # 格林函数 G(r0,r)
    # G^R(r0,r) = G^A(r,r0)^†
    gfa = ih.calculate_Green_function(hmtn, mu, -eps, r0_ls=[idx_imp], n_orb=n_orb)
    gf_0r = np.reshape(gfa.toarray(), (n_s, n_orb, n_orb))  # G(r,o,o')
    gf_0r = np.transpose(gf_0r, (0, 2, 1)).conj()
    # 格林函数 G(r0,r0)
    gf_00 = gf_r0[idx_imp]

    # G差值
    gf_diff_r = calculate_greenfunc_difference_from_impurity(imp_mtx, gf_00, gf_r0, gf_0r)
    # LDoS差值: LDoS = -Im[G] / π
    ldos_diff_r = -np.imag([np.diag(gf) for gf in gf_diff_r]) / np.pi  # DoS(r,o)
    return ldos_diff_r


def calculate_greenfunc_difference_from_impurity(imp_mtx, g0_00, g0_r0, g0_0r):
    """
    计算格林函数差值, 使用T矩阵
    """

    # T矩阵: T = (I - V @ G0)^(-1) @ V
    t_mtx = np.linalg.inv(np.eye(len(imp_mtx)) - imp_mtx @ g0_00) @ imp_mtx
    # G变化: δG = G0 @ T @ G0
    gf_diff_r = [g0_r0[i_] @ t_mtx @ g0_0r[i_] for i_ in np.arange(len(g0_r0))]

    return gf_diff_r


""" ================================================================================
    LDoS(r)
================================================================================ """


def get_pos_atom(pos_uc, pos_atom_1uc):
    """
    整合原子轨道位置列表
    """
    n_uc, n_o = len(pos_uc), len(pos_atom_1uc)

    pos_uc_ = np.repeat(pos_uc, n_o, axis=0)
    pos_atom_1uc_ = np.tile(pos_atom_1uc, (n_uc, 1))

    pos_atom = pos_uc_ + pos_atom_1uc_
    return pos_atom


def transform_ldos_orbit2position(ldos_o, func_atom, pos_atom, pos_probe):
    """
    LDoS原子位置转测量位置
    """
    # 拆分
    len_p = 256 * 1024 * 1024 // len(pos_atom)
    n_p = len(pos_probe) // len_p
    pos_probe_ls = [pos_probe[i_p * len_p : (i_p + 1) * len_p] for i_p in range(n_p)] + [pos_probe[(n_p) * len_p :]]

    # LDoS(r_p)
    def transform_part(pos_probe_):
        x1_ls, x0_ls = np.meshgrid(pos_probe_[:, 0], pos_atom[:, 0], indexing="ij")
        y1_ls, y0_ls = np.meshgrid(pos_probe_[:, 1], pos_atom[:, 1], indexing="ij")
        # cvv_mtx = func_atom(x1_ls - x0_ls) * func_atom(y1_ls - y0_ls)
        cvv_mtx = func_atom(((x1_ls - x0_ls) ** 2 + (y1_ls - y0_ls) ** 2) ** 0.5)
        ldos_p_ = cvv_mtx @ ldos_o
        return ldos_p_

    from tqdm import tqdm

    ldos_p = np.concatenate([transform_part(pos_probe_) for pos_probe_ in tqdm(pos_probe_ls)])
    return ldos_p


def transform_ldos_orbit2position_iso(ldos_o, func_atom, pos_atom, pos_probe):
    """
    LDoS原子位置转测量位置
    """

    # LDoS(r_p)
    def transform_part(pos_probe_):
        r_diff_ls = ((pos_probe_[0] - pos_atom[:, 0]) ** 2 + (pos_probe_[1] - pos_atom[:, 1]) ** 2) ** 0.5
        cvv_mtx = func_atom(r_diff_ls)
        ldos_p_ = cvv_mtx @ ldos_o
        return ldos_p_

    ldos_p = np.asarray([transform_part(pos_probe_) for pos_probe_ in (pos_probe)])
    return ldos_p


def transform_ldos_orbit2position_(ldos_o, func_atom, pos_atom, pos_probe):
    """
    LDoS原子位置转测量位置
    """

    # LDoS(r_p)
    def transform_part(pos_probe_):
        cvv_mtx = func_atom(pos_probe_[0] - pos_atom[:, 0], pos_probe_[1] - pos_atom[:, 1])
        ldos_p_ = cvv_mtx @ ldos_o
        return ldos_p_

    ldos_p = np.asarray([transform_part(pos_probe_) for pos_probe_ in (pos_probe)])
    return ldos_p


def transform_ldos_orbit2position_in_Cartesian(ldos_o, atom_func, atom_pos, x0_ls, y0_ls, x1_ls, y1_ls):
    """
    LDoS原子位置r0转测量位置r1
    """

    ldos_p = np.sum(
        [
            convolve_matrix_2d(
                ldos_o[:, :, i_o], atom_func, x0_ls + atom_pos[i_o, 0], y0_ls + atom_pos[i_o, 1], x1_ls, y1_ls
            )
            for i_o in np.arange(ldos_o.shape[-1])
        ],
        axis=0,
    )
    return ldos_p


def convolve_matrix_2d(mtx0, func_cvv, x0_ls, y0_ls, x1_ls, y1_ls):
    """
    二维空间矩阵变换
    M1(y1,x1) = Σ_{x0,y0} M0(y0,x0) * f(x1-x0) * f(y1-y0)
    """

    x1_ls, x0_ls = np.meshgrid(x1_ls, x0_ls, indexing="ij")
    cvv_x = func_cvv(x1_ls - x0_ls)
    y1_ls, y0_ls = np.meshgrid(y1_ls, y0_ls, indexing="ij")
    cvv_y = func_cvv(y1_ls - y0_ls)

    mtx1 = cvv_y @ mtx0 @ cvv_x.T
    return mtx1


def gaussian_delta(x, mu=0, sigma=1):
    """
    高斯δ函数
    """
    delta = np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    return delta


""" ================================================================================
    QPI
================================================================================ """


def calculate_qpi_from_ldos_2d(ldos_r, is_gamma_center=True):
    """
    计算QPI图样, 由2维LDoS作FFT得到
    """
    qpi_q = np.fft.fft2(ldos_r)  # QPI = FT[δLDoS(x,y)]
    if is_gamma_center:
        n1, n0 = qpi_q.shape  # Nk = N
        qpi_q = np.block(
            [
                [qpi_q[n1 // 2 :, n0 // 2 :], qpi_q[n1 // 2 :, : n0 // 2]],
                [qpi_q[: n1 // 2, n0 // 2 :], qpi_q[: n1 // 2, : n0 // 2]],
            ]
        )
    return qpi_q


def get_q_ls(qpi_shape, is_gamma_center=True):
    """
    QPI横纵坐标
    """
    n1, n0 = qpi_shape
    q0_ls = np.linspace(0, 2 * np.pi, n0 + 1)[:-1]
    q1_ls = np.linspace(0, 2 * np.pi, n1 + 1)[:-1]
    if is_gamma_center:
        q0_ls = np.concatenate((q0_ls[n0 // 2 :] - 2 * np.pi, q0_ls[: n0 // 2]))
        q1_ls = np.concatenate((q1_ls[n1 // 2 :] - 2 * np.pi, q1_ls[: n1 // 2]))
    return q0_ls, q1_ls
