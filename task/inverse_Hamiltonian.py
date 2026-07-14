import time
import numpy as np
import scipy.sparse as ssp


def inverse_matrix_part(mtx, row_idx_ls=None, col_idx_ls=None):
    """
    计算稀疏矩阵的逆(的一部分)
    :param mtx: 待求逆稀疏矩阵
    :param row_idx_ls: 逆矩阵行索引列表
    :param col_idx_ls: 逆矩阵列索引列表
    :return: 逆(子)矩阵
    """
    if row_idx_ls is None:  # 行指标为None, 计算全部行
        row_idx_ls = np.arange(mtx.shape[1])
    if col_idx_ls is None:  # 列指标为None, 计算全部列
        col_idx_ls = np.arange(mtx.shape[0])

    # 准备单位矩阵(的子矩阵)
    n_row = mtx.shape[0]
    n_col = len(col_idx_ls)
    iden_part = ssp.csc_array((np.ones(n_col), (col_idx_ls, np.arange(n_col))), shape=(n_row, n_col))

    # 求逆(取子矩阵)
    inv_part = ssp.linalg.spsolve(mtx, iden_part)
    inv_part = inv_part[row_idx_ls, :]

    return inv_part


def calculate_Green_function(hmtn, energy, eta=1e-3, self_eng=None, r1_ls=None, r0_ls=None, n_orb=1):
    """
    计算格林函数 G(r1,r0)
    :param hmtn: 哈密顿量
    :param energy: 能量
    :param eta: 虚部小量
    :param self_eng: 自能
    :param r1_ls: r1索引
    :param r0_ls: r0索引
    :param n_orb: 轨道数
    :return: 格林函数(子矩阵)
    """

    if r1_ls is None and r0_ls is None:
        # 转换为稠密矩阵求逆
        hmtn = hmtn.toarray() if ssp.issparse(hmtn) else hmtn

        # G^-1 = E + iη - H - Σ
        gf_inv = np.eye(hmtn.shape[0], dtype=complex) * (energy + 1j * eta) - hmtn
        # 补上自能Σ
        if self_eng is not None:
            self_eng = self_eng.toarray() if ssp.issparse(self_eng) else self_eng
            gf_inv -= self_eng

        # 求逆计算格林函数
        gf = np.linalg.inv(gf_inv)

    else:
        # 使用稀疏矩阵求逆矩阵局部
        hmtn = ssp.csc_array(hmtn) if not ssp.issparse(hmtn) else hmtn

        # G^-1 = E + iη - H - Σ
        gf_inv = ssp.eye_array(hmtn.shape[0], dtype=complex, format="csc") * (energy + 1j * eta) - hmtn
        # 补上自能Σ
        if self_eng is not None:
            self_eng = ssp.csc_array(self_eng) if not ssp.issparse(self_eng) else self_eng
            gf_inv -= self_eng

        # 子矩阵位置
        if r1_ls is not None:
            r1_ls = np.repeat(r1_ls, n_orb) * n_orb + np.tile(np.arange(n_orb), len(r1_ls))
        if r0_ls is not None:
            r0_ls = np.repeat(r0_ls, n_orb) * n_orb + np.tile(np.arange(n_orb), len(r0_ls))

        # 求逆计算格林函数
        gf = inverse_matrix_part(gf_inv, r1_ls, r0_ls)

    return gf


def calculate_surfase_Green_function_semiinf(ginv_ons, hmtn_hop, tol=1e-6, max_iter=100, is_2dir=False):
    """
    计算表面格林函数
    :param ginv_ons: (E+iη-H) on-site矩阵
    :param hmtn_hop: 哈密顿量中从无穷远往表面方向的hopping矩阵
    :param tol: 相对容差
    :param max_iter: 最大迭代次数
    :return: 表面格林函数
    """
    # 初始值
    ginv_surf0_rev = ginv_surf0 = ginv_bulk0 = ginv_ons  # d, D
    hop_sb0 = hmtn_hop  # A
    hop_bs0 = hmtn_hop.T.conj()  # B

    is_cvg = False
    for i in range(max_iter):
        gf_isol0 = np.linalg.inv(ginv_bulk0)  # 独立格林函数 D^-1
        ginv_surf_diff = hop_sb0 @ gf_isol0 @ hop_bs0  # 表面格林函数变化 A D^-1 B
        ginv_surf_diff_rev = hop_bs0 @ gf_isol0 @ hop_sb0  # 反向表面格林函数变化 B D^-1 A
        # 更新表面
        ginv_surf = ginv_surf0 - ginv_surf_diff  # 表面 d = d - A D^-1 B
        if is_2dir:
            ginv_surf_rev = ginv_surf0_rev - ginv_surf_diff_rev  # 反向表面 d = d - B D^-1 A
        # 检查收敛
        err_rel = np.linalg.norm(ginv_surf_diff) / np.linalg.norm(ginv_surf)
        if err_rel < tol:
            is_cvg = True
            # print("Self-Engergy Converged. @ i = %d, δ = %.3e" % (i, err_rel))
            break
        # 更新内部
        ginv_bulk = ginv_bulk0 - ginv_surf_diff - ginv_surf_diff_rev  # D = D - A D^-1 B - B D^-1 A
        hop_sb = hop_sb0 @ gf_isol0 @ hop_sb0  # A = A D^-1 A
        hop_bs = hop_bs0 @ gf_isol0 @ hop_bs0  # B = B D^-1 B
        # 覆盖旧值
        ginv_surf0, ginv_bulk0, hop_sb0, hop_bs0 = ginv_surf, ginv_bulk, hop_sb, hop_bs
        if is_2dir:
            ginv_surf0_rev = ginv_surf_rev

    if not is_cvg:
        print("\033[33;7m G_surf NOT Converge! δ = %.3e \033[0m" % err_rel)

    # 表面格林函数 g = d^-1
    gf_surf = np.linalg.inv(ginv_surf)
    if is_2dir:
        gf_surf = (gf_surf, np.linalg.inv(ginv_surf_rev))
    return gf_surf


""" ================================================================================
    虚频格林函数
================================================================================ """


def calculate_expectation_1freq(opr, freq, hmtn, self_eng=None):
    """
    计算算符期望值(单虚频)
    :param freq: 频率能量
    :param opr: 算符
    :param hmtn: 哈密顿量
    :param self_eng: 自能
    :return: 算符期望值/温度
    """
    hmtn = ssp.csc_array(hmtn) if not ssp.issparse(hmtn) else hmtn
    # G^-1 = iω - H - Σ
    gf_inv = ssp.eye_array(hmtn.shape[0], dtype=complex, format="csc") * (1j * freq) - hmtn
    # 补上自能Σ
    if self_eng is not None:
        self_eng = ssp.csc_array(self_eng) if not ssp.issparse(self_eng) else self_eng
        gf_inv -= self_eng

    # <O> / T = Tr[ G(iω) O ]
    # t0 = time.perf_counter_ns()
    #
    # expt = ssp.linalg.spsolve(gf_inv, opr).trace() # 未利用稀疏性
    # expt = (ssp.linalg.inv(gf_inv) @ opr).trace()
    # expt = (np.linalg.inv(gf_inv.toarray()) @ opr.toarray()).trace()
    col_nnz = np.unique(opr.nonzero()[1])
    gf_dot_opr = ssp.linalg.spsolve(gf_inv, opr[:, col_nnz])
    expt = np.sum([gf_dot_opr[i, j] for j, i in enumerate(col_nnz)])
    #
    # t1 = time.perf_counter_ns()
    # print("耗时: ", t1 - t0)
    return expt


def calculate_Matsubara_sum(func_of_freq, temp=0.1, max_freq=101, tol=1e-6, is_msg=False):
    """
    计算Matsubara求和
    :param func_of_freq: 关于虚频的函数
    :param temp: 温度
    :param max_freq: 最大频率
    :param tol: 相对容差
    :return: Matsubara和
    """

    freq_ls = np.arange(1, max_freq + 1e-12, 2)

    mtbr_sum = 0
    is_cvg = False
    for freq_ in freq_ls:
        freq = freq_ * (np.pi * temp)
        new_term = np.asarray(func_of_freq(freq)) + np.asarray(func_of_freq(-freq))
        mtbr_sum += new_term
        # 检查收敛
        err_abs = np.linalg.norm(new_term)
        if err_abs < tol:
            is_cvg = True
        else:
            err_rel = err_abs / np.linalg.norm(mtbr_sum)
            if is_msg:
                print("n_ω = %d, \tδ = %.3e" % (freq_, err_rel))
            if err_rel < tol:
                is_cvg = True
        if is_cvg:  # 求和收敛
            # print("Matsubara Sum Converged at n_ω = %d" % freq_)
            break

    if not is_cvg:
        print("\033[33;7m M_Sum NOT Converged! δ = %.3e \033[0m" % err_rel)

    return mtbr_sum
