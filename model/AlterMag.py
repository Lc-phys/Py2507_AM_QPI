import numpy as np
from scipy import sparse as sspar

from model.Bravais_Rect import Lattice_Rectangular

SIGMA_0 = np.eye(2)
SIGMA_X = np.array([[0, 1], [1, 0]])
SIGMA_Y = np.array([[0, -1j], [1j, 0]])
SIGMA_Z = np.array([[1, 0], [0, -1]])
SIGMA_P = (SIGMA_X + 1j * SIGMA_Y) / 2
SIGMA_M = (SIGMA_X - 1j * SIGMA_Y) / 2
SIGMA_00 = np.eye(4)
SIGMA_X0 = np.kron(SIGMA_X, SIGMA_0)
SIGMA_Z0 = np.kron(SIGMA_Z, SIGMA_0)
SIGMA_0Z = np.kron(SIGMA_0, SIGMA_Z)
SIGMA_ZZ = np.kron(SIGMA_Z, SIGMA_Z)
SIGMA_P0 = np.kron(SIGMA_P, SIGMA_0)
SIGMA_000 = np.eye(8)
SIGMA_X00 = np.kron(SIGMA_X, SIGMA_00)
SIGMA_YZ0 = np.kron(SIGMA_Y, SIGMA_Z0)
SIGMA_Y00 = np.kron(SIGMA_Y, SIGMA_00)
SIGMA_M00 = np.kron(SIGMA_M, SIGMA_00)
SIGMA_MZ0 = np.kron(SIGMA_M, SIGMA_Z0)
SIGMA_P00 = np.kron(SIGMA_P, SIGMA_00)
SIGMA_PZ0 = np.kron(SIGMA_P, SIGMA_Z0)
SIGMA_0X0 = np.kron(SIGMA_0, SIGMA_X0)
SIGMA_XX0 = np.kron(SIGMA_0, SIGMA_X0)
SIGMA_ZY0 = np.kron(SIGMA_Z, np.kron(SIGMA_Y, SIGMA_0))
SIGMA_MX0 = np.kron(SIGMA_M, SIGMA_X0)
SIGMA_PX0 = np.kron(SIGMA_P, SIGMA_X0)
SIGMA_YX0 = np.kron(SIGMA_Y, SIGMA_X0)
SIGMA_0YZ = np.kron(np.kron(SIGMA_0, SIGMA_Y), SIGMA_Z)
SIGMA_XYZ = np.kron(np.kron(SIGMA_X, SIGMA_Y), SIGMA_Z)
SIGMA_YYZ = np.kron(np.kron(SIGMA_Y, SIGMA_Y), SIGMA_Z)
SIGMA_ZXZ = np.kron(np.kron(SIGMA_Z, SIGMA_X), SIGMA_Z)


class Model_AM_xy:
    def __init__(self, n_site=(100, 100)):
        self.name = "AM-xy"
        # 晶格
        self.n_x, self.n_y = n_site
        self.lattice = Lattice_Rectangular(n_site)
        self.n_orbit = 2
        # Hamiltonian参数
        self.t_hop = [1.0, 0.7]
        # self.mu = 2.4
        self.am = 0.2
        # 杂质
        self.v_imp = 0.0
        self.tag_imp = (self.n_x // 2, self.n_y // 2)
        #
        self.temp = 0

    def set_parameter_k1v2se2o1(self):
        self.t_hop = [0.09, -0.05]
        self.am = 0.2

    def get_Hamiltonian(self, is_sparse=True):
        latt = self.lattice
        t1, t2 = self.t_hop
        am = self.am
        v_imp = self.v_imp

        # hopping标志
        flag_ons = latt.get_flag_onsite()
        flag_px = latt.get_flag_hop((1, 0))
        flag_py = latt.get_flag_hop((0, 1))
        flag_nn = latt.get_flag_hop((1, 1)) + latt.get_flag_hop((1, -1))
        #
        # band
        hmtn_ons = sspar.kron(flag_ons * 0j, SIGMA_0, format="csc")
        hmtn_off = sspar.kron(flag_px + flag_py, -t1 * SIGMA_0, format="csc")
        hmtn_off += sspar.kron(flag_nn, -t2 * SIGMA_0, format="csc")
        #
        # AM
        hmtn_off += sspar.kron(flag_px * (-am) + flag_py * (+am), SIGMA_Z, format="csc")
        #
        # 杂质
        idx_imp = latt.tag2idx(self.tag_imp)
        imp_ls = np.zeros(latt.n_site)
        imp_ls[idx_imp] = v_imp
        hmtn_ons += sspar.kron(flag_ons * imp_ls, SIGMA_0, format="csc")
        #
        # 合并
        hmtn = hmtn_ons + hmtn_off + hmtn_off.conj().T

        if not is_sparse:
            hmtn = hmtn.toarray()
        return hmtn


class Model_KV2Se2O_kk:
    def __init__(self):
        self.name = "KV2Se2O-kk"
        #
        self.n_orbit = 4
        #
        self.k_ = [0.0, 0.0]
        # Hamiltonian参数
        self.t2 = [-0.29, 0.10]
        self.t3 = 0.034
        self.t1 = 0.0
        self.m0 = -0.8
        self.mu = -0.68

    def set_parameter(self, name="FS"):
        if name == "FS":
            self.t2 = [-0.36, 0.08]
            self.t3 = 0.01
            # self.mu = -0.8
            self.m0 = -0.9
            self.t1 = 0.03
            self.mu = -0.9

    def get_Hamiltonian(self, k_=None):
        kx, ky = self.k_ if k_ is None else k_
        ts, tp = self.t2
        t3, t1 = self.t3, self.t1
        m0 = self.m0
        mu = self.mu

        hmtn = (
            ((ts + tp) * (np.cos(kx) + np.cos(ky)) + 4 * t3 * np.cos(kx) * np.cos(ky) - mu) * SIGMA_00
            + (ts - tp) * (np.cos(kx) - np.cos(ky)) * SIGMA_Z0
            + m0 * SIGMA_ZZ
            - 4 * t1 * np.sin(kx / 2) * np.sin(ky / 2) * SIGMA_X0
        )
        return hmtn


class Model_KV2Se2O_xy(Model_KV2Se2O_kk):
    def __init__(self, n_site=(100, 100)):
        super().__init__()
        self.name = "KV2Se2O-xy"
        # 晶格
        self.n_x, self.n_y = n_site
        self.lattice = Lattice_Rectangular(n_site)
        # 杂质
        self.v_imp = 0.0
        self.tag_imp = (self.n_x // 2, self.n_y // 2)
        #
        self.temp = 0

    def get_Hamiltonian(self, is_sparse=True):
        latt = self.lattice
        ts, tp = self.t2
        t3, t1 = self.t3, self.t1
        m0 = self.m0
        mu = self.mu
        v_imp = self.v_imp

        # hopping标志
        flag_ons = latt.get_flag_onsite()
        flag_px = latt.get_flag_hop((1, 0))
        flag_py = latt.get_flag_hop((0, 1))
        flag_pxpy = latt.get_flag_hop((1, 1))
        flag_pxmy = latt.get_flag_hop((1, -1))
        #
        # band
        hmtn_ons = sspar.kron(flag_ons * (-mu), SIGMA_00, format="csc")
        hmtn_off = sspar.csc_array(hmtn_ons.shape)
        hmtn_off += sspar.kron(flag_px + flag_py, (ts + tp) / 2 * SIGMA_00, format="csc")
        hmtn_off += sspar.kron(flag_px - flag_py, (ts - tp) / 2 * SIGMA_Z0, format="csc")
        # t3
        hmtn_off += sspar.kron(flag_pxpy + flag_pxmy, t3 * SIGMA_00, format="csc")
        # t1
        hmtn_ons += sspar.kron(flag_ons, t1 * SIGMA_X0, format="csc")
        hmtn_off += sspar.kron(flag_px + flag_py, -t1 * SIGMA_P0, format="csc")
        hmtn_off += sspar.kron(flag_pxpy, t1 * SIGMA_P0, format="csc")
        #
        # AFM
        hmtn_ons += sspar.kron(flag_ons * m0, SIGMA_ZZ, format="csc")
        #
        # 杂质
        idx_imp = latt.tag2idx(self.tag_imp)
        imp_ls = np.zeros(latt.n_site)
        imp_ls[idx_imp] = v_imp
        hmtn_ons += sspar.kron(flag_ons * imp_ls, SIGMA_00, format="csc")
        #
        # 合并
        hmtn = hmtn_ons + hmtn_off + hmtn_off.conj().T

        if not is_sparse:
            hmtn = hmtn.toarray()
        return hmtn


class Model_KV2Se2O_SDW_kk:
    def __init__(self):
        self.name = "KV2Se2O-SDW-kk"
        #
        self.n_orbit = 8
        #
        self.k_ = [0.0, 0.0]
        # Hamiltonian参数
        self.t2 = [-0.29, 0.10]
        self.t3 = 0.034
        self.t1 = 0.0
        self.m0 = [-0.8, 0.0]
        self.mu = -0.68
        self.so = 0.0

    def set_parameter(self, name="FS"):
        if name == "FS":
            self.t2 = [-0.36, 0.08]
            self.t3 = 0.01
            # self.mu = -0.8
            self.m0 = [-0.9, -0.02]
            self.t1 = 0.03
            self.mu = -0.9

    def get_Hamiltonian(self, k_=None):
        k1, k2 = self.k_ if k_ is None else k_
        ts, tp = self.t2
        t3, t1 = self.t3, self.t1
        m0p, m0m = self.m0
        mu = self.mu

        hmtn = (
            # np.asarray(
            #     [
            #         [
            #             0,
            #             t1 * (1 + np.exp(-1j * k1)),
            #             tp * (1 + np.exp(-1j * k1 + 1j * k2)) + ts * (np.exp(-1j * k1) + np.exp(1j * k2)),
            #             -t1 * (1 + np.exp(1j * k2)),
            #         ],
            #         [
            #             t1 * (1 + np.exp(-1j * k1)),
            #             0,
            #             -t1 * (1 + np.exp(1j * k2)),
            #             tp * (1 + np.exp(1j * k1 + 1j * k2)) + ts * (np.exp(1j * k1) + np.exp(1j * k2)),
            #         ],
            #         [
            #             tp * (1 + np.exp(1j * k1 - 1j * k2)) + ts * (np.exp(1j * k1) + np.exp(-1j * k2)),
            #             -t1 * (1 + np.exp(-1j * k2)),
            #             0,
            #             t1 * (1 + np.exp(1j * k1)),
            #         ],
            #         [
            #             -t1 * (1 + np.exp(-1j * k2)),
            #             tp * (1 + np.exp(-1j * k1 - 1j * k2)) + ts * (np.exp(-1j * k1) + np.exp(-1j * k2)),
            #             t1 * (1 + np.exp(-1j * k1)),
            #             0,
            #         ],
            #     ]
            # )
            np.asarray(
                [
                    [
                        0,
                        t1 * (1 + np.exp(-1j * k2)),
                        tp * (1 + np.exp(-1j * k1 - 1j * k2)) + ts * (np.exp(-1j * k1) + np.exp(-1j * k2)),
                        -t1 * (1 + np.exp(-1j * k1)),
                    ],
                    [
                        t1 * (1 + np.exp(1j * k2)),
                        0,
                        -t1 * (1 + np.exp(-1j * k1)),
                        tp * (1 + np.exp(-1j * k1 + 1j * k2)) + ts * (np.exp(-1j * k1) + np.exp(1j * k2)),
                    ],
                    [
                        tp * (1 + np.exp(1j * k1 + 1j * k2)) + ts * (np.exp(1j * k1) + np.exp(1j * k2)),
                        -t1 * (1 + np.exp(1j * k1)),
                        0,
                        t1 * (1 + np.exp(1j * k2)),
                    ],
                    [
                        -t1 * (1 + np.exp(1j * k1)),
                        tp * (1 + np.exp(1j * k1 - 1j * k2)) + ts * (np.exp(1j * k1) + np.exp(-1j * k2)),
                        t1 * (1 + np.exp(-1j * k2)),
                        0,
                    ],
                ]
            )
            + (2 * t3 * (np.cos(k1) + np.cos(k2)) - mu) * SIGMA_00
        )
        # 磁性
        hmtn = np.kron(hmtn, SIGMA_0) + np.kron(m0p * SIGMA_0 + m0m * SIGMA_Z, SIGMA_ZZ)
        # hmtn = np.kron(hmtn, SIGMA_0) + np.kron(m0p * SIGMA_0, SIGMA_ZZ) - np.kron(m0m * SIGMA_Z, SIGMA_0Z)  # SDW转90°
        # 自旋轨道耦合
        hmtn += (
            self.so
            * 1j
            * np.kron(
                [
                    [0, -(1 + np.exp(-1j * k2)), 0, +(1 + np.exp(-1j * k1))],
                    [+(1 + np.exp(1j * k2)), 0, -(1 + np.exp(-1j * k1)), 0],
                    [0, +(1 + np.exp(1j * k1)), 0, -(1 + np.exp(1j * k2))],
                    [-(1 + np.exp(1j * k1)), 0, +(1 + np.exp(-1j * k2)), 0],
                ],
                SIGMA_Z,
            )
        )

        return hmtn


class Model_KV2Se2O_CDW_kk:
    def __init__(self):
        self.name = "KV2Se2O-CDW-kk"
        #
        self.n_orbit = 8
        #
        self.k_ = [0.0, 0.0]
        # Hamiltonian参数
        self.t2 = [-0.29, 0.10]
        self.t3 = 0.034
        self.t1 = 0.0
        self.m0 = [-0.8, 0.0]
        self.mu = -0.68

    def set_parameter(self, name="FS"):
        if name == "FS":
            self.t2 = [-0.36, 0.08]
            self.t3 = 0.01
            self.mu = -0.8

    def get_Hamiltonian(self, k_=None):
        k1, k2 = self.k_ if k_ is None else k_
        ts, tp = self.t2
        t3, t1 = self.t3, self.t1
        m0p, m0m = self.m0
        mu = self.mu

        hmtn = (
            np.asarray(
                [
                    [
                        0,
                        t1 * (1 + np.exp(-1j * k1)),
                        tp * (1 + np.exp(-1j * k1 + 1j * k2)) + ts * (np.exp(-1j * k1) + np.exp(1j * k2)),
                        -t1 * (1 + np.exp(1j * k2)),
                    ],
                    [
                        t1 * (1 + np.exp(-1j * k1)),
                        0,
                        -t1 * (1 + np.exp(1j * k2)),
                        tp * (1 + np.exp(1j * k1 + 1j * k2)) + ts * (np.exp(1j * k1) + np.exp(1j * k2)),
                    ],
                    [
                        tp * (1 + np.exp(1j * k1 - 1j * k2)) + ts * (np.exp(1j * k1) + np.exp(-1j * k2)),
                        -t1 * (1 + np.exp(-1j * k2)),
                        0,
                        t1 * (1 + np.exp(1j * k1)),
                    ],
                    [
                        -t1 * (1 + np.exp(-1j * k2)),
                        tp * (1 + np.exp(-1j * k1 - 1j * k2)) + ts * (np.exp(-1j * k1) + np.exp(-1j * k2)),
                        t1 * (1 + np.exp(-1j * k1)),
                        0,
                    ],
                ]
            )
            + (2 * t3 * (np.cos(k1) + np.cos(k2)) - mu) * SIGMA_00
        )
        hmtn = np.kron(hmtn, SIGMA_0) + np.kron(m0p * SIGMA_0, SIGMA_ZZ) + np.kron(m0m * SIGMA_Z, SIGMA_00)

        return hmtn


class Model_KV2Se2O_SDW_kkk:
    def __init__(self):
        self.name = "KV2Se2O-SDW-kkk"
        #
        self.n_orbit = 8
        #
        self.k_ = [0.0, 0.0,0.0]
        # Hamiltonian参数
        self.t2 = [-0.29, 0.10]
        self.t3 = 0.034
        self.t1 = 0.0
        self.tz=0.01
        self.m0 = [-0.8, 0.0]
        self.mu = -0.68
        self.so = 0.0

    def set_parameter(self, name="FS"):
        if name == "FS":
            self.t2 = [-0.36, 0.08]
            self.t3 = 0.01
            # self.mu = -0.8
            self.m0 = [-0.9, -0.02]
            self.t1 = 0.03
            self.mu = -0.9

    def get_Hamiltonian(self, k_=None):
        k1, k2,k3 = self.k_ if k_ is None else k_
        ts, tp = self.t2
        t3, t1 = self.t3, self.t1
        tz = self.tz
        m0p, m0m = self.m0
        mu = self.mu

        hmtn = (
            np.asarray(
                [
                    [
                        0,
                        t1 * (1 + np.exp(-1j * k2)),
                        tp * (1 + np.exp(-1j * k1 - 1j * k2)) + ts * (np.exp(-1j * k1) + np.exp(-1j * k2)),
                        -t1 * (1 + np.exp(-1j * k1)),
                    ],
                    [
                        t1 * (1 + np.exp(1j * k2)),
                        0,
                        -t1 * (1 + np.exp(-1j * k1)),
                        tp * (1 + np.exp(-1j * k1 + 1j * k2)) + ts * (np.exp(-1j * k1) + np.exp(1j * k2)),
                    ],
                    [
                        tp * (1 + np.exp(1j * k1 + 1j * k2)) + ts * (np.exp(1j * k1) + np.exp(1j * k2)),
                        -t1 * (1 + np.exp(1j * k1)),
                        0,
                        t1 * (1 + np.exp(1j * k2)),
                    ],
                    [
                        -t1 * (1 + np.exp(1j * k1)),
                        tp * (1 + np.exp(1j * k1 - 1j * k2)) + ts * (np.exp(1j * k1) + np.exp(-1j * k2)),
                        t1 * (1 + np.exp(-1j * k2)),
                        0,
                    ],
                ]
            )
            + (2 * t3 * (np.cos(k1) + np.cos(k2))+2*tz*(1-np.cos(k3)) - mu) * SIGMA_00
        )
        # 磁性
        hmtn = np.kron(hmtn, SIGMA_0) + np.kron(m0p * SIGMA_0 + m0m * SIGMA_Z, SIGMA_ZZ)
        # hmtn = np.kron(hmtn, SIGMA_0) + np.kron(m0p * SIGMA_0, SIGMA_ZZ) - np.kron(m0m * SIGMA_Z, SIGMA_0Z)  # SDW转90°
        # 自旋轨道耦合
        hmtn += (
            self.so
            * 1j
            * np.kron(
                [
                    [0, -(1 + np.exp(-1j * k2)), 0, +(1 + np.exp(-1j * k1))],
                    [+(1 + np.exp(1j * k2)), 0, -(1 + np.exp(-1j * k1)), 0],
                    [0, +(1 + np.exp(1j * k1)), 0, -(1 + np.exp(1j * k2))],
                    [-(1 + np.exp(1j * k1)), 0, +(1 + np.exp(-1j * k2)), 0],
                ],
                SIGMA_Z,
            )
        )

        return hmtn


class Model_KV2Se2O_SDW_xy(Model_KV2Se2O_SDW_kk):
    def __init__(self, n_site=(100, 100)):
        super().__init__()
        self.name = "KV2Se2O-SDW-xy"
        # 晶格
        self.n_x, self.n_y = n_site
        self.lattice = Lattice_Rectangular(n_site)
        # 杂质
        self.v_imp = 0.0
        self.tag_imp = np.asarray((self.n_x // 2, self.n_y // 2))
        #
        self.temp = 0

    def get_Hamiltonian(self, is_sparse=True):
        latt = self.lattice
        ts, tp = self.t2
        t3, t1 = self.t3, self.t1
        m0p, m0m = self.m0
        mu = self.mu
        v_imp = self.v_imp

        # hopping标志
        flag_ons = latt.get_flag_onsite()
        flag_px = latt.get_flag_hop((1, 0))
        flag_py = latt.get_flag_hop((0, 1))
        flag_pxpy = latt.get_flag_hop((1, 1))
        flag_pxmy = latt.get_flag_hop((1, -1))
        #
        # band
        hmtn_ons = sspar.kron(flag_ons, -mu * SIGMA_000, format="csc")
        hmtn_ons += sspar.kron(flag_ons, tp * SIGMA_X00, format="csc")
        hmtn_off = sspar.csc_array(hmtn_ons.shape)
        hmtn_off += sspar.kron(flag_px, ts * (SIGMA_X00 + 1j * SIGMA_YZ0) / 2, format="csc")
        hmtn_off += sspar.kron(flag_py, ts * (SIGMA_X00 - 1j * SIGMA_Y00) / 2, format="csc")
        hmtn_off += sspar.kron(flag_pxpy, tp * (SIGMA_M00 - SIGMA_MZ0) / 2, format="csc")
        hmtn_off += sspar.kron(flag_pxmy, tp * (SIGMA_P00 + SIGMA_PZ0) / 2, format="csc")
        # t3
        hmtn_off += sspar.kron(flag_px + flag_py, t3 * SIGMA_000, format="csc")
        # t1
        hmtn_ons += sspar.kron(flag_ons, t1 * (SIGMA_0X0 - SIGMA_XX0), format="csc")
        hmtn_off += sspar.kron(flag_px, +t1 * (SIGMA_0X0 + 1j * SIGMA_ZY0) / 2, format="csc")
        hmtn_off += sspar.kron(flag_py, -t1 * SIGMA_MX0, format="csc")
        #
        # AFM
        hmtn_ons += sspar.kron(flag_ons, np.kron(m0p * SIGMA_0, SIGMA_ZZ), format="csc")
        #
        # SDW
        # hmtn_ons += sspar.kron(flag_ons, np.kron(m0m * SIGMA_Z, SIGMA_ZZ), format="csc")
        # hmtn_ons += sspar.kron(flag_ons, np.kron(m0m * SIGMA_Z, SIGMA_0Z), format="csc")  # SDW转90°
        r2imp_ls = np.linalg.norm(latt.tag_ls- self.tag_imp, axis=1)
        away_imp_ls = (r2imp_ls > 3.1)
        hmtn_ons += sspar.kron(flag_ons * away_imp_ls, np.kron(m0m * SIGMA_Z, SIGMA_ZZ), format="csc") # 近关
        #
        # 杂质
        idx_imp = latt.tag2idx(self.tag_imp)
        imp_ls = np.zeros(latt.n_site)
        imp_ls[idx_imp] = v_imp
        hmtn_ons += sspar.kron(flag_ons * imp_ls, SIGMA_000, format="csc")
        #
        # 合并
        hmtn = hmtn_ons + hmtn_off + hmtn_off.conj().T

        if not is_sparse:
            hmtn = hmtn.toarray()
        return hmtn


class Model_KV2Se2O_SDW_xy_rev(Model_KV2Se2O_SDW_kk):
    def __init__(self, n_site=(100, 100)):
        super().__init__()
        self.name = "KV2Se2O-SDW-xy"
        # 晶格
        self.n_x, self.n_y = n_site
        self.lattice = Lattice_Rectangular(n_site)
        # 杂质
        self.v_imp = 0.0
        self.tag_imp = np.asarray((self.n_x // 2, self.n_y // 2))
        #
        self.temp = 0

    def get_Hamiltonian(self, is_sparse=True):
        latt = self.lattice
        ts, tp = self.t2
        t3, t1 = self.t3, self.t1
        so = self.so
        m0p, m0m = self.m0
        mu = self.mu
        v_imp = self.v_imp

        # hopping标志
        flag_ons = latt.get_flag_onsite()
        flag_px = latt.get_flag_hop((1, 0))
        flag_py = latt.get_flag_hop((0, 1))
        flag_pxpy = latt.get_flag_hop((1, 1))
        flag_pxmy = latt.get_flag_hop((1, -1))
        #
        # band
        hmtn_ons = sspar.kron(flag_ons, -mu * SIGMA_000, format="csc")
        hmtn_ons += sspar.kron(flag_ons, tp * SIGMA_X00, format="csc")
        hmtn_off = sspar.csc_array(hmtn_ons.shape)
        hmtn_off += sspar.kron(flag_px, ts * (SIGMA_X00 + 1j * SIGMA_Y00) / 2, format="csc")
        hmtn_off += sspar.kron(flag_py, ts * (SIGMA_X00 + 1j * SIGMA_YZ0) / 2, format="csc")
        hmtn_off += sspar.kron(flag_pxpy, tp * (SIGMA_P00 + SIGMA_PZ0) / 2, format="csc")
        hmtn_off += sspar.kron(flag_pxmy, tp * (SIGMA_P00 - SIGMA_PZ0) / 2, format="csc")
        # t3
        hmtn_off += sspar.kron(flag_px + flag_py, t3 * SIGMA_000, format="csc")
        # t1
        hmtn_ons += sspar.kron(flag_ons, t1 * (SIGMA_0X0 - SIGMA_XX0), format="csc")
        hmtn_off += sspar.kron(flag_px, -t1 * (SIGMA_XX0 + 1j * SIGMA_YX0) / 2, format="csc")
        hmtn_off += sspar.kron(flag_py, +t1 * (SIGMA_0X0 + 1j * SIGMA_ZY0) / 2, format="csc")
        #
        # SOC
        hmtn_ons += sspar.kron(flag_ons, so * (SIGMA_0YZ - SIGMA_XYZ), format="csc")
        hmtn_off += sspar.kron(flag_px, -so * (SIGMA_XYZ + 1j * SIGMA_YYZ) / 2, format="csc")
        hmtn_off += sspar.kron(flag_py, +so * (SIGMA_0YZ - 1j * SIGMA_ZXZ) / 2, format="csc")
        #
        # AFM
        hmtn_ons += sspar.kron(flag_ons, np.kron(m0p * SIGMA_0, SIGMA_ZZ), format="csc")
        #
        # SDW
        hmtn_ons += sspar.kron(flag_ons, np.kron(m0m * SIGMA_Z, SIGMA_ZZ), format="csc")
        # hmtn_ons += sspar.kron(flag_ons, np.kron(m0m * SIGMA_Z, SIGMA_0Z), format="csc")  # SDW转90°
        # r2imp_ls = np.linalg.norm(latt.tag_ls- self.tag_imp, axis=1)
        # away_imp_ls = (r2imp_ls > 2.1)
        # hmtn_ons += sspar.kron(flag_ons * away_imp_ls, np.kron(m0m * SIGMA_Z, SIGMA_ZZ), format="csc") # 近关
        #
        # 杂质
        idx_imp = latt.tag2idx(self.tag_imp)
        imp_ls = np.zeros(latt.n_site)
        imp_ls[idx_imp] = v_imp
        hmtn_ons += sspar.kron(flag_ons * imp_ls, SIGMA_000, format="csc")
        #
        # 合并
        hmtn = hmtn_ons + hmtn_off + hmtn_off.conj().T

        if not is_sparse:
            hmtn = hmtn.toarray()
        return hmtn


class Model_KV2Se2O_SDW_xy_itf(Model_KV2Se2O_SDW_kk):
    def __init__(self, n_site=(100, 100)):
        super().__init__()
        self.name = "KV2Se2O-SDW-xy"
        # 晶格
        self.n_x, self.n_y = n_site
        self.lattice = Lattice_Rectangular(n_site)
        # 杂质
        self.v_imp = 0.0
        self.tag_imp = np.asarray((self.n_x // 2, self.n_y // 2))
        #
        self.temp = 0

    def get_Hamiltonian(self, is_sparse=True):
        latt = self.lattice
        ts, tp = self.t2
        t3, t1 = self.t3, self.t1
        m0p, m0m = self.m0
        mu = self.mu
        v_imp = self.v_imp

        # hopping标志
        flag_ons = latt.get_flag_onsite()
        flag_px = latt.get_flag_hop((1, 0))
        flag_py = latt.get_flag_hop((0, 1))
        flag_pxpy = latt.get_flag_hop((1, 1))
        flag_pxmy = latt.get_flag_hop((1, -1))
        #
        # band
        hmtn_ons = sspar.kron(flag_ons, -mu * SIGMA_000, format="csc")
        hmtn_ons += sspar.kron(flag_ons, tp * SIGMA_X00, format="csc")
        hmtn_off = sspar.csc_array(hmtn_ons.shape)
        hmtn_off += sspar.kron(flag_px, ts * (SIGMA_X00 + 1j * SIGMA_YZ0) / 2, format="csc")
        hmtn_off += sspar.kron(flag_py, ts * (SIGMA_X00 - 1j * SIGMA_Y00) / 2, format="csc")
        hmtn_off += sspar.kron(flag_pxpy, tp * (SIGMA_M00 - SIGMA_MZ0) / 2, format="csc")
        hmtn_off += sspar.kron(flag_pxmy, tp * (SIGMA_P00 + SIGMA_PZ0) / 2, format="csc")
        # t3
        hmtn_off += sspar.kron(flag_px + flag_py, t3 * SIGMA_000, format="csc")
        # t1
        hmtn_ons += sspar.kron(flag_ons, t1 * (SIGMA_0X0 - SIGMA_XX0), format="csc")
        hmtn_off += sspar.kron(flag_px, +t1 * (SIGMA_0X0 + 1j * SIGMA_ZY0) / 2, format="csc")
        hmtn_off += sspar.kron(flag_py, -t1 * SIGMA_MX0, format="csc")
        #
        # AFM
        hmtn_ons += sspar.kron(flag_ons, np.kron(m0p * SIGMA_0, SIGMA_ZZ), format="csc")
        flag_sdw = np.where((latt.tag_ls[:, 0] == self.n_x // 2) & (latt.tag_ls[:, 1] == self.n_y // 2), 1, -1)
        hmtn_ons += sspar.kron(flag_ons * flag_sdw, np.kron(m0m * SIGMA_Z, SIGMA_ZZ), format="csc")
        #
        # 杂质
        idx_imp = latt.tag2idx(self.tag_imp)
        imp_ls = np.zeros(latt.n_site)
        imp_ls[idx_imp] = v_imp
        hmtn_ons += sspar.kron(flag_ons * imp_ls, SIGMA_000, format="csc")
        #
        # 合并
        hmtn = hmtn_ons + hmtn_off + hmtn_off.conj().T

        if not is_sparse:
            hmtn = hmtn.toarray()
        return hmtn


class Model_KV2Se2O_SDW_xy_bak1(Model_KV2Se2O_SDW_kk):
    def __init__(self, n_site=(100, 100)):
        super().__init__()
        self.name = "KV2Se2O-SDW-xy"
        # 晶格
        self.n_x, self.n_y = n_site
        self.lattice = Lattice_Rectangular(n_site)
        # 杂质
        self.v_imp = 0.0
        self.tag_imp = (self.n_x // 2, self.n_y // 2)
        #
        self.temp = 0

    def get_Hamiltonian(self, is_sparse=True):
        latt = self.lattice
        ts, tp = self.t2
        t3, t1 = self.t3, self.t1
        m0p, m0m = self.m0
        mu = self.mu
        v_imp = self.v_imp

        # hopping标志
        flag_ons = latt.get_flag_onsite()
        flag_px = latt.get_flag_hop((1, 0))
        flag_py = latt.get_flag_hop((0, 1))
        flag_pxpy = latt.get_flag_hop((1, 1))
        flag_pxmy = latt.get_flag_hop((1, -1))
        #
        # band
        hmtn_ons = sspar.kron(flag_ons, -mu * SIGMA_000, format="csc")
        hmtn_ons += sspar.kron(flag_ons, tp * SIGMA_X00, format="csc")
        hmtn_off = sspar.csc_array(hmtn_ons.shape)
        hmtn_off += sspar.kron(flag_px, ts * SIGMA_M00, format="csc")
        hmtn_off += sspar.kron(flag_py, ts * (SIGMA_X00 + 1j * SIGMA_YZ0) / 2, format="csc")
        hmtn_off += sspar.kron(flag_pxpy, tp * (SIGMA_P00 + SIGMA_PZ0) / 2, format="csc")
        hmtn_off += sspar.kron(flag_pxmy, tp * (SIGMA_P00 - SIGMA_PZ0) / 2, format="csc")
        # t3
        hmtn_off += sspar.kron(flag_px + flag_py, t3 * SIGMA_000, format="csc")
        # t1
        hmtn_ons += sspar.kron(flag_ons, t1 * (SIGMA_0X0 - SIGMA_XX0), format="csc")
        hmtn_off += sspar.kron(flag_px, -t1 * SIGMA_PX0, format="csc")
        hmtn_off += sspar.kron(flag_py, +t1 * (SIGMA_0X0 + 1j * SIGMA_ZY0) / 2, format="csc")
        #
        # AFM
        hmtn_ons += sspar.kron(flag_ons, np.kron(m0p * SIGMA_0 + m0m * SIGMA_Z, SIGMA_ZZ), format="csc")
        #
        # 杂质
        idx_imp = latt.tag2idx(self.tag_imp)
        imp_ls = np.zeros(latt.n_site)
        imp_ls[idx_imp] = v_imp
        hmtn_ons += sspar.kron(flag_ons * imp_ls, SIGMA_000, format="csc")
        #
        # 合并
        hmtn = hmtn_ons + hmtn_off + hmtn_off.conj().T

        if not is_sparse:
            hmtn = hmtn.toarray()
        return hmtn


class Model_KV2Se2O_CDW_xy(Model_KV2Se2O_SDW_kk):
    def __init__(self, n_site=(100, 100)):
        super().__init__()
        self.name = "KV2Se2O-SDW-xy"
        # 晶格
        self.n_x, self.n_y = n_site
        self.lattice = Lattice_Rectangular(n_site)
        # 杂质
        self.v_imp = 0.0
        self.tag_imp = (self.n_x // 2, self.n_y // 2)
        #
        self.temp = 0

    def get_Hamiltonian(self, is_sparse=True):
        latt = self.lattice
        ts, tp = self.t2
        t3, t1 = self.t3, self.t1
        m0p, m0m = self.m0
        mu = self.mu
        v_imp = self.v_imp

        # hopping标志
        flag_ons = latt.get_flag_onsite()
        flag_px = latt.get_flag_hop((1, 0))
        flag_py = latt.get_flag_hop((0, 1))
        flag_pxpy = latt.get_flag_hop((1, 1))
        flag_pxmy = latt.get_flag_hop((1, -1))
        #
        # band
        hmtn_ons = sspar.kron(flag_ons, -mu * SIGMA_000, format="csc")
        hmtn_ons += sspar.kron(flag_ons, tp * SIGMA_X00, format="csc")
        hmtn_off = sspar.csc_array(hmtn_ons.shape)
        hmtn_off += sspar.kron(flag_px, ts * (SIGMA_X00 + 1j * SIGMA_YZ0) / 2, format="csc")
        hmtn_off += sspar.kron(flag_py, ts * (SIGMA_X00 - 1j * SIGMA_Y00) / 2, format="csc")
        hmtn_off += sspar.kron(flag_pxpy, tp * (SIGMA_M00 - SIGMA_MZ0) / 2, format="csc")
        hmtn_off += sspar.kron(flag_pxmy, tp * (SIGMA_P00 + SIGMA_PZ0) / 2, format="csc")
        # t3
        hmtn_off += sspar.kron(flag_px + flag_py, t3 * SIGMA_000, format="csc")
        # t1
        hmtn_ons += sspar.kron(flag_ons, t1 * (SIGMA_0X0 - SIGMA_XX0), format="csc")
        hmtn_off += sspar.kron(flag_px, +t1 * (SIGMA_0X0 + 1j * SIGMA_ZY0) / 2, format="csc")
        hmtn_off += sspar.kron(flag_py, -t1 * SIGMA_MX0, format="csc")
        #
        # AFM
        hmtn_ons += sspar.kron(flag_ons, np.kron(m0p * SIGMA_0, SIGMA_ZZ), format="csc")
        hmtn_ons += sspar.kron(flag_ons, np.kron(m0m * SIGMA_Z, SIGMA_00), format="csc")
        #
        # 杂质
        idx_imp = latt.tag2idx(self.tag_imp)
        imp_ls = np.zeros(latt.n_site)
        imp_ls[idx_imp] = v_imp
        hmtn_ons += sspar.kron(flag_ons * imp_ls, SIGMA_000, format="csc")
        #
        # 合并
        hmtn = hmtn_ons + hmtn_off + hmtn_off.conj().T

        if not is_sparse:
            hmtn = hmtn.toarray()
        return hmtn


class Model_KV2Se2O_dxy_kk:
    def __init__(self):
        self.name = "KV2Se2O-dxy-kk"
        #
        self.n_orbit = 4
        #
        self.k_ = [0.0, 0.0]
        # Hamiltonian参数
        self.t2 = [-0.11, 0.1]
        self.t3 = -0.07
        self.t1 = -0.2
        self.m0 = -0.8
        self.mu = -0.18

    def set_parameter(self, name="FS"):
        if name == "FS":
            self.t1 = -0.27
            self.t2 = [-0.25, -0.1]
            self.t3 = -0.23
            self.m0 = -2
            self.mu = -0.9

    def get_Hamiltonian(self, k_=None):
        kx, ky = self.k_ if k_ is None else k_
        ts, tp = self.t2
        t3, t1 = self.t3, self.t1
        m0 = self.m0
        mu = self.mu

        hmtn = (
            ((ts + tp) * (np.cos(kx) + np.cos(ky)) + 4 * t3 * np.cos(kx) * np.cos(ky) - mu) * SIGMA_00
            + (ts - tp) * (np.cos(kx) - np.cos(ky)) * SIGMA_Z0
            + m0 * SIGMA_ZZ
            + 4 * t1 * np.cos(kx / 2) * np.cos(ky / 2) * SIGMA_X0
        )
        return hmtn
