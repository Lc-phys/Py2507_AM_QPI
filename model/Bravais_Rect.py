import numpy as np
from scipy import sparse as ssp
from functools import lru_cache


class Lattice_Linear:
    """
    一维链格点
    """

    def __init__(self, n_site=100):
        self.n_site = n_site
        self.lattice_constant = 1.0
        self.is_PBC = False  # 默认开边界

    @lru_cache
    def get_flag_onsite(self, idx_ls=None):
        """
        晶格在位标志矩阵
        """
        n_s = self.n_site

        if idx_ls is None:  # 完整标志
            flag = ssp.eye_array(n_s, format="csc")
        else:  # 指定格点标志
            flag = ssp.csc_array((np.ones_like(idx_ls), (idx_ls, idx_ls)), shape=(n_s, n_s))
        return flag

    def get_flag_hop_px(self, is_bulk=True, is_bound=True, idx_ls=None):
        """
        晶格跃迁标志矩阵
        """
        n_s = self.n_site

        flag = ssp.csc_array(([], ([], [])), shape=(n_s, n_s))
        if idx_ls is None:
            if is_bulk:  # 内部部分
                flag += ssp.eye_array(n_s, k=-1, format="csc")
            if is_bound and self.is_PBC:  # 边界部分
                flag += ssp.eye_array(n_s, k=n_s - 1, format="csc")
        else:  # 指定格点标志
            # 目标点索引
            idx_ls = np.asarray(idx_ls)
            idx_px = idx_ls + 1
            if self.is_PBC:
                idx_px = idx_px % n_s
            flag = ssp.csc_array((np.ones_like(idx_ls), (idx_px, idx_ls)), shape=(n_s, n_s))
        return flag

    def get_flag_hop_p2x(self, is_bulk=True, is_bound=True, idx_ls=None):
        """
        晶格次近邻跃迁标志矩阵
        """
        n_s = self.n_site

        flag = ssp.csc_array(([], ([], [])), shape=(n_s, n_s))
        if idx_ls is None:
            if is_bulk:  # 内部部分
                flag_hop_p2x_bulk = ssp.eye_array(n_s, k=-2, format="csc")
                flag += flag_hop_p2x_bulk
            if is_bound and self.is_PBC:  # 边界部分
                flag_hop_p2x_bound = ssp.eye_array(n_s, k=n_s - 2, format="csc")
                flag += flag_hop_p2x_bound
        else:  # 指定格点标志
            # 目标点索引
            idx_ls = np.asarray(idx_ls)
            idx_p2x = idx_ls + 2
            if self.is_PBC:
                idx_p2x = idx_p2x % n_s
            flag = ssp.csc_array((np.ones_like(idx_ls), (idx_p2x, idx_ls)), shape=(n_s, n_s))
        return flag


class Lattice_Rectangular:
    """
    二维方格点
    """

    def __init__(self, n_site=(10, 10)):
        self.n1, self.n2 = n_site
        self.n_site = self.n1 * self.n2
        self.latt_const = (1.0, 1.0)
        self.is_PBC = (False, False)
        # 格点索引表
        self.tag_ls = self._build_lattice()  # idx -> tag

    def _build_lattice(self):
        """
        构建索引表
        """
        n1_ls, n2_ls = np.meshgrid(np.arange(self.n1), np.arange(self.n2))
        tag_ls = np.transpose((n1_ls.flatten(), n2_ls.flatten()))
        return tag_ls

    def tag2idx(self, tag_ls):
        """
        格点标签 -> 索引
        :param tag_ls: 格点标签(单个或多个)
        :return: 对应格点索引
        """
        idx_ls = np.sum(np.asarray(tag_ls) * (1, self.n1), axis=-1)
        return idx_ls

    def tag2pos(self, tag_ls):
        """
        格点标签 -> 位置
        :param tag_ls: 格点标签(单个或多个)
        :return: 对应格点位置
        """
        pos_ls = np.asarray(tag_ls) * self.latt_const
        return pos_ls

    def _expand_hop_cross_period(self, hop=(1, 0), cross_period=None):
        """
        将 hopping tag 扩展为包含跨周期的 hopping tag 列表
        """
        if cross_period is None:
            cross_period = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        # 边界条件允许的跨周期
        cross_period = np.fromiter(
            filter(lambda xp: (self.is_PBC[0] or xp[0] == 0) and (self.is_PBC[1] or xp[1] == 0), cross_period),
            dtype=np.dtype((int, 2)),
        )
        # 转换为hopping列表
        hop_ls = -cross_period * np.asarray((self.n1, self.n2)) + np.asarray(hop)
        return hop_ls

    def filter_idx_hop_valid_bak(self, hop=(1, 0), cross_period=None, idx_ls=None):
        """
        筛选索引列表中 hopping 后还在系统内的格点
        :param hop: hopping tag
        :param idx_ls: 待筛选的起点索引列表
        :return: 终点索引列表, 起点索引列表
        """
        # 起点索引
        idx0_ls = idx_ls if idx_ls is not None else np.arange(self.n_site)
        # 起点索引 -> 起点标签
        tag0_ls = self.tag_ls[idx0_ls]
        # 跨周期hopping
        hop_ls = self._expand_hop_cross_period(hop, cross_period)

        def filter_hop_idx():
            for hop_ in hop_ls:
                # 起点标签 -> 终点标签
                tag1_ls = tag0_ls + np.asarray(hop_)

                for tag1, idx0 in zip(tag1_ls, idx0_ls):
                    # 判断终点是否在系统内
                    if 0 <= tag1[0] < self.n1 and 0 <= tag1[1] < self.n2:
                        # 终点标签 -> 终点索引
                        idx1 = self.tag2idx(tag1)
                        # 返回索引
                        yield idx1, idx0

        idx1_ls, idx0_ls = np.fromiter(filter_hop_idx(), dtype=np.dtype((int, 2))).T
        return idx1_ls, idx0_ls

    def filter_idx_hop_valid(self, hop=(1, 0), cross_period=None, idx_ls=None):
        """
        筛选索引列表中 hopping 后还在系统内的格点
        :param hop: hopping tag
        :param idx_ls: 待筛选的起点索引列表
        :return: 终点索引列表, 起点索引列表
        """
        # 起点索引
        idx0_ls = idx_ls if idx_ls is not None else np.arange(self.n_site)
        # 起点索引 -> 起点标签
        tag0_ls = self.tag_ls[idx0_ls]
        # 跨周期hopping
        hop_ls = self._expand_hop_cross_period(hop, cross_period)

        def idx_of_hop(hop_):
            # 起点标签 -> 终点标签
            tag1_ls = tag0_ls + np.asarray(hop_)
            # 判断终点是否在系统内
            is_in_syst = (
                (0 <= tag1_ls[:, 0]) & (tag1_ls[:, 0] < self.n1) & (0 <= tag1_ls[:, 1]) & (tag1_ls[:, 1] < self.n2)
            )
            # 标签 -> 索引
            idx0_ls_ = self.tag2idx(tag0_ls[is_in_syst])
            idx1_ls_ = self.tag2idx(tag1_ls[is_in_syst])
            return idx1_ls_, idx0_ls_

        idx1_ls, idx0_ls = np.concatenate([idx_of_hop(hop_) for hop_ in hop_ls], axis=-1)
        return idx1_ls, idx0_ls

    def get_flag_onsite(self, idx_ls=None):
        """
        晶格在位标志矩阵
        """
        n_s = self.n_site

        if idx_ls is None:  # 完整标志
            flag = ssp.eye_array(n_s, format="csc")
        else:  # 指定格点标志
            flag = ssp.csc_array((np.ones_like(idx_ls), (idx_ls, idx_ls)), shape=(n_s, n_s))
        return flag

    def get_flag_hop(self, hop=(1, 0), cross_period=None):
        """
        hopping 标志矩阵
        """
        # 系统内hopping的终起点索引
        idx1_ls, idx0_ls = self.filter_idx_hop_valid(hop, cross_period)
        # 标志矩阵
        flag = ssp.csc_array((np.ones_like(idx0_ls), (idx1_ls, idx0_ls)), shape=(self.n_site, self.n_site))
        return flag
