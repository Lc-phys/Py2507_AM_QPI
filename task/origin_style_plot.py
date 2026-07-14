### origin plotting style
import matplotlib as mpl

# from matplotlib import pyplot as plt
import numpy as np
import warnings
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection


def set_rc_origin(ver=None):

    if ver is None:
        ### LINES
        mpl.rcParams["lines.linewidth"] = 2.0  # line width in points
        mpl.rcParams["lines.markersize"] = 5  # markersize, in points

        ### LATEX
        mpl.rcParams["mathtext.fontset"] = (
            "stixsans"  # 'dejavusans', 'dejavuserif', 'cm' (Computer Modern), 'stix', 'stixsans'
        )

        ### FONT
        # note that font.size controls default text sizes.  To configure
        # special text sizes tick labels, axes, labels, title, etc, see the rc
        # settings for axes and ticks. Special text sizes can be defined
        # relative to font.size, using the following values: xx-small, x-small,
        # small, medium, large, x-large, xx-large, larger, or smaller
        mpl.rcParams["font.size"] = 24.0

        ### AXES
        mpl.rcParams["axes.linewidth"] = 1.5  # edge linewidth
        # mpl.rcParams["axes.titlelocation"] = "left"
        # mpl.rcParams["axes.titlesize"] = "large"  # fontsize of the axes title
        # mpl.rcParams["axes.titlepad"] = 10  # pad between axes and title in points
        mpl.rcParams["axes.labelsize"] = "large"  # fontsize of the x any y labels
        mpl.rcParams["axes.xmargin"] = 0  # x margin.  See `axes.Axes.margins`
        # use scientific notation if log10 of the axis range is smaller than the first or larger than the second
        mpl.rcParams["axes.formatter.limits"] = -3, 4
        mpl.rcParams["axes.formatter.use_mathtext"] = True  # When True, use mathtext for scientific notation.

        ### TICKS
        mpl.rcParams["xtick.top"] = True  # draw ticks on the top side
        mpl.rcParams["xtick.major.size"] = 8  # major tick size in points
        mpl.rcParams["xtick.minor.size"] = 4  # minor tick size in points
        mpl.rcParams["xtick.major.width"] = 1.5  # major tick width in points
        mpl.rcParams["xtick.minor.width"] = 1.5  # minor tick width in points
        # mpl.rcParams["xtick.labelsize"] = "medium"  # font size of the tick labels
        mpl.rcParams["xtick.direction"] = "in"  # direction: in, out, or inout
        mpl.rcParams["xtick.minor.visible"] = True  # visibility of minor ticks on x-axis

        mpl.rcParams["ytick.right"] = True  # draw ticks on the right side
        mpl.rcParams["ytick.major.size"] = 8  # major tick size in points
        mpl.rcParams["ytick.minor.size"] = 4  # minor tick size in points
        mpl.rcParams["ytick.major.width"] = 1.5  # major tick width in points
        mpl.rcParams["ytick.minor.width"] = 1.5  # minor tick width in points
        # mpl.rcParams["ytick.labelsize"] = "medium"  # font size of the tick labels
        mpl.rcParams["ytick.direction"] = "in"  # direction: in, out, or inout
        mpl.rcParams["ytick.minor.visible"] = True  # visibility of minor ticks on y-axis

        ### Legend
        mpl.rcParams["legend.fontsize"] = "small"
        mpl.rcParams["legend.fancybox"] = False
        mpl.rcParams["legend.edgecolor"] = "0.2"
        #
        mpl.rcParams["legend.fancybox"] = (
            False  # if True, use a rounded box for the legend background, else a rectangle
        )
        mpl.rcParams["legend.framealpha"] = "0.0"  # legend patch transparency
        mpl.rcParams["legend.edgecolor"] = "1.0"  # background patch boundary color

        ### FIGURE
        mpl.rcParams["figure.figsize"] = 12, 10  # figure size in inches
        mpl.rcParams["figure.dpi"] = 60  # figure dots per inch
        mpl.rcParams["figure.constrained_layout.use"] = (
            True  # When True, automatically make plot elements fit on the figure.
        )
        ## Padding (in inches) around axes; defaults to 3/72 inches, i.e. 3 points.
        # figure.constrained_layout.h_pad:  0.04167
        # figure.constrained_layout.w_pad:  0.04167
        ## Spacing between subplots, relative to the subplot sizes.  Much smaller than for
        ## tight_layout (figure.subplot.hspace, figure.subplot.wspace) as constrained_layout
        ## already takes surrounding texts (titles, labels, # ticklabels) into account.
        mpl.rcParams["figure.constrained_layout.hspace"] = 0.04
        mpl.rcParams["figure.constrained_layout.wspace"] = 0.02

        ### SAVING FIGURES
        mpl.rcParams["savefig.dpi"] = 300  # figure dots per inch or 'figure'
        mpl.rcParams["savefig.bbox"] = (
            "tight"  # {tight, standard} 'tight' is incompatible with generating frames for animation
        )
        mpl.rcParams["savefig.pad_inches"] = 0.1  # padding to be used, when bbox is set to 'tight'

    elif ver in {"note"}:
        mpl.rcParams["lines.linewidth"] = 1.5
        mpl.rcParams["lines.markersize"] = 4
        mpl.rcParams["font.size"] = 18.0
        mpl.rcParams["axes.linewidth"] = 1.2
        mpl.rcParams["axes.labelsize"] = "large"
        # mpl.rcParams["axes.formatter.limits"] = -4, 5
        mpl.rcParams["axes.formatter.use_mathtext"] = True
        mpl.rcParams["xtick.major.size"] = 8
        mpl.rcParams["xtick.minor.size"] = 4
        mpl.rcParams["xtick.major.width"] = 1.2
        mpl.rcParams["xtick.minor.width"] = 1.2
        mpl.rcParams["xtick.minor.visible"] = True
        mpl.rcParams["ytick.major.size"] = 8
        mpl.rcParams["ytick.minor.size"] = 4
        mpl.rcParams["ytick.major.width"] = 1.2
        mpl.rcParams["ytick.minor.width"] = 1.2
        mpl.rcParams["ytick.minor.visible"] = True
        mpl.rcParams["legend.fontsize"] = "small"
        # mpl.rcParams["legend.fancybox"] = False
        mpl.rcParams["legend.edgecolor"] = "0.2"
        mpl.rcParams["figure.figsize"] = 12, 10
        mpl.rcParams["figure.dpi"] = 96
        mpl.rcParams["figure.constrained_layout.use"] = True
        mpl.rcParams["savefig.dpi"] = 150
        mpl.rcParams["savefig.bbox"] = "tight"
        mpl.rcParams["savefig.pad_inches"] = 0.1
        mpl.rcParams["figure.constrained_layout.hspace"] = 0.05
        # mpl.rcParams["figure.constrained_layout.wspace"] = 0.02


def number_subplots(ax_ls=None, offset=(-0.4, +0.5), in_layout=[0]):
    if ax_ls is None:
        ax_ls = [ax for ax in mpl.pyplot.gcf().axes if ax._label != "<colorbar>"]
    if not isinstance(offset[0], (list, tuple)):
        offset = [offset] * len(ax_ls)

    for idx, ax in enumerate(ax_ls):
        num_anno = ax.annotate(
            "(" + chr(idx + 97) + ")",
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=offset[idx],
            textcoords="offset fontsize",
            size="large",
        )
        if idx not in in_layout:
            num_anno.set_in_layout(False)


def cmap(cname):
    # diverging colormaps
    if cname == "青蓝黑橙黄":
        rgb_ls = np.asarray([(132, 255, 255), (41, 121, 255), (0, 0, 0), (255, 145, 0), (255, 255, 141)]) / 255
        inr_ls = [0, 0.25, 0.5, 0.75, 1]
    elif cname == "自定义蓝青白黄橙":
        rgb_ls = np.asarray([(41, 121, 255), (132, 255, 255), (255, 255, 255), (255, 255, 141), (255, 145, 0)]) / 255
        inr_ls = [0, 0.25, 0.5, 0.75, 1]
    elif cname == "自定义紫白橙":
        rgb_ls = np.asarray([(124, 77, 255), (179, 136, 255), (255, 255, 255), (255, 215, 64), (255, 145, 0)]) / 255
        inr_ls = [0, 0.25, 0.5, 0.75, 1]
    elif cname == "蓝白橙":  # 取自 set3
        rgb_ls = np.asarray([(128, 177, 211), (255, 255, 255), (253, 180, 98)]) / 255
        inr_ls = [0, 0.5, 1]
    elif cname == "蓝白橙2":  # 超轻, 取自 paired
        rgb_ls = np.asarray([(166, 206, 227), (255, 255, 255), (253, 191, 111)]) / 255
        inr_ls = [0, 0.5, 1]
    elif cname == "紫白橙":  # 超轻, 取自 paired
        rgb_ls = np.asarray([(202, 178, 214), (255, 255, 255), (253, 191, 111)]) / 255
        inr_ls = [0, 0.5, 1]
    elif cname == "绿白橙":  # 超轻, 取自 paired
        rgb_ls = np.asarray([(178, 223, 138), (255, 255, 255), (253, 191, 111)]) / 255
        inr_ls = [0, 0.5, 1]
    elif cname == "绿白粉":  # 超嫩, 取自 paired/set3
        rgb_ls = np.asarray([(178, 223, 138), (255, 255, 255), (252, 205, 229)]) / 255
        inr_ls = [0, 0.5, 1]
    # sequential colormaps
    elif cname == "白紫":
        rgb_ls = np.asarray([(255, 255, 255), (49, 27, 146)]) / 255
        inr_ls = [0, 1]
    elif cname == "自定义白紫2":
        rgb_ls = np.asarray([(255, 255, 255), (253, 174, 107), (230, 85, 13)]) / 255
        inr_ls = [0, 0.5, 1]
    elif cname == "自定义蓝白":
        rgb_ls = np.asarray([(31, 119, 180), (152, 223, 138), (255, 255, 179)]) / 255
        inr_ls = [0, 0.5, 1]
    else:
        raise ValueError("Color Map NOT Found!!")

    rd_ls = np.transpose([inr_ls, rgb_ls[:, 0], rgb_ls[:, 0]])
    gn_ls = np.transpose([inr_ls, rgb_ls[:, 1], rgb_ls[:, 1]])
    bu_ls = np.transpose([inr_ls, rgb_ls[:, 2], rgb_ls[:, 2]])
    cdict = {"red": rd_ls, "green": gn_ls, "blue": bu_ls}
    cmap = LinearSegmentedColormap(cname, cdict)
    return cmap


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)
