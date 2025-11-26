# for add_scale_bar()
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm


# add scale bar to an axis
def add_scale_bar(ax, pixel_size, length_m, label, loc='lower right', color='white'):
    """
    Add a simple scale bar to a matplotlib axis using AnchoredSizeBar.

    [inputs]
    ax : matplotlib.axes.Axes
        Axis to which the scale bar will be added.
    pixel_size : float
        Physical size of one pixel (meters).
    length_m : float
        Physical length represented by the scale bar (meters).
    label : str
        Text label to display next to the bar (e.g. '0.5 mm').
    loc : str
        Location string for AnchoredSizeBar (e.g. 'lower right').
    color : str
        Color of the scale bar and label text.
    """

    # Convert physical length to pixels
    length_px = length_m / pixel_size

    # Font properties for the label
    fontprops = fm.FontProperties(size=8)

    # Create the scale bar artist
    scalebar = AnchoredSizeBar(
        ax.transData,        # transform: data coordinates
        length_px,           # length in data (pixels)
        label,               # scale bar label
        loc,                 # location
        pad=0.2,             # padding around the bar
        color=color,         # bar color
        frameon=False,       # no surrounding box
        size_vertical=length_px * 0.02,  # bar thickness in pixels
        fontproperties=fontprops
    )

    # Add the scale bar to the axis
    ax.add_artist(scalebar)
