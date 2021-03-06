import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
from matplotlib.colors import LinearSegmentedColormap as lsc
# from numpy import array
import numpy as np


def list_custom_cmaps():
    import glob, re
    from seren3 import config

    _REGEX = ".+?(?=_256)"

    cmap_dir = "%s/cmaps/" % config.get("data", "data_dir")
    files = glob.glob("%s/*.txt" % cmap_dir)

    cmap_names = []
    for f in files:
        m = re.search(_REGEX, f)
        cmap_name = m.group(0).split("/")[-1]
        cmap_names.append(cmap_name)

    return cmap_names


def load_custom_cmaps(cmap_name):
    import numpy as np
    import glob, re
    from seren3 import config

    if (cmap_name == "parula"):
        from seren3.utils import parula
        return parula.parula_map

    cmap_dir = "%s/cmaps/" % config.get("data", "data_dir")

    _REGEX = ".+?(?=_256)"

    files = glob.glob("%s/*.txt" % cmap_dir)
    for f in files:
        m = re.search(_REGEX, f)
        name = m.group(0).split("/")[-1]

        if cmap_name == name:
            C = np.loadtxt(f)
            cm = matplotlib.colors.ListedColormap(C/255.0)
            return cm

    raise IOError("No cmap found with name: %s" % cmap_name)


def velocity_plot(s, width='15 kpc', cmap1='hot', cmap2='viridis', stream=False):
    import pynbody
    import pynbody.plot.sph as sph
    import matplotlib.pylab as plt
    f, axs = plt.subplots(1,2,figsize=(14,6))
    pynbody.analysis.angmom.faceon(s.g)
    sph.velocity_image(s.g, qty='temp', width=width, cmap=cmap1, vector_color='cyan', denoise=True, vector_resolution=20, subplot=axs[0], show_cbar=False)

    mode = 'stream' if stream else 'quiver'
    pynbody.analysis.angmom.faceon(s.g)
    sph.velocity_image(s.g, width=width, cmap=cmap2, vector_color='black', denoise=True, units='Msol kpc^-2', mode=mode, vector_resolution=20, subplot=axs[1], show_cbar=False)

# def cmap_map(function, cmap):
#     """ Applies function (which should operate on vectors of shape 3:
#     [r, g, b], on colormap cmap. This routine will break any discontinuous     points in a colormap.
#     """
#     cdict = cmap._segmentdata
#     step_dict = {}
#     # Firt get the list of points where the segments start or end
#     for key in ('red','green','blue'):         step_dict[key] = map(lambda x: x[0], cdict[key])
#     step_list = sum(step_dict.values(), [])
#     step_list = array(list(set(step_list)))
#     # Then compute the LUT, and apply the function to the LUT
#     reduced_cmap = lambda step : array(cmap(step)[0:3])
#     old_LUT = array(map( reduced_cmap, step_list))
#     new_LUT = array(map( function, old_LUT))
#     # Now try to make a minimal segment definition of the new LUT
#     cdict = {}
#     for i,key in enumerate(('red','green','blue')):
#         this_cdict = {}
#         for j,step in enumerate(step_list):
#             if step in step_dict[key]:
#                 this_cdict[step] = new_LUT[j,i]
#             elif new_LUT[j,i]!=old_LUT[j,i]:
#                 this_cdict[step] = new_LUT[j,i]
#         colorvector=  map(lambda x: x + (x[1], ), this_cdict.items())
#         colorvector.sort()
#         cdict[key] = colorvector
#
#     return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

def cmap_map(function, cmap, name='colormap_mod', N=None, gamma=None):
    """
    Modify a colormap using `function` which must operate on 3-element
    arrays of [r, g, b] values.

    You may specify the number of colors, `N`, and the opacity, `gamma`,
    value of the returned colormap. These values default to the ones in
    the input `cmap`.

    You may also specify a `name` for the colormap, so that it can be
    loaded using plt.get_cmap(name).
    """
    if N is None:
        N = cmap.N
    if gamma is None:
        gamma = cmap._gamma
    cdict = cmap._segmentdata
    # Cast the steps into lists:
    step_dict = {key: map(lambda x: x[0], cdict[key]) for key in cdict}
    # Now get the unique steps (first column of the arrays):
    step_list = np.unique(sum(step_dict.values(), []))
    # 'y0', 'y1' are as defined in LinearSegmentedColormap docstring:
    y0 = cmap(step_list)[:, :3]
    y1 = y0.copy()[:, :3]
    # Go back to catch the discontinuities, and place them into y0, y1
    for iclr, key in enumerate(['red', 'green', 'blue']):
        for istp, step in enumerate(step_list):
            try:
                ind = step_dict[key].index(step)
            except ValueError:
                # This step is not in this color
                continue
            y0[istp, iclr] = cdict[key][ind][1]
            y1[istp, iclr] = cdict[key][ind][2]
    # Map the colors to their new values:
    y0 = np.array(map(function, y0))
    y1 = np.array(map(function, y1))
    # Build the new colormap (overwriting step_dict):
    for iclr, clr in enumerate(['red', 'green', 'blue']):
        step_dict[clr] = np.vstack((step_list, y0[:, iclr], y1[:, iclr])).T
    return lsc(name, step_dict, N=N, gamma=gamma)


def add_colorbar(im):
    fig = plt.gcf(); ax = plt.gca()
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    return fig.colorbar(im, cax=cax);


def force_aspect(ax, aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def ncols(n, cmap='rainbow'):
    import matplotlib.pylab as plt
    cmap = plt.get_cmap(cmap)
    return cmap(np.linspace(0, 1, n))

    