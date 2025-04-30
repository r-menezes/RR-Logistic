"""
@author: Rafael Menezes (github: r-menezes)
@date: 2025
@license: MIT License
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from figs_settings import *

# FIGURE 2 - CARRYING CAPACITY VS. HOME RANGE AREA

def plot_carrCap_simulated_values(
        df,
        xvar,
        hue_var,
        yvar='N*',
        stripplot=False,
        SS_pointplot_kws={},
        BM_pointplot_kws={},
        strip_kws={},
        axs=None,
        title='Movement gradient',
        figsize=(8,6),
        MFpred=None,
        spatLogisticPred=None,
        BM_pad=.1,
        SS_pad=.05,
        OU_pad=.05,
        pi_scale = 95,
        err_style='bars',
        err_kws={},
        savefig=None,
        palette='deep',
        ):

    # SETUP
    #  initialize figure and axes
    if axs is None:
        fig, (axSS, axOU, axBM) = plt.subplots(1, 3,
                                        sharey = True,
                                        gridspec_kw={'width_ratios': [0.05, 0.8, 0.15], 'wspace':0.02},
                                        figsize=figsize)
    else:
        if len(axs) != 3:
            raise ValueError('axs must be a list of length 3')
        axSS, axOU, axBM = axs
        fig = axSS.get_figure()

    #  stripplot
    if stripplot is True:
        _strip_kws = {
            "alpha":0.15,
            "marker":'$\\circ$',
            "s":6,
            "linewidth":1,
            "zorder": 1,
            "jitter": 0.2,
            "palette": palette,
            "legend": False,
            "edgecolor":'face',
            }
        
        _strip_kws = _strip_kws | strip_kws
        strip_kws = _strip_kws
    
    pointplot_kws = {
        "palette": palette,
        "estimator": np.median,
        "errorbar": ('pi', pi_scale),
        "err_kws": {'linewidth': .8, 'solid_joinstyle':'round', 'solid_capstyle':'round'},
        "capsize": .1,
        "markersize": .8,
        "markers": '+',
        }


    # sorted unique hue values
    unique_hue_vals = np.sort(df[hue_var].unique())
    hue_idx = {hue:i for i, hue in enumerate(unique_hue_vals)}
    
    # SS
    dfSS = df[df['mov'] == 'SS']

    # get x values
    _xvals = np.ones(len(dfSS[dfSS['mov'] == 'SS']))
    strip_kws['jitter'] = SS_pad

    if stripplot is True:
        axSS = sns.stripplot(data=dfSS,
                        x=_xvals,
                        y=yvar,
                        ax = axSS,
                        hue = hue_var,
                        **strip_kws)

    axSS = sns.pointplot(data=dfSS,
                        x=_xvals,
                        y=yvar,
                        ax = axSS,
                        hue=hue_var,
                        **pointplot_kws)
    
    if axSS.get_legend() is not None:
        axSS.get_legend().remove()
    axSS.set_title("SS")
    axSS.set_ylabel(r"Carrying Capacity - $\left\langle N \right\rangle/K_{CSR}$")
    axSS.set_xticks([])
    axSS.set_xlabel("")

    # get xlims for axBM
    # since axBM is a series of boxplots, the xlims need to be taken from the lines using get_lines() and get_xdata()
    # the list comprehension is used to flatten the list of lists
    xdata = np.array([x for line in axSS.get_lines() for x in line.get_xdata()])
    # filter out the nan values
    xdata = xdata[~np.isnan(xdata)]
    xmin = xdata.min()
    xmax = xdata.max()
    SS_xlims = (xmin-SS_pad, xmax+SS_pad)
    axSS.set_xlim(SS_xlims)

    # OU
    axOU.set_xscale('log')

    if stripplot is True:
        axOU = sns.stripplot(data=df[df['mov'] == 'OU'],
                            x=xvar,
                            y=yvar,
                            ax=axOU,
                            hue=hue_var,
                            native_scale=True,
                            **strip_kws)
    
    if err_style == 'bars':
        for idx, (color, hue) in enumerate(zip(palette, unique_hue_vals)):
            _data = df[(df['mov'] == 'OU') & (df[hue_var] == hue)]
            _x = _data[xvar]
            _y = _data[yvar]
            sns.lineplot(data=_data,
                        x=xvar,
                        y=yvar,
                        ax=axOU,
                        color=color,
                        dashes=['', (3,1,1,1), (1,1)][idx],
                        estimator=np.median,
                        errorbar=('pi', pi_scale),
                        err_style=err_style,
                        label=hue,
                        err_kws=err_kws | {'capsize': 2.2 - 0.55*idx, 'capthick':1.2 - 0.3*idx},
                        linewidth=1.5)

    else:
        axOU = sns.lineplot(data=df[(df[hue_var].notna()) & (df['mov'] == 'OU')],
                            x=xvar,
                            y=yvar,
                            ax=axOU,
                            hue=hue_var,
                            style=hue_var,
                            dashes=['', (3,1,1,1), (1,1)],
                            estimator=np.median,
                            errorbar=('pi', pi_scale),
                            err_style=err_style,
                            err_kws=err_kws,
                            solid_joinstyle='round',
                            solid_capstyle='round',
                            dash_joinstyle='round',
                            # dash_capstyle='round',
                            palette=palette)

    # Legends and labels
    if axOU.get_legend() is not None:
        axOU.get_legend().remove()
    axOU.set_title(f"OU")
    axOU.set_xlabel(r'Home Range Size - $\sigma$')

    # BM
    # determine the length of the filtered dataframe to construct a constant numpy vector
    # This constant vector will be used as the x-variable for the BM plot
    # The length of the vector is the number of rows in the filtered dataframe
    dfBM = df[df['mov'] == 'BM']
    
    _xvals = np.array([hue_idx[hue] for hue in dfBM[hue_var]])

    strip_kws['jitter'] = 0.3

    if stripplot is True:
        axBM = sns.stripplot(data=dfBM,
                            x=_xvals,
                            y=yvar,
                            ax = axBM,
                            hue=hue_var,
                            # dodge=True,
                            **strip_kws)

    pointplot_kws = pointplot_kws | {
        "err_kws": {'linewidth': .8},
        "capsize": 1.,
        "markersize": .8,
        }

    pointplot_kws.pop('markers')

    axBM = sns.pointplot(data=dfBM,
                            x=_xvals,
                            y=yvar,
                            ax = axBM,
                            hue=hue_var,
                            markers='+',
                            **pointplot_kws)

    if axBM.get_legend() is not None:
        axBM.get_legend().remove()
    axBM.set_title("BM")
    axBM.set_ylabel("")
    axBM.set_xticks([])
    axBM.set_xlabel("")

    # get xlims for axBM
    # since axBM is a series of boxplots, the xlims need to be taken from the lines using get_lines() and get_xdata()
    # the list comprehension is used to flatten the list of lists
    try:
        xdata = np.array([x for line in axBM.get_lines() for x in line.get_xdata()])
        # filter out the nan values
        xdata = xdata[~np.isnan(xdata)]
        xmin = xdata.min()
        xmax = xdata.max()
        BM_xlims = (xmin-BM_pad, xmax+BM_pad)
        
    except ValueError:
        BM_xlims = (-BM_pad, 1.+BM_pad)
        
    axBM.set_xlim(BM_xlims)

    # Predictions
    # Mean-field
    if MFpred is not None:
        axSS.hlines([MFpred], *axSS.get_xlim(), linestyles='--',color='k', alpha=0.5, zorder=0)
        axOU.hlines([MFpred], *axOU.get_xlim(), linestyles='--', color='k', label='CSR', alpha=0.5, zorder=0)
        axBM.hlines([MFpred], *axBM.get_xlim(), linestyles='--',color='k', alpha=0.5, zorder=0)

    # GENERAL AESTHETICS

    # ticks
    #  remove the ticks from the y-axis of the OU and BM plots
    #  using set_visible
    axOU.tick_params(axis='y', which='both', left=False, labelleft=False)
    
    # custom xticks
    # insert an xtick at the middle of the axis
    axSS.set_xticks([0.])
    axSS.set_xticklabels(['0'])
    
    axBM.set_xticks([1.])
    axBM.set_xticklabels([r'$\infty$'], fontsize=14)
    
    # grid
    #  set the y grid for all plots
    axSS.grid(axis='y', alpha=0.4)
    axOU.grid(axis='y', alpha=0.4)
    axBM.grid(axis='y', alpha=0.4)

    # legend
    # create legend based on elements of axOU in the upper right corner of the figure
    # the legend is created using the hue variable
    axOU.legend(loc='upper right',
            bbox_to_anchor=(1.2, 1.0),
            # smaller legend
            fontsize=12,
            facecolor=axOU.get_facecolor(),
            edgecolor=axOU.get_facecolor(),
            title='Dispersal')

    # set zorder of axOU to be above axSS
    axOU.set_zorder(axBM.get_zorder()+1)
    axOU.patch.set_visible(False)

    # Broken axis
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html#broken-axis
    # We create line objects in axes coordinates, in which (0,0), (0,1),
    # (1,0), and (1,1) are the four corners of the axes.
    # The slanted lines themselves are markers at those locations, such that the
    # lines keep their angle and position, independent of the axes size or scale
    # Finally, we need to disable clipping.
    d = 2.  # proportion of vertical to horizontal extent of the slanted line
    kwargs = {
        "marker": [(-1, -d), (1, d)],
        "markersize": 6,
        "linestyle": "none",
        "color":'k',
        "mec":'k',
        "mew":1,
        "clip_on":False
        }
    
    dx = 0.0005
    axSS.plot([1, 1], [0, 1], transform=axSS.transAxes, **kwargs)
    axOU.plot([0-dx, 1, 0-dx, 1], [0, 0, 1, 1], transform=axOU.transAxes, **kwargs)
    axBM.plot([0-dx, 0-dx], [0, 1], transform=axBM.transAxes, **kwargs)

    # Despine
    sns.despine(ax=axSS, right=True, top=False)
    sns.despine(ax=axOU, left=True, right=True, top=False)
    sns.despine(ax=axBM, left=True, right=False, top=False)

    fig.suptitle(title)

    if savefig is not None:
        fig.savefig(savefig)

    return fig, (axSS, axOU, axBM)


def fig2(data, save=None):
    '''
    Plot normalized carrying capacity vs. home-range size for different movement models and dispersal ranges

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the data to be plotted
    save : str, optional
        Path to save the figure, by default None
    '''
    df = data.copy()

    width = 100*mm_to_inches
    height = 60*mm_to_inches

    # generate the figure
    fig = plt.figure(figsize=(width, height))

    # generate the subplots
    gs = mpl.gridspec.GridSpec(1, 3, width_ratios=[.4, 8.4, 1.2], wspace=0.02, hspace=0.)

    axSS0 = fig.add_subplot(gs[0])
    axOU0 = fig.add_subplot(gs[1], sharey=axSS0)
    axBM0 = fig.add_subplot(gs[2], sharey=axSS0)

    # colors
    from palettable.cmocean.sequential import Turbid_3
    turbid_cmap = Turbid_3.get_mpl_colormap()

    # error kws
    _err_kws = {'capsize':1.1, 'elinewidth':.8, 'capthick':.8, 'solid_joinstyle':'round', 'solid_capstyle':'round', 'dash_joinstyle':'round', 'dash_capstyle':'round'}

    # plot the carrCap measurements
    fig, (axSS0,axOU0,axBM0) = plot_carrCap_simulated_values(df,
                                    'HRarea',
                                    'disp',
                                    yvar='N*',
                                    MFpred=[1.],
                                    BM_pad=.3,
                                    pi_scale=90,
                                    palette=list(turbid_cmap([.2, .5, 1.])),
                                    strip_kws={'alpha': 0.08},
                                    axs=(axSS0, axOU0, axBM0),
                                    title='',
                                    err_style='bars',
                                    err_kws=_err_kws,
                                    stripplot=False)
                                    

    # remove legend from axOU
    axOU0.legend().remove()

    # remove y labels
    axOU0.set_ylabel('')
    axBM0.set_ylabel('')

    # remove x labels
    axOU0.set_xlabel('')

    # set x ticks as invisible
    axSS0.xaxis.set_tick_params(labelbottom=False)
    axOU0.xaxis.set_tick_params(labelbottom=True)
    axBM0.xaxis.set_tick_params(labelbottom=False)

    # set y tick labels invisible
    axBM0.yaxis.set_tick_params(labelright=False)

    # Set ticks
    _major_ticks = np.logspace(-6, 1, 8)
    _minor_ticks = []
    for t in _major_ticks:
        for i in range(2, 10):
            _minor_ticks.append(t*i)

    axOU0.xaxis.set_major_locator(mpl.ticker.FixedLocator(_major_ticks))
    axOU0.xaxis.set_minor_locator(mpl.ticker.FixedLocator(_minor_ticks))
    axOU0.xaxis.set_ticklabels(['', '', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^{0}$', '$10^{1}$'])

    _major_ticks = [0,1,2]
    _minor_ticks = [0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2.25, 2.5]
    axSS0.yaxis.set_major_locator(mpl.ticker.FixedLocator(_major_ticks))
    axSS0.yaxis.set_minor_locator(mpl.ticker.FixedLocator(_minor_ticks))
    axSS0.yaxis.set_ticklabels([r'$0$', r'$1$', r'$2$'])

    # show ticks
    axOU0.tick_params(axis='x', which='major', bottom=True, top=False, direction='in', length=3, width=0.7, color=[0,0,0,0.3])
    axOU0.tick_params(axis='x', which='minor', bottom=True, top=False, direction='in', length=3, width=0.3, color=[0,0,0,0.3])
    axSS0.tick_params(axis='y', which='major', left=True, right=False, direction='in', length=5, width=0.7, color=[0,0,0,0.3])
    axSS0.tick_params(axis='y', which='minor', left=True, right=False, direction='in', length=5, width=0.3, color=[0,0,0,0.3])
    axBM0.tick_params(axis='y', which='both', left=False, right=False)

    # Adjust limits
    axSS0.set_ylim(-0.2, 2.4)

    # remove grid
    axSS0.grid(False)
    axOU0.grid(False)
    axBM0.grid(False)

    # add text inside the plot
    expr = r'$n^* =  \frac{n_{t\to\infty}}{n^*_{\text{CSR}}}$'
    axOU0.text(.9, 0.85, expr, transform=axOU0.transAxes, ha='right', va='center', fontsize=8, color='k')

    # legend
    axOU0.legend(loc='center right', bbox_to_anchor=(1.15, 0.25), fontsize=7, title='', frameon=False, fancybox=False)
    leg_format = lambda x: r'$\sigma_d = $ ' + magnitude_format(x) if x != 'CSR' else r'$n^*_{\text{CSR}}$'
    for t in axOU0.get_legend().texts:
        t.set_text(leg_format(t.get_text()))

    # XY Labels
    axSS0.set_ylabel(r'Carrying Capacity, $n^*$', fontsize=8, labelpad=6)
    axOU0.set_xlabel(r'Home Range Area - $A_{\text{HR}}$', va='center', fontsize=8, labelpad=8)

    plt.subplots_adjust(bottom=0.18)  # Adjust the bottom margin
    
    # save
    # save figure png, svg, pdf
    if save:
        my_savefig(fig, save)

    return fig


# FIGURE 3 - HEATMAPS

def generate_heatmap(data=None,
                index='disp',
                columns='hr',
                values='N*',
                cmap=default_cmap,
                norm=default_norm,
                cont_kwargs={},
                interp_multiplier=4,
                levels=np.geomspace(1e-1,1e1,9),
                gaussian_noise=4.,
                min_tresh=1e-3,
                contour_line=True,
                **kwargs):
    
    from scipy.signal import savgol_filter

    ax = plt.gca()

    pivot = data[[index, columns, values]].pivot_table(index=index, columns=columns, values=values)
    expanded = pivot.to_numpy()
    original = expanded.copy()

    # SAVGOL FILTER
    # # Apply log transformation to non-zero values
    # log_abundance = np.log1p(expanded)
    
    # # Smooth the log-transformed data
    # window_length=5
    # poly_order=3
    # smoothed_log = savgol_filter(log_abundance, window_length, poly_order)
    
    # # Transform back to linear scale
    # expanded = np.expm1(smoothed_log)

    # GAUSSIAN NOISE
    if gaussian_noise > 0:
        from scipy.ndimage import gaussian_filter
        expanded = gaussian_filter(input=expanded, sigma=gaussian_noise)

    # INTERPOLATION
    if interp_multiplier > 1:
        from scipy.interpolate import griddata, interpn

        # get the original x values and augment them
        x_orig_vals = data[index].unique().astype(float)
        x_orig_vals.sort()
        x_n_orig_vals = len(x_orig_vals)
        x_interp_vals = np.geomspace(x_orig_vals.min(), x_orig_vals.max(), int(x_n_orig_vals*interp_multiplier - 1))
        
        # get the original y values and augment them
        y_orig_vals = data[index].unique().astype(float)
        y_orig_vals.sort()
        y_n_orig_vals = len(y_orig_vals)
        y_interp_vals = np.geomspace(y_orig_vals.min(), y_orig_vals.max(), int(y_n_orig_vals*interp_multiplier - 1))
        
        # generate grid
        grid_x, grid_y = np.meshgrid(x_interp_vals, y_interp_vals)

        # use interpn to interpolate the values
        expanded = interpn(points=(x_orig_vals, y_orig_vals),
                            values=expanded,
                            xi=(grid_y, grid_x),
                        method="cubic")

    else:
        x_interp_vals = data[columns].unique().astype(float)
        x_interp_vals.sort()
        y_interp_vals = data[index].unique().astype(float)
        y_interp_vals.sort()

    # MASKING
    if min_tresh is not None:
        expanded[expanded < min_tresh] = 0.

    # PLOTTING

    # colors
    if levels is not None:

        # remove color and noise from kwargs
        _ = kwargs.pop('color', None)
        _ = kwargs.pop('noise', None)

        contours = ax.contourf(x_interp_vals, y_interp_vals, expanded,
                    levels = levels,
                    cmap = cmap,
                    norm = norm,
                    origin = 'lower',
                    extend='both',
                    **kwargs
                    )

        # extinction contour
        # fill in the color for the masked region
        # get mask
        mask = (expanded >= min_tresh)
        ma = np.ma.masked_array(np.ones_like(expanded), mask)
        
        # fill in the masked region
        ax.contourf(x_interp_vals, y_interp_vals, ma, levels=[0, 1], colors='#CA9C9E')

        # limits
        ax.set_xlim(x_interp_vals.min(), x_interp_vals.max())
        ax.set_ylim(y_interp_vals.min(), y_interp_vals.max())

        # scale
        ax.set_xscale('log')
        ax.set_yscale('log')

    else:
        # get the original x values and augment them
        # generate grid
        grid_x, grid_y = np.meshgrid(x_interp_vals, y_interp_vals)

        # use interpn to interpolate the values
        # plot the heatmap with the continuous colormap
        # aux_df = data.pivot(index=index, columns=columns, values=values)

        # contours = sns.heatmap(aux_df,
        #                 cmap=cmap,
        #                 norm=norm,
        #                 ax=ax,
        #                 cbar=False,)

        # # remove color from kwargs
        _ = kwargs.pop('color', None)

        # extinction threshold
        # get mask
        mask = (original >= min_tresh)
        ma = np.ma.masked_array(original, mask)
        ma_inverted = np.ma.masked_array(original, ~mask)
        
        # fill in the masked region
        # ax.contourf(x_interp_vals, y_interp_vals, ma, levels=[0, 1], colors='#CA9C9E', alpha=.5)

        # contours_extinction = ax.pcolormesh(grid_x, grid_y, ma,
        #             # color='#CA9C9E',
        #             shading='nearest',
        #             **kwargs
        #             )

        ax.set_facecolor('#CA9C9E')

        contours = ax.pcolormesh(grid_x, grid_y, ma_inverted,
                    cmap=cmap,
                    norm=norm,
                    facecolor='#CA9C9E',
                    shading='nearest',
                    **kwargs
                    )

        # ticks
        ax.set_xticks(np.arange(len(x_interp_vals)))
        ax.set_yticks(np.arange(len(y_interp_vals)))
        ax.set_xticklabels(x_interp_vals)
        ax.set_yticklabels(y_interp_vals)

        # limits
        ax.set_xlim(x_interp_vals.min(), x_interp_vals.max())
        ax.set_ylim(y_interp_vals.min(), y_interp_vals.max())

        # scale
        ax.set_xscale('log')
        ax.set_yscale('log')

    # lines
    if contour_line is not None:
        ax.contour(x_interp_vals, y_interp_vals, expanded,
                    levels=contour_line,
                    linewidths=[4, 5],
                    colors=['k', 'white'],
                    extend='both'
                    )

    ax.set_aspect('equal')

    return ax


def generate_heatmap_panel(df,
                        outer_x='comp',
                        outer_y='tau',
                        inner_x='disp',
                        inner_y='hr',
                        values='N*',
                        cmap=default_cmap,
                        norm=default_norm,
                        # vmin=None,
                        # vmax=None,
                        # contour=1.,
                        # cont_kwargs=None,
                        sharex=False,
                        sharey=False,
                        post_process=None,
                        height=2.,
                        facet_kwargs={},
                        **kwargs):
    
    fg = sns.FacetGrid(df,
                        height=height,
                        aspect=1.,
                        col=outer_x,
                        row=outer_y,
                        sharex=sharex,
                        sharey=sharey,
                        margin_titles=True,
                        **facet_kwargs)

    
    fg.map_dataframe(generate_heatmap,
        index=inner_y,
        columns=inner_x,
        values=values,
        cmap=cmap,
        norm=norm,
        # square=True,
        # vmin=vmin,
        # vmax=vmax,
        # cont_kwargs=cont_kwargs,
        **kwargs
        )

    # reduce space between plots setting the distance between them to 0
    # fg.fig.subplots_adjust(wspace=0.02, hspace=0.02)

    # max_j = len(pd.unique(df[outer_x]))

    # add colorbar above the right plots
    # add inset axes for the colorbar on the right
    # cbar_ax = fg.fig.add_axes([1.02, 0.5, 0.03, 0.2])
    # fg.figure.colorbar(fg.axes[0][max_j-1].collections[0], cax=cbar_ax, shrink=1.0, pad=0.01)
    # cbar_ax.set_title(values)

    if post_process is not None:
        fg = post_process(fg)

    return fg

def invert_yaxis(fg):
    for ax in fg.axes.flat:
        ax.invert_yaxis()
    return fg

def fig3(data, save=None):
    '''
    Plot heatmap of carrying capacity as a function of dispersal range and home-range size

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the data to be plotted
    save : str, optional
        Path to save the figure, by default None
    '''
    df = data.copy()

    fg2 = generate_heatmap_panel(
                        df,
                        outer_x='comp',
                        outer_y='disp',
                        inner_x='hr',
                        inner_y='tau',
                        cmap=default_cmap,
                        interp_multiplier=1,
                        gaussian_noise=0.7,
                        levels=None,
                        norm=default_norm,
                        height=40*mm_to_inches,
                        contour_line=[5e-3],
                        facet_kwargs={'row_order': df['dispersal'].sort_values().unique()[::-1]}
                        )

    def even_formatter(x, pos):
        if pos % 2 == 0:
            return magnitude_format(x, pos)
        return ''

    def odd_formatter(x, pos):
        if pos % 2 != 0:
            return magnitude_format(x, pos)
        return ''

    def _null_formatter(x, pos):
        return ''

    # box
    fg2.despine(left=False, bottom=False, right=False, top=False)
    # labels
    # use scientific notation 10^x
    # Remove any existing title texts
    for text in fg2._margin_titles_texts:
        text.remove()
    fg2._margin_titles_texts = []

    # Draw the row titles on the right edge of the grid
    for i, row_name in enumerate(fg2.row_names):
        ax = fg2.axes[i, -1]
        title = r'$\sigma_d = 10^{{{}}}$'.format(int(np.log10(row_name)))
        text = ax.annotate(
            title, xy=(1.02, .5), xycoords="axes fraction",
            rotation=270, ha="left", va="center",
        )
        fg2._margin_titles_texts.append(text)

    # Draw the column titles  as normal titles
    for j, col_name in enumerate(fg2.col_names):
        title = r'$\sigma_q = 10^{{{}}}$'.format(int(np.log10(col_name)))
        fg2.axes[0, j].set_title(title)

    _major_ticks = np.logspace(-6, 1, 8)
    _minor_ticks = []
    for t in _major_ticks:
        for i in range(2, 10):
            _minor_ticks.append(t*i)

    if True:
        for i, ax_row in enumerate(fg2.axes):
            for j, ax in enumerate(ax_row):
                # ticks
                ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(_major_ticks))
                ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(_minor_ticks))
                ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(_major_ticks))
                ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator(_minor_ticks))

                # axes on the bottom
                if i == len(fg2.axes) - 1:
                    if j % 2 == 0:
                        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(even_formatter))
                    else:
                        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(odd_formatter))
                    
                    if j == 1:
                        ax.set_xlabel(r'HR size, $A_{HR}$')
                
                else:
                    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_null_formatter))
                    
                
                # axes on the left
                if j == 0:
                    if i % 2 == 0:
                        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(even_formatter))
                    else:
                        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(odd_formatter))
                    
                    if i == 1:
                        ax.set_ylabel(r'HR crossing time, $\tau$')

                else:
                    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_null_formatter))

        delta_x = 0.068
        # add colorbar below the plots
        cax = fg2.fig.add_axes([0.25 + delta_x, -0.0-0.045, 0.4, 0.03])
        cmap = default_cmap
        norm = default_norm
        cb = fg2.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=cax,
                    orientation='horizontal',
                    cmap=cmap,
                    extend='both',
                    norm=norm)

        # Set the colorbar label and title
        cb.set_label('')
        cb.set_ticks([1e-1, 1e0, 1e1])
        cb.set_ticklabels(['$10^{-1}$', '$10^0$ (CSR)', '$10^1$'])
        # remove minor ticks
        cb.ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        # add text with the label to the right of the colorbar
        fg2.fig.text(0.68 + delta_x, 0.01-0.045, r'Carr. capacity, $n^*$', fontsize=10, ha='left', va='center')

        # text with extinction
        square = mpl.patches.FancyBboxPatch((0.15+delta_x, 0.004-0.045), 0.01, 0.01, fc="#CA9C9E", ec='k', lw=2, boxstyle=mpl.patches.BoxStyle("Round", pad=0.02))
        fg2.fig.add_artist(square)
        fg2.fig.text(0.155+delta_x, -0.02-0.045, 'Extinction', fontsize=10, ha='center', va='top')

        # add abcd labels
        for i, ax_row in enumerate(fg2.axes):
            for j, ax in enumerate(ax_row):
                ax.text(0.9, 0.9, f'({chr(97+j*3+i)})', transform=ax.transAxes, fontsize=10, fontfamily='sans-serif', fontweight='bold', ha='center', va='center')

    # add the lines for SS, OU, BM
    minhr = 1.8822741005438125e-05
    maxhr = 18.82274100543813
    mintau = 1.9999999999999998e-05
    maxtau = 20.0

    from palettable.cmocean.sequential import Turbid_3

    turbid_cmap = Turbid_3.mpl_colors
    turbid_cmap = mpl.colors.ListedColormap(turbid_cmap)
    colors = list(turbid_cmap([.2, .5, 1.]))
    dashes = ['', (3,1,1,1), (1,1)]

    fg2.axes[2, 1].plot([minhr, maxhr], [mintau, maxtau], color='k', linestyle='-', lw=2.5, alpha=.2)
    fg2.axes[2, 1].plot([minhr, maxhr], [mintau, maxtau], color=colors[0], linestyle='-', lw=2, alpha=.7)
    fg2.axes[1, 1].plot([minhr, maxhr], [mintau, maxtau], color=colors[1], linestyle='-.', lw=2, alpha=.7)
    fg2.axes[0, 1].plot([minhr, maxhr], [mintau, maxtau], color=colors[2], linestyle=':', lw=2, alpha=.7)


    # reduce space between plots setting the distance between them to 0
    # wspace ~~ hspace + 0.13
    fg2.fig.subplots_adjust(wspace=0.03, hspace=-0.10)

    # save figure png, svg, pdf
    if save:
        my_savefig(fg2, save)

    return fg2

# FIGURE 4 - PREDICTED VS SIMULATED

def fig4(data, save=None, filters=True):
    '''
    Plot predicted vs simulated carrying capacity

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the data to be plotted
    save : str, optional
        Path to save the figure, by default None
    '''

    df = data.copy()

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(80*mm_to_inches, 120*mm_to_inches), sharey=True)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_yscale('log')


    xvar = 'pred_n_pos'
    xvar2 = 'pred_n_hr'
    yvar = 'N*'

    minx = 2.5e-3
    maxx = 3.

    # ---- filter and aggregate data

    if filters:
        dff = df\
            .query('comp > 1e-3')\
            .query('final_N_org > 2')\
            .select_dtypes('number')\
            .groupby(['hr', 'comp', 'disp', 'tau'])\
            .mean()\
            .reset_index()
            # .query('comp == 1e-2')\
            # .query('final_N_org > 50')\
            # .query('hr < 3.16e-1')\
            # .groupby(['hr', 'comp', 'disp', 'tau'])\
            # .filter(lambda x: len(x) == 5)\
            # .groupby(['hr', 'comp', 'disp', 'tau'])\
            # .mean()\
    
    else:
        dff = df\
            .query('final_N_org > 2')\
            .select_dtypes('number')\
            .groupby(['hr', 'comp', 'disp', 'tau'])\
            .mean()\
            .reset_index()

        minx=1e-4
        maxx=1e2

    dff_og = dff.copy()

    # POSITION

    # filter out the nan values
    dff = dff[[xvar, yvar]].copy()
    dff = dff.replace([np.inf, -np.inf, 0], np.nan).dropna()
    xvals = dff[xvar]
    yvals = dff[yvar]
    x = np.log(xvals)
    y = np.log(yvals)

    # Calculate R2
    xbar = np.mean(x)
    SSres = np.sum((x - y) ** 2)
    SStot = np.sum((x - xbar) ** 2)
    r2_1 = 1 - SSres/SStot

    # plot
    ax1.plot([minx, maxx], [minx, maxx], 'k-', lw=4, alpha=0.3, zorder=1)
    ax1 = sns.scatterplot(data=dff, x=xvar, y=yvar, ax=ax1, marker='x', alpha=.5, linewidth=.75, legend=False, color=qcmap[0])
    
    # HOME RANGE 

    # filter out the nan values
    dff = dff_og[[xvar2, yvar]].copy()
    dff = dff.replace([np.inf, -np.inf, 0], np.nan).dropna()
    xvals2 = dff[xvar2]
    yvals = dff[yvar]
    x2 = np.log(xvals2)
    y = np.log(yvals)

    # Calculate R2
    xbar = np.mean(x2)
    SSres = np.sum((x2 - y) ** 2)
    SStot = np.sum((x2 - xbar) ** 2)
    r2_2 = 1 - SSres/SStot

    # plot
    ax2.plot([minx, maxx], [minx, maxx], 'k-', lw=4, alpha=0.3, zorder=1)
    ax2 = sns.scatterplot(data=dff, x=xvar2, y=yvar, ax=ax2, marker='x', alpha=.5,  linewidth=.75, legend=False, color=qcmap[1])

    # annotation with R2
    ax1.text(0.9, 0.1,
            f"$R^2 = {r2_1:.3f}$",
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax1.transAxes,
            fontsize=11)

    ax2.text(0.9, 0.1,
            f"$R^2 = {r2_2:.3f}$",
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax2.transAxes,
            fontsize=11)

    # labels
    ax1.set_ylabel("")

    # set top ticks, remove bottom ticks
    ax1.set_xlabel(r"Predicted, $1/c_\text{pos}$")

    # set xlabel in the upper part of the plot
    ax1.set_aspect('equal')
    ax1.set_xlim(minx, maxx)
    ax1.set_ylim(minx, maxx)

    ax2.set_xlabel(r"Predicted, $1/c_\text{HR}$")
    ax2.set_ylabel("")
    ax2.set_aspect('equal')
    ax2.set_xlim(minx, maxx)
    ax2.set_ylim(minx, maxx)

    fig.supylabel(r"Actual, $n_{t\to\infty}/n_{\text{CSR}}^*$", x=0.09, va='center', rotation='vertical')

    # remove grid
    ax1.grid(False)
    ax2.grid(False)

    # add abcd labels
    ax1.text(0.1, 0.9, f'({chr(97)})', transform=ax1.transAxes, fontsize=10, fontfamily='sans-serif', fontweight='bold', ha='center', va='center')
    ax2.text(0.1, 0.9, f'({chr(98)})', transform=ax2.transAxes, fontsize=10, fontfamily='sans-serif', fontweight='bold', ha='center', va='center')


    # save figure png, svg, pdf
    if save:
        my_savefig(fig, save)

    plt.show()

# ------ FIGURE 5 - auxiliary functions
def ou_step(x, rand, sigmadt, thetadt, mu):
    # sigmadt = \sqrt(g)*\sqrt{dt}
    # thetadt = dt/\tau
    # rand ~ N(0,1)
    return x + thetadt*(mu - x) + sigmadt*rand

def trajectory(g, tau, T, dt):
    import math
    N = int(T/dt)
    # Create the time vector
    t = np.linspace(0., T, N+1)
    # Create the trajectory
    sigmadt = math.sqrt(g*dt)
    thetadt = dt/tau
    x = np.zeros(N+1)
    y = np.zeros(N+1)
    x[0] = 0.
    y[0] = 0.

    randoms1 = np.random.normal(0., 1., N)
    randoms2 = np.random.normal(0., 1., N)

    for idx in range(N):
        x[idx+1] = ou_step(x[idx], randoms1[idx], sigmadt, thetadt, 0.)
        y[idx+1] = ou_step(y[idx], randoms2[idx], sigmadt, thetadt, 0.)

    return t, x, y

def color_arry_to_hex(color_arr):
    return ["#{:02x}{:02x}{:02x}".format(int(cor[0]*255), int(cor[1]*255), int(cor[2]*255)) for cor in color_arr]

def fig5(save=None, seed=None, doublesigma2=2e-2, max_t=1e-1, taus=[1e-3, 1e-2, 1e-1], usetex=True):
    from scipy.stats import rayleigh
    import math

    if seed is None:
        seed = 3141592653

    # set the seed
    np.random.seed(seed)

    # parameters
    # hr_size = sqrt(tau * g / 2)
    radius_hr_95 = rayleigh.ppf(0.95, math.sqrt(doublesigma2/2))*math.sqrt(doublesigma2/2)
    gs = [doublesigma2 / tau for tau in taus]

    # Styling
    sns.set_style("ticks")
    # set font scale
    sns.set_context("notebook", font_scale=1.6)
    mpl.rcParams['text.usetex'] = usetex

    # Create the figure
    fig, axs = plt.subplots(1, 4, width_ratios=[1,1,1,0.05], figsize=(16/1.2, 5/1.2))

    axs[1].sharey(axs[0])
    axs[2].sharey(axs[0])
    axs[1].sharex(axs[0])
    axs[2].sharex(axs[0])
    axs[3].axis('off')

    for i, (tau, g) in enumerate(zip(taus, gs)):
        # Create the data
        t, x, y = trajectory(g, tau, max_t, 1e-5)

        # parametric plot with color corresponding to the time
        # uses the plasma colormap
        colors = plt.cm.viridis(np.linspace(0, 1., len(t)))
        colors = color_arry_to_hex(colors)
        
        ax = axs[i]
        ax.scatter(x, y, c=colors, s=.5, alpha=0.5)

        ax.set_title(r"$\tau = {:.0e}$".format(tau))

        # Set the aspect
        ax.set_aspect('equal')
        ax.set_facecolor('#f6f6f6')
        circ = plt.Circle((0, 0), radius_hr_95, color='white', fill=True, lw=2, zorder=0)
        ax.add_artist(circ)

    # set the overall x and y labels
    fig.text(0.5, 0.04, r"$x$", ha='center', va='center')
    axs[0].set_ylabel(r"$y$")

    # add abc labels
    for i, ax in enumerate(axs[:-1]):
        ax.text(0.05, .95, chr(97+i)+')', transform=ax.transAxes, fontsize=20, va='top')

    # add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=max_t))
    sm._A = []
    cbar = fig.colorbar(sm, ax=axs[-1], orientation='vertical', fraction=0.9)
    cbar.set_label('time, $t$')

    plt.tight_layout()

    # save figure png, svg, pdf
    if save:
        my_savefig(fig, save)

    plt.show()


def main():
    raise NotImplementedError("This module is not meant to be executed directly.\
        # Please call `python generate_figures.py` from the root directory.")
    # from generate_figs import main as generate_figs
    # generate_figs()


if __name__ == "__main__":
    main()