#!/usr/bin/env python3

# Copyright 2022 Autarky.ai LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A collection of handy plotting utilities.

Functions
---------
corr_heatmap()
    Plot a correlation matrix as a seaborn heatmap.
scatter()
    Draw a scatter plot with a colorbar for dates.

"""

__all__ = ['corr_heatmap', 'scatter']

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


try:
    plt.style.use('autarkydotai')
except OSError:
    plt.style.use(['seaborn-poster', 'seaborn-whitegrid'])
    plt.rcParams['figure.figsize'] = (18, 8)


def corr_heatmap(data, cmap=None, method='pearson', min_periods=1, tril=True):
    """Plot the correlation matrix of `data` as a seaborn heatmap.

    Parameters
    ----------
    data : `~pandas.DataFrame`
        The rectangular data set on which to compute the pairwise
        column correlations. The index/column information will be
        used to label the axes of the heatmap chart.
    cmap : str or `~matplotlib.colors.Colormap`, or list of colors,
           optional
        The mapping from data values to color space.
    method : {'pearson', 'kendall', 'spearman'} or callable
        Method of correlation:
        * pearson - standard Pearson correlation coefficient
        * kendall - Kendall's tau correlation coefficient
        * spearman - Spearman rank correlation
        * callable - callable with input two 1D ndarrays and returning
                     a float. Note that the returned matrix from corr
                     will have 1s along the diagonal, and will be
                     symmetric regardless of the callable's behaviour.
    min_periods : int, optional
        Minimum number of observations required per pair of columns to
        have a valid result. Currently only available for Pearson and
        Spearman correlation.
    tril : bool, default=True
        Whether to mask the upper triangle of the correlation matrix.
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    if data.shape[1] < 2:
        raise ValueError("'data' must contain at least 2 columns")

    corr = data.corr(method=method, min_periods=min_periods)
    corr = corr.multiply(100)
    mask = np.triu(np.ones_like(corr, dtype=bool)) if tril else None
    sns.heatmap(corr, cmap=cmap, annot=True, fmt='.0f',
                cbar_kws={'label': '%'}, mask=mask)
    plt.show()


def scatter(x, y, cmap='jet', xlabel=None, ylabel=None):
    """Draw a scatter plot of `y` vs `x`.

    Adds a colorbar to indicate the date each point corresponds to.

    Parameters
    ----------
    x, y : `~pandas.Series` or `~pandas.DataFrame`
        The data to plot. Index values must only consist of dates.
    cmap : str or `~matplotlib.colors.Colormap`, default='jet'
        A `~matplotlib.colors.Colormap` instance or registered
        colormap name, passed as the `cmap` argument to
        :func:`~matplotlib.pyplot.scatter`.
    xlabel : str, optional
        Label to use for the x-axis.
    ylabel : str, optional
        Label to use for the y-axis.
    """
    if (not isinstance(x, (pd.Series, pd.DataFrame))
            or not isinstance(y, (pd.Series, pd.DataFrame))):
        raise ValueError("Both 'x' and 'y' must be a pandas Series "
                         "or pandas DataFrame")
    if len(x) != len(y):
        raise ValueError("'x' and 'y' must be the same size")
    if not x.index.is_all_dates:
        x.index = pd.to_datetime(x.index)
    if not y.index.is_all_dates:
        y.index = pd.to_datetime(y.index)
    x = x.sort_index()
    y = y.sort_index()

    colors = np.linspace(0, 1, len(x))
    cm = plt.get_cmap(cmap)
    sc = plt.scatter(x, y, c=colors, cmap=cm, lw=0)
    step = 1 if len(x) < 20 else len(x) // 10
    ticks = colors[::step]
    ticklabels = [str(val.date()) for val in x[::step].index]
    cb = plt.colorbar(sc, ticks=ticks)
    cb.ax.set_yticklabels(ticklabels)
    plt.axline((0, 0), slope=1, c='k', label=r'$y=x$')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')
    plt.show()