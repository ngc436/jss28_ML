__author__ = 'Maria Khodorchenko'

from sklearn.manifold import t_sne

import numpy as np

import plotly.graph_objs as go
from plotly.offline import plot
from plotly.plotly import image
from plotly import tools

COLUMN_NAMES = ['T_xacc', 'T_yacc', 'T_zacc', 'T_xgyro', 'T_ygyro', 'T_zgyro', 'T_xmag', 'T_ymag', 'T_zmag',
                'RA_xacc', 'RA_yacc', 'RA_zacc', 'RA_xgyro', 'RA_ygyro', 'RA_zgyro', 'RA_xmag', 'RA_ymag', 'RA_zmag',
                'LA_xacc', 'LA_yacc', 'LA_zacc', 'LA_xgyro', 'LA_ygyro', 'LA_zgyro', 'LA_xmag', 'LA_ymag', 'LA_zmag',
                'RL_xacc', 'RL_yacc', 'RL_zacc', 'RL_xgyro', 'RL_ygyro', 'RL_zgyro', 'RL_xmag', 'RL_ymag', 'RL_zmag',
                'LL_xacc', 'LL_yacc', 'LL_zacc', 'LL_xgyro', 'LL_ygyro', 'LL_zgyro', 'LL_xmag', 'LL_ymag', 'LL_zmag']

path_to_figure = 'Figures\\'


def plot_timeseries(data, people=8):
    traces = []
    names = ['sitting', 'standing', 'laying (back)', 'laying (right)',
             'asc stairs', 'desc stairs', 'standing in elev',
             'moving in elev', 'walking in a parking lot', 'threadmill 4km/h (flat)',
             'thredmill 4km/h (15 deg)', 'running 8 km/h', 'on a stepper',
             'cross trainer', 'cycling on a bike (horiz)', 'cycling on a bike (vert)',
             'rowing', 'jumping', 'playing basketball']
    fig = tools.make_subplots(rows=people, col=1)
    #y = [i for i in range(0,len(data.shape[0]))]
    x = [i for i in range(len(data[1][1]))]
    for i in range(people):
        fig.append_trace(go.Scatter(
            x=x,
            y=data[i][0],
            name=str(i)
        ), i+1, 1)

    image.save_as(fig, filename=''.join([path_to_figure, 'uncertanty_new', '.jpeg']))
