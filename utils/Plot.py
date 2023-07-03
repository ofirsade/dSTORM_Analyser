import pandas as pd
import numpy as np
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import ast
import scipy
import time
import os
import matplotlib.pyplot as plt
import sys
import csv
import collections
import os
from datetime import datetime
from PyQt5 import QtWidgets
from pyqtgraph import PlotWidget, ScatterPlotItem, PlotCurveItem
import pyqtgraph as pg

from PyQt5.QtWidgets import *



sys.setrecursionlimit(10**6)


##def show_in_window(fig, name):
##    
##    plotly.offline.plot(fig, filename = name, auto_open = True)
##    
##    app = QApplication(sys.argv)
##    web = QWebEngineView()
##    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), name))
##    web.load(QUrl.fromLocalFile(file_path))
##    web.show()
####    sys.exit(app.exec_())


def plot_res(xyzl, cpd, fname, alg, output_path, dt_string):
    """
    @params:
    @@xyzl -
    @@cpd - [loc_num, volume, radius, density_2d, density_3d]
    @@fname -
    @@alg -
    @@output_path -
    """

    fig = make_subplots(
        rows = 1, 
        cols = 1, 
##        vertical_spacing=0.05,
##        subplot_titles = fname + alg
    )

    bg_df = xyzl.loc[xyzl['Label'] == -1]
    clustered_df = xyzl.loc[xyzl['Label'] != -1]
    unique_labels = set(clustered_df['Label'].values.tolist())
    # Probe 0
    fig.add_trace(
        go.Scattergl(
            x = bg_df['x'].values,
            y = bg_df['y'].values,
            mode = 'markers',
            marker=dict(
                color='grey',           # set color to an array/list of desired values
                opacity=0.1
            )
        ),
        row = 1, col = 1
    ) 
        
    # Draw clusters
    for i in unique_labels:
        pc = clustered_df.loc[clustered_df['Label'] == i]
        tmp_cpd = cpd[str(i)]
        
        fig.add_trace(
            go.Scattergl(
                x = pc['x'].values,
                y = pc['y'].values,
                mode = 'markers',
                marker = dict(
                    color = i, #cmap(np.sqrt(colocalization[i])),    
##                    colorscale = 'rainbow',
                    colorscale = px.colors.qualitative.Pastel2,
                    opacity = 0.5
                ),
                hovertemplate = ("<b>{text}</b><br><br>".format(text = 'Cluster #' + str(i))
                                 + "Loc No.: {a}<br>".format(a = tmp_cpd[0])
                                 + "Volume: {b}<br>".format(b = tmp_cpd[1])
                                 + "Radius: {c}<br>".format(c = tmp_cpd[2])
                                 + "Density (2D): {d}<br>".format(d = tmp_cpd[3])
                                 + "Density (3D): {e}".format(e = tmp_cpd[4])
                                 + "<extra></extra>")
            ),
            row = 1, col = 1
        )
    
    fig.update_xaxes(range = [0, 19000], row = 1, col = 1)     
    fig.update_yaxes(scaleanchor = 'x', scaleratio = 1, row = 1, col = 1)
    fig.update_layout({'plot_bgcolor': 'whitesmoke'})
    fig.update_layout(showlegend = False)
    fig.update_layout(title = alg + ' ' + fname)
    fig.update_layout(hoverlabel = dict(font_size = 18,
                                        font_color = 'white',
                                        font_family="Rockwell"))


    html_name = dt_string + ' ' + fname + ' ' + alg + '.html'
    html_path = os.path.join(output_path, html_name)
    fig.write_html(html_path)
    fig.show()
    
