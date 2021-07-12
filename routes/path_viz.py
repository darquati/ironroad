#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IronRoad
Path Viz


@author: davearquati
"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.pyplot as plt

import site
#module_folder = Path(r'//onelondon.tfl.local/shared/4600b/Data/01_Development/01_PythonModules') # Allow import of custom modules/scripts
module_folder =  Path(r'/Users/davearquati/OneDrive - Transport for London/Data/IronRoad')
site.addsitedir(module_folder)
from ironroad.locations import naptan
#import seaborn as sns

working_folder = Path(r'/Users/davearquati/OneDrive - Transport for London/Data/IronRoad')

output_labels = {'version': 'v2', # Pathfind version
                  'run':    'railtest',  # Pathfind run
                  'datatype': '{}'}   # Leave blank as this is autofilled later

# File locations
path_files = working_folder / Path(r'_outputs/IronRoad-{version}-{run}_{datatype}.parquet.gzip'.format(**output_labels))
log_file   = working_folder / Path(r'_outputs/IronRoad-{version}-{run}_log.txt'.format(**output_labels))

#%%

naptan = naptan.assemble_tfl_naptan()

result_paths = pd.read_parquet(str(path_files).format('Paths'))

summary = result_paths.groupby(['Origin','OriginDep','Destin','DestinArr'])[['t_walk','t_invehicle','t_wait','n_boardings','GJT','JT']].sum()
summary['Difficulty'] = summary['GJT'] / summary['JT']

best = summary.groupby(['Origin','Destin']).min()

contour = (best.groupby('Origin').mean()+best.groupby('Destin').mean())/2
contour = contour.join(naptan['Nodes'][['Easting','Northing']])
contour = contour.dropna()

#%%
from scipy.interpolate import griddata

x = contour['Easting'].values
y = contour['Northing'].values
z = contour['Difficulty'].values

def plot_contour(x,y,z,resolution=50,contour_method='linear'):
    resolution = str(resolution)+'j'
    X,Y = np.mgrid[min(x):max(x):complex(resolution), min(y):max(y):complex(resolution)]
    points = [[a,b] for a,b in zip(x,y)]
    Z = griddata(points, z, (X, Y), method=contour_method)
    return X,Y,Z

X,Y,Z = plot_contour(x,y,z, resolution=5000, contour_method='linear')

with plt.style.context("seaborn-pastel"):
    fig, ax = plt.subplots(figsize=(13,8), dpi=200)
    ax.scatter(x,y, color="black", linewidth=1, edgecolor="ivory", s=15)
    ax.contour(X,Y,Z)

#%%
"""#%%
#Takes ages

group_nodes = origin_nodes.join(naptan['StopAreas'].set_index('StopAreaCode')[['Name','Easting','Northing']],on='StopAreaCode')
pair_info = result_paths.index.droplevel([-1,-2,-4]).unique()\
                .to_frame().reset_index(drop=True)\
                .join(group_nodes[['Name','Easting','Northing']], on='Origin')\
                .rename(columns={'Name':'OriginName','Easting':'x1','Northing':'y1'})\
                .join(group_nodes[['Name','Easting','Northing']], on='Destination')\
                .rename(columns={'Name':'DestinName','Easting':'x2','Northing':'y2'})\
                .set_index(['Origin','Destination'])
pair_info['dist'] = pythag_distance(pair_info.eval('x2-x1'), pair_info.eval('y2-y1'))

t = Timer("Converting paths into journey times & finding best")
# Convert paths into a set of journey times and determine best JT
result_jts = result_paths.groupby(level=['Origin','Destination','OriginDep','DestinArr'])[['jt','gjt']].sum()
result_jts['ichgs'] = result_paths['Mode'].eq('i').groupby(level=['Origin','Destination','OriginDep']).sum()
result_jts['difficulty'] = result_jts['gjt'] / result_jts['jt']
#result_jts = result_jts.join(pair_info)
t.stop()

#%%
result_best = result_jts.loc[result_jts.groupby(['Origin','Destination','OriginDep'])['jt'].idxmin()].join(pair_info)
fig, ax = plt.subplots(figsize=(25,20))
p_result = result_best.loc['AAPr']
p_result.iloc[0:1].plot.scatter(ax=ax, x='x1', y='y1', 
                                 s=25, alpha=0.25, c='black')
p_result.plot.scatter(ax=ax, x='x2', y='y2', s=50,
                         c=p_result['difficulty'], 
                         cmap='RdPu', alpha=0.5, 
                         xlim=(505000,555000), ylim=(158000,198000))

result_best.plot.scatter(x='dist', y='jt')

#%%
# JT, dist relationship
#result_best.plot.hexbin(x='aerial_dist', y='JT', figsize=(20,15))

import seaborn as sns
sns.kdeplot(result_best['dist'],  
            result_best['jt'], color='b', 
            shade=True,
            cmap="Blues", shade_lowest=False) 

#%%
# Contour plot
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

result_surface = result_best.dropna(subset=['x2','y2']).query("jt<10000")[['x2','y2','jt']]
fig = plt.figure(figsize=(20,15), dpi=300)
ax = Axes3D(fig)
ax.plot_trisurf(result_surface.x2, 
                result_surface.y2, 
                result_surface.jt, 
                cmap=cm.jet, linewidth=0.2,
                antialiased=True)
ax.set_xlim3d(505000,555000)
ax.set_ylim3d(158000,198000)

#%%

t = Timer("Writing results to file...")

result_paths.to_csv(str(path_files).format('paths'))
result_jts.to_csv(str(path_files).format('jts'))
result_best.to_csv(str(path_files).format('best'))
target_nodes.loc[result_paths['j'].unique()].to_csv(str(path_files).format('nodes'))
"""