#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IronRoad
Path Reconstruction
Reconstructs RAPTOR results into a pathlegs table

@author: davearquati
"""

#import dask.dataframe as dd
#from dask.distributed import Client
import pandas as pd
#import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx

import site
ironroad_folder =  Path(r'/Users/davearquati/OneDrive - Transport for London/Data/IronRoad')
site.addsitedir(ironroad_folder)
from ironroad.main import Timer

working_folder = Path(r'/Users/davearquati/OneDrive - Transport for London/Data/IronRoad')

output_labels = {'version': 'v2', # Pathfind version
                  'run':    'railtest',  # Pathfind run
                  'datatype': '{}'}   # Leave blank as this is autofilled later

# File locations
path_files = working_folder / Path(r'_outputs/IronRoad-{version}-{run}_{datatype}.parquet.gzip'.format(**output_labels))
log_file = working_folder / Path(r'_outputs/IronRoad-{version}-{run}_log.txt'.format(**output_labels))

lines_file = Path('/Users/davearquati/OneDrive - Transport for London/Data/IronRoad/_reference/nr_network/NRTT_Definitions_v9.xlsx')

gjt_weighting = {'t_invehicle':    1,    # Multiplier for in-vehicle JT
                 't_walk':         2,    # Multiplier for walk JT
                 'n_boardings':    3.5*60,    # Fixed time penalty for accessing a platform (=board)
                 't_wait':         2,    # Multiplier for wait time
                 't_osi':          0}    # Multiplier for OSIs (in addition to board penalty)


#%%
t = Timer("Loading results...")
#client = Client(processes=False)
# open at http://localhost:8787/status

result    = pd.read_parquet(str(path_files).format('Multilabels')) #, index_col=['S_origin', 'T_origin', 'target', 'k_round'])
origin_nodes = result['S_origin'].unique()#.compute().tolist() #n=702
origin_times = result['T_origin'].unique()#.compute().tolist() #n=2

result[['t_start','t_dep','t_arr','JT']] = result[['t_start','t_dep','t_arr','JT']].astype(int)

lines = pd.read_excel(lines_file,'Lines',index_col=0)

t.stop()

#%%
t = Timer("Extracting paths (may take around 10 mins)...")

#O, O_dep = 'TBYu', 28800;
#group = result[(result.S_origin==O) & (result.T_origin==O_dep)].compute()

def reconstruct_path(group):
    print(group.name)
    O, O_dep = group.name
    #group = result[(result.S_origin==O) & (result.T_origin==O_dep)]
    G = nx.DiGraph()
    sources = list(group[['p_board','t_start']].itertuples(index=False, name=None))
    targets = list(group[['target','t_arr']].itertuples(index=False, name=None))
    G.add_edges_from(zip(sources, targets))
    attr_col_headings = group[['k_round','t_dep','p_method','JT']]
    attribute_data = zip(*[group[col] for col in attr_col_headings])
    for s, t, attrs in zip(sources, targets, attribute_data):
        G[s][t].update(zip(attr_col_headings, attrs))
    T = nx.dfs_tree(G, source=(O, O_dep))
    ot_paths = nx.shortest_path(T, (O, O_dep))
    ot_paths = [(O,int(O_dep),D,int(D_arr),k,i, int(i_arr)) for (D, D_arr), nodes in ot_paths.items() for k, (i, i_arr) in enumerate(nodes)]
    pathframe = pd.DataFrame(ot_paths, columns=['Origin','OriginDep','Destin','DestinArr','PathSeq','i','i_arr'])
    return pathframe

"""
meta={'Origin':str,'OriginDep':float,
    'Destin':str,'DestinArr':float,
    'PathSeq':int,
    'i':str,'i_arr':float})
"""
paths = result.groupby(['S_origin','T_origin']).apply(reconstruct_path)
                                                    
#paths = paths.compute()

paths['j'] = paths['i'].shift(-1).mask(paths.shift(-1)['PathSeq']==0)

full_paths = paths.merge(result, 
                         left_on=['Origin','OriginDep','i','j','i_arr'], 
                         right_on=['S_origin','T_origin','p_board','target','t_start'])\
                  .drop(['S_origin','T_origin','p_board','target','t_start'], axis=1)\
                  .rename(columns={'p_method':'ScheduleID'})
full_paths = full_paths[full_paths.Destin.isin(origin_nodes)]
full_paths['ScheduleID'] = full_paths['ScheduleID'].astype('str')

t.stop()

#%%
t = Timer("Reconstructing itinerary & saving result...")

tt_lookup = pd.read_parquet(str(path_files).format('Timetable'), columns=['ScheduleID','i','Mode','Line']).set_index(['ScheduleID','i'])

result_paths = full_paths.join(tt_lookup, on=['ScheduleID','i']).set_index(['Origin','OriginDep','Destin','DestinArr','PathSeq']).sort_index()

#result_paths[['i_arr','t_dep','t_arr','JT']] = result_paths[['i_arr','t_dep','t_arr','JT']].astype(int)
result_paths['Mode'] = result_paths['Mode'].fillna('i')
result_paths['Line'] = result_paths['Line'].replace(lines['LineCode'])
#result_paths.loc[result_paths.Mode=='d', 'Line'] = 'DLR'

result_paths.loc[result_paths.k_round==1, 'Line'] = 'N'
result_paths.loc[result_paths.k_round.shift(-1)==1, 'Line'] = 'X'
result_paths.Line = result_paths.Line.fillna(result_paths.Line.shift(1)+'>a|b>'+result_paths.Line.shift(-1))
result_paths['Line'] = result_paths['Line'].str.replace('N>a|','')

result_paths['t_walk']      = (result_paths['t_arr'] - result_paths['i_arr']).where(result_paths['Mode']=='i')
result_paths['t_invehicle'] = (result_paths['t_arr'] - result_paths['t_dep']).where(result_paths['Mode']!='i')
result_paths['t_wait']      = result_paths['t_dep'] - result_paths['i_arr']
result_paths['n_boardings']     = result_paths['t_wait'].gt(0)
result_paths['GJT'] = (result_paths[['t_walk','t_invehicle','t_wait','n_boardings']] * pd.Series(gjt_weighting)).sum(axis=1)
result_paths['JT'] = result_paths[['t_walk','t_invehicle','t_wait']].sum(axis=1)

result_paths.to_parquet(str(path_files).format('Paths'), compression='gzip')

t.stop()

