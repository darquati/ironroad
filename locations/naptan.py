#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:57:45 2021

@author: davearquati
"""

from itertools import combinations
import logging
from numpy import sqrt
import pandas as pd
from pathlib import Path
import networkx as nx

from .. import cfg

logger = logging.getLogger(__name__)

walk_speed  = cfg.cfg['raptor'].getfloat('walk_speed') #80 # metres per minute
gjt_weighting = {k: float(v) for k, v in cfg.cfg['raptor'].items()}

def all_paired_nodes(N):
    ichg_sets = list( nx.connected_components(N) )
    group_pairs = [combinations(s, 2) for s in ichg_sets]
    all_pairs = [item for sublist in group_pairs for item in sublist]
    
    return all_pairs

def load_naptan():
    naptan_ref = {n.stem: pd.read_csv(n, encoding='iso-8859-1') for n in Path(cfg['paths']['naptan_folder']).glob('*.csv')}
    return naptan_ref

def clean_naptan(naptan_ref):
    
    structure = naptan_ref['naptan_data_structure']
    structure_stops = structure[~structure.IsGroup].drop('IsGroup', axis=1).set_index(['NodeTypeCode','BusStopType'])
    structure_area = structure[structure.IsGroup].drop(['IsGroup','BusStopType'], axis=1).set_index('NodeTypeCode')
    
    f_stopareas_status = naptan_ref['StopAreas'].Modification.ne('del') & naptan_ref['StopAreas'].Status.ne('del')
    f_stopareas_ldn_metro = naptan_ref['StopAreas'].StopAreaCode.str.contains('940GZZCR|940GZZDL|940GZZLU')
    f_stopareas_nr = naptan_ref['StopAreas'].StopAreaCode.str.contains('910G')
    f_stopareas_ldn_admin = naptan_ref['StopAreas'].AdministrativeAreaCode==82
    f_stopareas = f_stopareas_ldn_metro | f_stopareas_nr | f_stopareas_ldn_admin

    stopareas = naptan_ref['StopAreas'].loc[f_stopareas_status & f_stopareas, ['StopAreaCode','AdministrativeAreaCode','StopAreaType','Name','Easting','Northing']]\
                        .set_index('StopAreaCode')\
                        .join(structure_area[['IsEntryPoint','Descriptor']], on='StopAreaType')

    f_stopsinarea = (naptan_ref['StopsInArea'].StopAreaCode.isin(stopareas.index) & 
                     naptan_ref['StopsInArea'].Modification.ne('del'))
    
    stopsinarea = naptan_ref['StopsInArea'].loc[f_stopsinarea, ['StopAreaCode', 'AtcoCode']]

    f_stops = ((naptan_ref['Stops'].AdministrativeAreaCode.eq(82) | 
                (naptan_ref['Stops'].AdministrativeAreaCode.isin([110,147]) & 
                 naptan_ref['Stops'].ATCOCode.isin(stopsinarea.AtcoCode))) & 
               naptan_ref['Stops'].Modification.ne('del') & 
               naptan_ref['Stops'].Status.ne('del'))


    stops = naptan_ref['Stops'].loc[f_stops, ['ATCOCode','CommonName','Easting','Northing','StopType','BusStopType','AdministrativeAreaCode']]\
                       .rename(columns={'ATCOCode':'AtcoCode'})\
                       .set_index('AtcoCode')   \
                       .join(structure_stops, on=['StopType','BusStopType'])
    stops = stops.join(stopsinarea.set_index('AtcoCode'))
    stops['AreaOrphan'] = stops['StopAreaCode'].isnull()
    
    areahierarchy = naptan_ref['AreaHierarchy'][['ParentStopAreaCode', 'ChildStopAreaCode']]
    areahierarchy = areahierarchy[areahierarchy.isin(stopareas.index).any(axis=1)]
    
    missing_low = naptan_ref['mASC'].set_index('TfL_mASC')[['StopAreaCode']]
    missing_hub = naptan_ref['mASC'].dropna(subset=['Hub']).set_index('Hub')[['TfL_mASC']]
    missing_cluster = naptan_ref['mASC'].dropna(subset=['Cluster']).set_index('Cluster')[['Hub']]
    
    missing = pd.concat([missing_low, missing_hub, missing_cluster]).stack().droplevel(-1).reset_index()
    missing.columns=['ParentStopAreaCode','ChildStopAreaCode']
    missing = missing[missing.ParentStopAreaCode != missing.ChildStopAreaCode]
    
    areahierarchy = areahierarchy.append(missing[['ParentStopAreaCode','ChildStopAreaCode']])
    
    return stops, stopareas, areahierarchy

def naptan_interchanges(stops, stopareas, areahierarchy):
    # Turn this into a network, find all connected nodes in each area
    A = nx.from_pandas_edgelist(areahierarchy, 'ParentStopAreaCode','ChildStopAreaCode')
    S = nx.from_pandas_edgelist(stops[~stops.AreaOrphan].reset_index(), 'StopAreaCode','AtcoCode')
    N = nx.compose(A, S)
    N.add_nodes_from(stops.to_dict(orient='index'))
    
    # Add coordinates
    pairs = pd.DataFrame(all_paired_nodes(N), columns=['i','j'])
    coords = stopareas[['Easting','Northing']].append(stops[['Easting','Northing']])
    pairs['dist'] = pairs.join(coords, on='i')\
                         .join(coords, on='j', rsuffix='_j')\
                         .eval("(Easting - Easting_j)**2 + (Northing-Northing_j)**2")\
                         .apply(sqrt)
    
    pairs['dist'] = pairs['dist'].fillna(pairs.groupby('i')['dist'].transform('max'))
    pairs['JT'] = (pairs['dist'] / walk_speed)*60
    pairs['GJT'] = pairs['JT'] * gjt_weighting['t_walk']
    
    pairs_ji = pairs.copy()
    pairs_ji[['i','j']] = pairs[['j','i']]
    
    pairs = pairs.append(pairs_ji)
        
    return pairs, N

def assemble_tfl_naptan():
    stns = pd.read_excel(Path(cfg.cfg['paths']['ref_stations_file']), 'List').dropna(subset=['Master ASC'])
    naptan_asc = pd.read_excel(Path(cfg.cfg['paths']['ref_naptan_folder'])/'Naptan-Numbat.xlsx', 
                                       'Naptan-ASC', 
                                       usecols=['StopAreaCode','AtcoCode',
                                                'TfL_mASC','PrimaryStopArea'])
    naptan = load_naptan()
    naptan['mASC'] = naptan_asc.join(stns.set_index('Master ASC')[['Hub','Cluster']], on='TfL_mASC')
    
    # Clean the data & transform into a set of interchange links (and a network graph)
    stops, stopareas, areahierarchy = clean_naptan(naptan)
    naptan['Interchanges'], naptan['N'] = naptan_interchanges(stops, stopareas, areahierarchy)
    #get_interchange_area(N, 'WIMu')
          
    # Get a list of all the unique network nodes
    naptan['Nodes'] = pd.concat([naptan['Stops']\
                            .set_index('ATCOCode')[['CommonName','Easting','Northing','StopType','BusStopType']]\
                            .rename(columns={'CommonName':'Name','StopType':'Type'}),
                        naptan['StopAreas']\
                        .set_index('StopAreaCode')[['Name','StopAreaType','Easting','Northing']]\
                        .rename(columns={'StopAreaType':'Type'})])\
                    .append(stns.set_index('Master ASC')[['Unique Station Name','Easting','Northing']]\
                           .rename(columns={'Unique Station Name':'Name'}))
    naptan['Nodes'] = naptan['Nodes'][~naptan['Nodes'].index.duplicated()]
    
    # Manual fix
    naptan['Nodes'].loc['9400ZZLUWIG3'] = naptan['Nodes'].loc['9400ZZLUWIG1']
    naptan['Nodes'].loc['940GZZLUBZP', ['Easting','Northing']]=[527375,185104]
    
    return naptan


if __name__ == '__main__':
    naptan_ref = assemble_tfl_naptan()
