# -*- coding: utf-8 -*-
"""
RAPTOR
Implementation of the Microsoft pathfinding algorithm for a Naptan/Transxchange network

"""
#import dask
import dask.bag as db
#from dask.distributed import Client
from itertools import product
import logging
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path

from ..timetables import get_timetables
from ..locations.naptan import assemble_tfl_naptan
from ..tools import Timer
from .. import cfg

#%%
# Input data definitions
timetable_inputs = {
                    'NR_Date':  '20210416',    # or None to download latest from NR Data Feeds
                    'TfL_Date': '20210222'  # or None to download latest from TfL API
                    }

# Output files will be labelled DUN_M_{datatype}_pathversion-pathrun
#  Use a new pathversion for a major change, or a pathrun for a minor one.
output_labels = {'version': 'v2', # Pathfind version
                  'run':    'rail_test',  # Pathfind run
                  'datatype': '{}'}   # Leave blank as this is autofilled later

#%%
# Set the logging to be to the console
logger = logging.getLogger(__name__)

# Fetch relevant variables form the configuration file
walk_speed  = cfg.cfg['raptor'].getfloat('walk_speed') #80 # metres per minute
wait_cutoff = cfg.cfg['raptor'].getfloat('wait_cutoff') #3600  # Pax should wait no longer than this (in seconds) for a train

gjt_weighting = {k: float(v) for k, v in cfg.cfg['raptor'].items()}
g_colormap    = {k: v for k, v in cfg.cfg['mode_colours'].items()}

path_files = cfg.cfg['raptor']['path_outputs_file'].format(**output_labels)
log_file   = cfg.cfg['raptor']['log_file'].format(**output_labels)


# Function definitions

"""
def get_pairwise(df, key, col):
    df = pd.DataFrame([
        [n, x, y]
        for n, g in df.groupby(key)[col]
        for x, y in combinations(g, 2)
        ], columns=[key, 'i', 'j'])\
        .set_index(key)

    return df
"""
def get_interchange_area(N, stop):
    """Shows the interchange links from a network for a stop"""
    c     = [ichg for ichg in nx.connected_components(N) if stop in ichg]
    nodes = [n for i in c for n in i]
    I     = nx.subgraph(N,nodes)
    nx.draw_networkx(I)
    return I

def pythag_distance(xdiffs, ydiffs):
    """Gets the distance between two sets of coordinate differences"""
    return np.sqrt(xdiffs**2 + ydiffs**2)                

def filter_nodes_by_timetable(tt, naptan, filt='mASC'):
    """Takes the timetable and Naptan reference data and provides nodes & interchange movements that intersect them"""

    unique_scheduled_edges   = tt['schedules'][['LocationID','LocationNext']].drop_duplicates().dropna().values
    unique_interchange_edges = naptan['Interchanges'][['i','j']].values
    unique_edges = np.concatenate([unique_scheduled_edges, unique_interchange_edges])
    G            = nx.from_edgelist(unique_edges)
    GC_inscope   = [G.subgraph(g) for g in nx.connected_components(G) if any(G.subgraph(g).has_edge(*e) for e in unique_scheduled_edges)]
    G_inscope    = nx.compose_all(GC_inscope)
    
    unique_nodes = list(G_inscope.nodes())
    unique_nodes = naptan['Nodes'].reindex(unique_nodes)\
                              .rename(columns={'Easting':'x','Northing':'y'})
    
    # Get all origin points, one per Primary Stop Area
    relevant_areas = naptan['StopsInArea'].loc[naptan['StopsInArea']['AtcoCode'].isin(unique_nodes.index), 'StopAreaCode']
    if filt=='mASC':
        relevant_ascs  = naptan['mASC'].loc[naptan['mASC']['StopAreaCode'].isin(relevant_areas), 'TfL_mASC'].unique()
        origin_nodes   = naptan['mASC'].set_index('TfL_mASC')\
                                     .query("PrimaryStopArea")\
                                     .loc[relevant_ascs, ['StopAreaCode']]
    else:
        origin_nodes   = relevant_areas.unique()
    f_interchanges = naptan['Interchanges'].i.isin(unique_nodes.index) & naptan['Interchanges'].j.isin(unique_nodes.index)
    interchange_times = naptan['Interchanges'][f_interchanges].set_index('i')[['j','JT']]
    
    return unique_nodes, origin_nodes, interchange_times

def filter_timetable(tt):
    """Turns a PTSP-format Timetable object (with trips and schedules tables) into a single Raptor-able dataframe
    Arguments:
        tt : Timetable object (from ironroad.timetables.get_timetables) with trips and schedules properties
    Returns:
        tt_raptor: dict of pd.DataFrames with keys:
                    Events: pd.DataFrame linking nodes to routes (i, RouteID: EventSeq)
                    Departures: pd.DataFrame of departure times (i, RouteID, RunID, EventSeq: SchDep)
                    Arrivals: pd.DataFrame of arrival times (RouteID, RunID: EventSeq, i, SchArr)
    
    """
    renamer = dict(
        #RouteID = 'Line',
        i        = 'LocationID',
        EventSeq = 'EventSeq',
        SchDep   = 'TimeDepS',
        SchArr   = 'TimeArrS',
        RunID    = 'ScheduleID',
        )
        
    # Pre-filtered delayed timetable objects to aid performance of the algorithm
    tt_filt = tt['schedules'].join(tt['trips']['RouteID'])\
                             .reset_index()\
                             .rename(columns={v:k for k,v in renamer.items()})\
                             .sort_values(['RouteID','RunID','EventSeq'])
    
    tt_raptor = {'Events':     tt_filt.drop_duplicates(subset=['i','RouteID'])[['i','RouteID','EventSeq']].dropna().set_index(['i','RouteID']).sort_index(),
                 'Departures': tt_filt.set_index(['i','RouteID','RunID','EventSeq'])[['SchDep']].dropna().sort_index(),
                 'Arrivals':   tt_filt.set_index(['RouteID','RunID'])[['EventSeq', 'i', 'SchArr']].dropna().sort_index()
                }
    return tt_raptor

def traverse_routes(improvements, schedules, multilabel):
    """ Finds available departures from each marked node and traverses them """

    # Find routes available at the marked stops, and timetable subset of these routes
    Q_routeset = schedules['Events'].reindex(improvements.index, level='i') # 0.973ms with 2C
    Q_nodes    = Q_routeset.index.get_level_values(0).unique()
    TT_subset  = schedules['Departures'].loc[Q_nodes]
    
    if len(Q_routeset) == 0:
        # If no available vehicle departures from marked stops, no new arrival times by vehicle
        trip_arrs = pd.Series(index=multilabel.index, dtype='bool')
        T_runset  = None

    else:
        Q_board = TT_subset.join(improvements['t_arr'], on='i')\
                           .rename(columns={'t_arr':'t_platform'})
        Q_board['wait'] = Q_board['SchDep'] - Q_board['t_platform']
        Q_board         = Q_board.loc[Q_board['wait'].between(0, wait_cutoff)]
        f_earliestdep   = Q_board['wait']==Q_board.groupby(['RouteID','i'])['wait'].transform('min') # 1ms
        Q_board         = Q_board.loc[f_earliestdep,['t_platform','SchDep','wait']]
        
        T_runset = Q_board.join(schedules['Arrivals'], on=['RouteID','RunID'])\
                          .rename(columns={'i':'target'})
        f_runset = ((T_runset['EventSeq'] > T_runset.index.get_level_values('EventSeq'))
                    & (T_runset['SchArr'] > T_runset['SchDep'])
                    & (T_runset['target'] != T_runset.index.get_level_values('i')))
        T_runset = T_runset[f_runset]
 
        # Get the earliest arrival times at each location visited

        T_runset['JT']  = T_runset['SchArr'] -  T_runset['t_platform']
        T_runset['IVT'] = T_runset['JT'] -  T_runset['wait']
        T_runset['GJT'] = T_runset['IVT']* gjt_weighting['t_invehicle'] + T_runset['wait'] * gjt_weighting['t_wait']

        f_best_arrival = T_runset.groupby('target')[['GJT','SchArr']]\
                                 .transform('min')\
                                 .eq(T_runset[['GJT','SchArr']])\
                                 .all(axis=1)
        T_runset = T_runset.loc[f_best_arrival]\
                            .droplevel('EventSeq')\
                            .drop('EventSeq', axis=1)\
                            .reset_index()\
                            .set_index('target')
        T_runset = T_runset[~T_runset.index.duplicated()]
        
        # Determine whether these arrivals represent a faster route
        prev_arrs = multilabel.reindex(T_runset.index)['t_arr']
        trip_arrs = T_runset['SchArr'].lt(prev_arrs)\
                     .reindex(multilabel.index, fill_value=False)
        
    if trip_arrs.any():
        # If any of the arrivals are earlier, update the multilabels for those locations
        multilabel.loc[trip_arrs, 'improved'] = True
        multilabel.loc[trip_arrs, 't_start']  = T_runset['t_platform']
        multilabel.loc[trip_arrs, 't_dep']    = T_runset['SchDep']
        multilabel.loc[trip_arrs, 't_arr']    = T_runset['SchArr']
        multilabel.loc[trip_arrs, 'p_board']  = T_runset['i'].astype('str')
        multilabel.loc[trip_arrs, 'p_method'] = T_runset['RunID'].astype('str')

    return multilabel

def traverse_footpaths(improvements, interchange, multilabel):
    """ Traverses interchange links within a RAPTOR network """

    # Use the available interchange links at each location to add onward arrival times at connected nodes
    footpaths = improvements[['t_arr']]\
                    .join(interchange, on='target', how='inner')
    footpaths['SchArr'] = footpaths['t_arr']+footpaths['JT']
    footpaths = footpaths.sort_values('SchArr')\
                    .drop_duplicates('j')\
                    .reset_index()\
                    .rename(columns={'target':'i'})\
                    .set_index('j')
    footpaths.index.name = 'target'
    
    # Determine whether these arrivals represent a faster route
    prev_footpath_arrs = multilabel.reindex(footpaths.index)['t_arr']
    footpath_arrs      = footpaths['SchArr'].lt(prev_footpath_arrs)\
                         .reindex(multilabel.index, fill_value=False)

    if footpath_arrs.any():
        # If any of the arrivals are earlier, update the multilabels for those locations
        multilabel.loc[footpath_arrs, 'improved'] = True
        multilabel.loc[footpath_arrs, 't_start']  = footpaths['t_arr']
        multilabel.loc[footpath_arrs, 't_dep']    = footpaths['t_arr']
        multilabel.loc[footpath_arrs, 't_arr']    = footpaths['SchArr']
        multilabel.loc[footpath_arrs, 'p_board']  = footpaths['i']
        multilabel.loc[footpath_arrs, 'p_method'] = 'i'

    return multilabel

def initialise_labels(S):
    """ Initialises a set of multilabels whereby each stop p gets the earliest arrival time t from source with k stages {p: {k: t}} """

    multilabels_index = pd.Index(S, name='target')
    multilabels_cols  = pd.Index(['t_start','t_dep','t_arr','improved','p_method','p_board'], name='value')
    multilabels       = pd.DataFrame(index=multilabels_index,columns=multilabels_cols)
    multilabels['t_start']    = np.nan
    multilabels['t_dep']    = np.nan
    multilabels['t_arr']    = np.inf
    multilabels['improved'] = False
    multilabels['p_board']  = ''
    multilabels['p_method'] = ''
    return multilabels

def raptor_round(prev_label, schedule, interchange):
    """ Executes a round k>1 of RAPTOR """
    
    improvements = prev_label.query('improved')
    
    # STAGE 1
    # Set all arrival times to be equivalent to the previous round 
    #   (upper bound on earliest arrival time with k stages)
    first_multilabel = prev_label.copy()
    first_multilabel['improved'] = False
    first_multilabel['t_start']  = np.nan
    first_multilabel['t_dep']    = np.nan
    first_multilabel['p_board']  = ''
    first_multilabel['p_method'] = ''
    
    # STAGE 2            
    # Traverse route from improvements (marked stops) by depart_events
    # If improvements made by traversing routes, then look at labels improved as a result
    # Otherwise extend the walk routes found previously
    vehicle_result = traverse_routes(improvements, schedule, first_multilabel)

    if vehicle_result.improved.any():
        #print("Vehicle improvements made")
        second_multilabel = first_multilabel.copy().mask(vehicle_result.improved, vehicle_result)
        improvements      = second_multilabel.query("improved")
    else:
        #print("No vehicle improvements made")
        second_multilabel = first_multilabel.copy()
        
    # STAGE 3: For each footpath (pi,pj), set τk(pi) = min{τk(pj) or τj(pi) + walk time)
    footpath_result  = traverse_footpaths(improvements, interchange, second_multilabel)

    if footpath_result.improved.any():
        #print("Interchange improvements made")
        final_multilabel = second_multilabel.mask(footpath_result.improved, footpath_result)
    else:
        #print("No interchange improvements made")
        final_multilabel = second_multilabel.copy()

    return final_multilabel

def get_raptor_paths(p_source, t_dep, S_nodes, schedule, interchange, K_max=50):
    """ Carries out RAPTOR pathfinding from node p_source at time t_dep to all S_nodes using provided schedule and interchange sets"""

    K = np.arange(0, K_max)

    # ROUND 0: Initialise all labels
    multilabels_pt = {}
    k_round = 0
    multilabels_pt[k_round] = initialise_labels(S_nodes)
    multilabels_pt[k_round].loc[p_source] = [t_dep, t_dep, t_dep, True, 'a', p_source]
    
    print(f"Established entry at {p_source} at {t_dep}")
    
    for k_round in K[1:]:
        #k_round+=1
        #print(f"Round {k_round}")
        prev_label = multilabels_pt[k_round-1].copy()
        multilabels_pt[k_round] = raptor_round(prev_label, schedule, interchange)
        multilabels_pt[k_round].query("improved")
        
        if not multilabels_pt[k_round]['improved'].any():
            return multilabels_pt
        
    return multilabels_pt

def run_raptor(tt, all_nodes, origin_nodes, origin_times, interchange_times, max_rounds=50, savefile=True, savepath=path_files):
    """Runs the Raptor algorithm against timetable tt with nodes all_nodes and interchange links interchange_times for origin_nodes at origin_times
    Arguments:
        tt                      : instance of Timetable
            A Raptorable timetable (Timetable class)
        all_nodes               : list or list-like object
            List of all the nodes in the network
        origin_nodes            : list or list-like object
            Origin nodes from which to calculate paths
        origin_times            : list or list-like object
            Origin times from which to calculate paths
        interchange_times       : pd.DataFrame
            Interchange links to use between timetable nodes
        max_rounds (optional)   : int (default: 50)
            Maximum number of Raptor rounds to carry out for each origin place and time
        savefile (optional)     : bool (default: True)
            Whether to save the multilabels (path results)
        savepath (optional)     : Path object with formatting string {} to insert 'Multilabels'
            Where to save the multilabels
    Returns:
        result              : pd.DataFrame
            Multilabels for each origin node/time to each destination node for each journey opportunity
        time_mean           : Time taken to run Raptor for each origin node/time
    
    
    """
    all_tx = Timer("Running RAPTOR...")

    seq = list(product(origin_nodes, origin_times))
    tt_raptor = filter_timetable(tt)

    pt = db.from_sequence(seq)
    multilabels = pt.starmap(get_raptor_paths, 
                             S_nodes=all_nodes, 
                             schedule=tt_raptor, 
                             interchange=interchange_times,
                             K_max=max_rounds)
    
    #multilabels = partial(get_raptor_paths, S_nodes=S, schedule=tt_raptor, 
    #                     interchange=interchange_times)
    #result = starmap(multilabels, seq) 
    result = multilabels.compute(scheduler='processes')
    result = {seq+(r,): rround for seq, data in zip(seq, result) for r, rround in data.items()}
    
    t_total = all_tx.stop()
    
    result = pd.concat(result, names=['S_origin', 'T_origin', 'k_round'])\
               .query("improved")\
               .drop('improved', axis=1)\
               .reset_index()\
               .assign(JT = lambda x: x.t_arr - x.t_dep)\
    
    time_mean = t_total / len(seq)
    
    if savefile:
        # Write results to file
        out_multilabels_path = Path(str(savepath).format('Multilabels'))
        
        result.to_csv(out_multilabels_path,
                  mode='a', 
                  index=False,
                  header=not out_multilabels_path.exists())          
    return result, time_mean

convert_pos = lambda df: {key: tuple(coord.values()) for key, coord in df.to_dict(orient='index').items()}

#%%
if __name__=='__main__':
    
    # Get timetables
    modes = ['d','u','t'] #,'r','b'] #,'c','f']
    
    t = Timer("Retrieving timetables & interchanges...")
    
    tt_nr  = get_timetables.get_nr(timetable_inputs['NR_Date'], fallback=False)
    tt_tfl = get_timetables.get_tfl(timetable_inputs['TfL_Date'], modes=modes)
    tt     = get_timetables.Timetable(tt_tfl, tt_nr, day=3) # Merges timetables together for a specified day of the week (if only one is needed, only provide one)
    
#%%
    # Load all relevant Naptan data
    naptan = assemble_tfl_naptan()
    
    t.stop()
    # Filter the unique network nodes against the provided timetable (slow - 204s for udtr)
    t = Timer("Filtering interchanges against timetable...")
    
    unique_nodes, origin_nodes, interchange_times = filter_nodes_by_timetable(tt, naptan) #.compute()
    
    t.stop()
    
    #%%
    #client = Client(dashboard_address='localhost:8787')
    # open http://localhost:8787/status
    origin_times      = [28800, 29700]
    origin_nodes_test = origin_nodes[:20].index.tolist()
    n_departures      = len(origin_nodes_test)*len(origin_times)
    n_arrivals        = len(unique_nodes)
    
    t = Timer(f"Raptorising {n_departures} departures...")
    try:
        result, mean_runtime = run_raptor(tt, 
                                all_nodes = unique_nodes.index.tolist(), 
                                origin_nodes = origin_nodes_test, 
                                origin_times = origin_times, 
                                interchange_times = interchange_times)
        time_estimate       = pd.Timedelta(mean_runtime * len(origin_nodes), 's')
        time_estimate_total = pd.Timedelta(mean_runtime * len(origin_nodes) * (96-21), 's')
        
        print(f"Mean time per n_departure to {n_arrivals} destinations: {mean_runtime:.3f}s")
        print(f"Expected time for {len(origin_nodes)} origins & 1 dep time: {time_estimate}")
        print(f"Expected time for {len(origin_nodes)} origins & {(96-21)} dep times: {time_estimate_total}")

    except Exception as e:
        print("Could not complete runs.")
        raise Exception(f"{e}\n\n").with_traceback(e.__traceback__)
    finally:
        #client.close()
        t_elapsed = t.stop()
        print(f"Check: {t_elapsed/n_departures:.3f}s per n_departure")
    
    
    #%%
    t = Timer("Saving results..")
    
    # Lookup for use at the end
    tt_lookup = tt['schedules'][['LocationID','TimeDepS','TimeArrS']]\
                    .join(tt['trips'][['Mode','Line','RouteID']])\
                    .reset_index('EventSeq',drop=True)\
                    .rename(columns={'TimeDepS':'SchDep','LocationID':'i','TimeArrS':'SchArr'})
    tt_lookup.index = tt_lookup.index.astype(str)
    tt_lookup       = tt_lookup.reset_index().astype({'RouteID':'str'})
    
    tt_lookup.to_parquet(str(path_files).format('Timetable'), compression='gzip')    
    result.to_parquet(str(path_files).format('Multilabels'), compression='gzip')
    unique_nodes.to_parquet(str(path_files).format('Nodes'), compression='gzip')
   
    t.stop()
    
