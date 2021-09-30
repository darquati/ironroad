#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IronRoad Timetable Objects

Provides three classes of Timetable in TfL Service Planning format:
- TimetableNR loads from an extracted NR CIF file (as SQLite) into an object and can save the object (as SQLite)
- TimetableTfL loads from an extracted TfL Journey Planner timetable (saving is available but untested - files are huge)
- Timetable takes one or both objects along with a specified weekday and joins them into a consistent object for that weekday

@author: DavidArquati

"""

import logging
import sqlite3
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt

from ..locations.naptan import load_naptan
from ..tools import (Timer, compass_direction, compass_to_cardinal,
                   pythag_distance)
from ..timetables import nr
from .. import cfg

#%%
logger = logging.getLogger(__name__)

def cif_to_time(series, out='timedelta'):
    """ Convert CIF times to Pandas timedeltas """
    tx = series.fillna('').str.replace('H','30').str.replace(' ','00')
    hr = tx.str[0:2]
    mn = tx.str[2:4]
    sc = tx.str[4:6]
    hms =  hr+':'+mn+':'+sc
    hms = hms.replace({'::':None, '  :  :00':None, '::00':None}) 
    hms_td = pd.to_timedelta(hms)
    if out=='timedelta':
        return hms_td
    elif out=='seconds':
        return hms_td.astype('timedelta64[s]')

def fill_by_dynamic_lookup(ttd, index_fields=['LocationID','LocationNext'], fill_field='DirectionLink'):
    """ New lookup """
    linkdirection_counts = ttd.groupby(index_fields+[fill_field]).size().unstack()
    links_that_have_one_direction = linkdirection_counts.notnull().sum(axis=1)==1
    unambiguous_links = linkdirection_counts[links_that_have_one_direction].stack().reset_index(fill_field)[fill_field]
    unambiguous_links.name = fill_field+'U'
    unambiguous_directions = ttd.join(unambiguous_links, how='left', on=index_fields)[unambiguous_links.name]
    directions = ttd[fill_field].fillna(unambiguous_directions)
    return directions

def fill_by_run(ttd, run_field='ScheduleID', fill_field='DirectionLink'):
    """ Fill missing directions where run has an otherwise constant direction """
    rundirection_counts = ttd.groupby([run_field,fill_field]).size().unstack()
    links_that_have_direction = rundirection_counts.notnull().sum(axis=1)==1
    unambiguous_runs = rundirection_counts[links_that_have_direction].stack()\
                                                                     .reset_index(fill_field)[fill_field]
    unambiguous_directions = unambiguous_runs.reindex(ttd.index, level=run_field)
    directions = ttd[fill_field].fillna(unambiguous_directions)    
    return directions

class Timetable:

    def __init__(self, tt_tfl=None, tt_nr=None, day=3):
        self.state = None
    
        t_trips = []
        t_schedules = []
    
        if tt_nr is not None:
            print("Extracting day's trips from NR schedule")
            nr_trips_inscope = tt_nr.trips.reindex(tt_nr.trips_byweekday.xs(day, 0, 'Day')['ScheduleID'].unique())\
                                    .drop(-1,errors='ignore')\
                                    .join(tt_nr.summary)
            nr_schedules_inscope = tt_nr.schedules[tt_nr.schedules.Flags.str.match("S[Cud]L")].loc[nr_trips_inscope.index]
    
            # Align formats
            print("Renaming NR fields for compatibility")
            nr_schedules_inscope.index = nr_schedules_inscope.index.droplevel('CallSeq')
            nr_schedules_inscope = nr_schedules_inscope.drop(['Track','Platform','Flags',
                                                              'LocationPrev','LocationNext',
                                                              'CallPrev','DirectionCompass',
                                                              'DirectionCardinal','Speed',
                                                              'TimePass','DirectionRun','UTime',
                                                              'Distance','easting','northing'], axis=1)
            nr_schedules_inscope = nr_schedules_inscope.rename(columns={'CallNext':'LocationNext',
                                                                        'TimeArr':'TimeArrS',
                                                                        'TimeDep':'TimeDepS',
                                                                        'RunTime':'RunTimeS',
                                                                        'Dwell':'DwellTimeS'})
            # Map NR location codes into Naptan ones so they can align with TfL/Transxchange format
            naptan_ref = load_naptan()
            
            print("Remapping NR codes to Naptan codes for compatibility")
            map_tiploc_naptan = naptan_ref['RailReferences'].set_index('TiplocCode')['AtcoCode']
            nr_schedules_inscope['LocationID'] = nr_schedules_inscope['LocationID'].replace(map_tiploc_naptan)
            nr_schedules_inscope['LocationNext'] = nr_schedules_inscope['LocationNext'].replace(map_tiploc_naptan)
    
            nr_trips_inscope['Mode'] = 'r'
            nr_trips_inscope = nr_trips_inscope[['Mode','Line','DirectionRun','ServiceGroupID','Origin','Destination','RouteID']]
    
            t_trips.append(nr_trips_inscope)
            t_schedules.append(nr_schedules_inscope)
    
        if tt_tfl is not None:
            print("Extracting day's trips from TfL schedule")
            tfl_trips_inscope     = tt_tfl.trips.loc[tt_tfl.trips_byweekday[tt_tfl.trips_byweekday["Weekday"]==3].index]
            tfl_schedules_inscope = tt_tfl.schedules.join(tfl_trips_inscope['Mode'], how='right').drop('Mode', axis=1)
            
            tfl_schedules_inscope[['TimeArrS','TimeDepS']] = tfl_schedules_inscope[['TimeArr','TimeDep']].astype('timedelta64[s]')
            tfl_schedules_inscope = tfl_schedules_inscope.drop(['TimingLinkID','RouteLinkID',
                                                                'LinkSeq','TimeBand','TimeArr','TimeDep'], axis=1)
            
            tfl_trips_inscope = tfl_trips_inscope[['Mode','Line','DirectionRun','ServiceGroupID','Origin','Destin','RouteID']]
            tfl_trips_inscope = tfl_trips_inscope.rename(columns={'Destin':'Destination'})
    
            # Get unique calling patterns, assume that runs with the same pattern don't overtake each other
            print("Converting RouteID to hashes based on calling patterns")
            schedule_patterns = tt_tfl.schedules.groupby('ScheduleID').agg(Pattern=('LocationID',tuple))
            unique_patterns = schedule_patterns.join(tt_tfl.trips[['Line','ServiceGroupID','RouteID']]).groupby(['Line','ServiceGroupID','RouteID','Pattern']).size().reset_index()
            unique_patterns['CallingPatternID'] = unique_patterns['Pattern'].apply(hash)
            calling_patterns = unique_patterns.drop_duplicates(subset=['RouteID','CallingPatternID']).set_index('RouteID')['CallingPatternID']
            tfl_trips_inscope['RouteID'] = tfl_trips_inscope['RouteID'].map(calling_patterns)
            
            t_trips.append(tfl_trips_inscope)
            t_schedules.append(tfl_schedules_inscope)
        
        print("Joining timetables together")
        self.trips     = pd.concat(t_trips)
        self.schedules = pd.concat(t_schedules)
        
    def __getitem__(self, key):
        if key == 'trips':
            return self.trips
        elif key == 'schedules':
            return self.schedules

class NRDefinitions:
    def __init__(self, definitions_file=cfg.cfg['paths']['ref_nr_definitions_file'], 
                           stations_file=cfg.cfg['paths']['ref_stations_file'], 
                           scope_field='NBT2021'):
        print(f"Loading NR definitions from {definitions_file}")
        # Get list of location codes
        with pd.ExcelFile(definitions_file) as definitions:
            l_tiplocs = pd.read_excel(definitions, 'Locations', index_col='TIPLOC')\
                          .rename(columns={'nrstationflag':'stationflag'})
            l_directions_l = pd.read_excel(definitions, 'Links', index_col=[0,1], 
                                           usecols=['origin','destination','line',
                                                    'direction_initial','direction_final','distance'])\
                               .rename(columns={'distance':'Distance'})
            line_remapping = pd.read_excel(definitions, 'LineRemapping', 
                                           usecols=['CallsAt','IfLine','MapLineTo'])
            l_tiplocs_nodes = pd.read_excel(definitions, 'Locations', index_col='TIPLOC')
        
        # Get London area of scope and our internal PTSP ASC codes
        print(f"Loading PTSP definitions from {stations_file} based on {scope_field} scope")
        with pd.ExcelFile(stations_file) as stations:
            l_scope = pd.read_excel(stations, 'ASC-NBTScope', index_col='MasterASC')
            l_scope = l_scope[l_scope[scope_field]]
            l_nr_asc = pd.read_excel(stations, 'NR-ASC', index_col='NR_TIPLOC', 
                                     usecols=['NR_TIPLOC','NR_CRS','MasterASC'])
            l_nr_asc.index.name='TIPLOC'
            l_scope = pd.read_excel(stations, 'Stations', index_col='MasterASC').reindex(l_scope.index)
            l_altnames = pd.read_excel(stations, 'AltNames-ASC', index_col='Name')
            l_modes = pd.read_excel(stations, 'ASC-Mode', index_col='MasterASC').drop(['UniqueStationName','Active'], axis=1)
            l_lineseq = pd.read_excel(stations, 'ASC-LineSeq', index_col='MasterStnSeq')
            l_servseq = pd.read_excel(stations, 'ASC-ServiceSeq', index_col='MasterStnSeq')
        
        # Remove duplicates from directions, prioritising non-bus (Line==Bus.all or Line!=Bus)
        all_links_bus   = (l_directions_l['line']=='BUS').groupby(level=['origin','destination']).transform('all')
        link_not_bus    = (l_directions_l['line']!='BUS')
        l_directions_l  = l_directions_l.loc[all_links_bus | link_not_bus, ['direction_initial','direction_final','Distance']].sort_values('Distance')
        l_directions_l  = l_directions_l[~l_directions_l.index.duplicated()].sort_index()

        l_directions    = l_directions_l[['direction_initial','direction_final']]
        l_distances     = l_directions_l['Distance']
        
        network = nx.MultiDiGraph()
        nx.from_pandas_edgelist(l_directions_l.replace({'U':1,'D':-1}).reset_index(), 
                                source='origin', target='destination', 
                                edge_attr=['direction_initial','direction_final','Distance'], 
                                create_using=network)
        
        
        l_tiplocs       = l_tiplocs.join(l_nr_asc['MasterASC'])
        l_tiplocs[['stationflag','londonflag','lonterminalflag']] = l_tiplocs[['stationflag','londonflag','lonterminalflag']].fillna(0).astype(bool)
        
        l_coords            = l_tiplocs[['easting','northing']].dropna()
        l_location_flags    = l_tiplocs[['stationflag','londonflag']]
        l_terminals         = l_tiplocs[l_tiplocs['lonterminalflag']].index.unique()
        
        nr_numbat           = l_scope.join(l_modes[l_modes['r']]['r'], how='inner')
        missing_nr_numbat   = ~nr_numbat.index.isin(l_nr_asc['MasterASC'])
        print(f"Are any NUMBAT NR stations missing from the Power Reference NR-ASC lookup?: {missing_nr_numbat.any()}")
        if missing_nr_numbat.any(): print(f"Missing: {', '.join(nr_numbat[missing_nr_numbat].index.tolist())}")
        
        self.lookups = {   'Directions': l_directions,
                           'Distances': l_distances,
                           'Tiplocs': l_tiplocs,
                           'Tiplocs-Nodes': l_tiplocs_nodes,
                           'Locations': l_location_flags,
                           'Modes': l_modes,
                           'NBTStations': l_scope,
                           'Coordinates': l_coords,
                           'NR-ASC': l_nr_asc,
                           'AltNames': l_altnames,
                           'LineRemapping': line_remapping,
                           'LineSeq': l_lineseq,
                           'ServSeq': l_servseq,
                           'Terminals': l_terminals,
                           'Network': network}
    def __getitem__(self, key):
        return self.lookups[key]

    def get_tiploc_for(self, simplename):
        return self.lookups['Tiplocs'].query(f"NameSimple=='{simplename}'").index[0]


class TimetableNR:

    refpoints = {
        # x0 (westernmost), x1 (easternmost), y0 (southernmost), y1 (northernmost)
        'london': (['PADTON','LIVST','WATRLMN','STFD','ECROYDN','CLPHMJC',
                          'HTRWAPT','ORPNGTN','MLHB','SUTTONC','CHESHNT','LEWISHM',
                          'WATFDJ','KGSTON','DARTFD','ROMFORD','ALEXNDP','SURBITN',
                          'EALINGB','ABWD'],
                   [[505000,560000],[160000,206000]] ),
    
        'lse':    (['PADTON','STFD','ECROYDN','CLPHMJC','HTRWAPT','MLHB','BARKING',
                          'CHESHNT','WATFDJ','DARTFD','ROMFORD','HWYCOMB','SHENFLD',
                          'STEVNGE','SLOUGH','WOKING','PBRO','OXFD','IPSWICH',
                          'BRGHTN','GTWK','GUILDFD','BROMLYS','CAMBDGE','BSNGSTK',
                          'SVNOAKS','STALBCY','SOTON','MKNSCEN','TONBDG','ASHFKY',
                          'GRVSEND','LUTON','BEDFDM'],
                   [[420000,630000],[90000,320000]]),
    
        'national': (['PADTON','MNCRPIC','GLGC','LEEDS','BHAMNWS','LESTER',
                            'EDINBUR','BRGHTN','CRDFCEN','RDNGSTN','NTNG','BRSTLTM',
                            'IPSWICH','CREWE','LVRPLSH','YORK','CAMBDGE','PRST','SOTON',
                            'HDRSFLD','NWCSTLE'],
                     [[140000,650000],[0,1030000]])}

    def __init__(self, databundle, 
                       scope_field='NBT2020',
                       datewindow=None,
                       filters={'Train Status':   ['P','1'],
                                'Train Category': ['OO','XX']},
                       #reprocess=False, # Start at state 0 (load from CIF) or highest available
                       load_until_state=6, # Load existing data where available no further than this state (0:start from CIF, 6:max)
                       process_until_state=6, # Continue processing after load, no further than this state (6:max)
                       prior_probabilities='2020',
                       verbose=False,
                       save=True):
        
        print("Establishing TimetableNR object...")
        self.state = -1
        """
        States
        -1: CIF file
        0: CIF database
        1: Scoped
        2: Cleaned
        3: Enriched
        4: Patterned
        5: Lined
        6: Summarised
        """
        
        self.databundle = databundle
        
        if datewindow is None:
            ttdate = pd.to_datetime(databundle, yearfirst=True)
            self.datewindow = (ttdate, ttdate+pd.Timedelta('7D'))
        else:
            self.datewindow = datewindow
        
        output_folder = Path(cfg.cfg['paths']['nr_output_folder'].format(databundle=databundle))
        if not output_folder.exists(): output_folder.mkdir(parents=True)
        
        self.input_db  = Path(cfg.cfg['paths']['nr_extracted_file'].format(databundle=databundle))
        self.output_db = Path(cfg.cfg['paths']['nr_processed_file'].format(databundle=databundle))

        input_exists = self.input_db.exists()
        output_exists = self.output_db.exists()
        
        output_file_template = output_folder/'{name}_{d}.csv'.format(name='{name}', d=databundle)
        
        self.output_file_names = {
            'output_uids_summary'        :  'NRTrainList_UIDs_Summary_',
            'output_uids_byweekday'      :  'NRTrainList_PermByWeekday',
            'output_uids_bydate'         :  'NRTrainList_AllByDate',
            #'output_uids_planned_bydate' :  'NRTrainList_PlannedByDate',
            
            'output_timetable_permbyday' :  'NRTimetable_PermanentDaily',
            'output_timetable_allbyuid'  :  'NRTimetable_All',
            
            'output_summary_link_frequency' :  'NRDepSummary',
            'output_stn_departures'         :  'NRStnOperatorCounts',
            'output_operated_volumes'       :  'NRDailyStats',
            'output_line_identification_results' : 'NRLineIDResults',
            
            'event_counts_file'          :   'NREventCounts',
            'output_line_probabilities_file':'NRLineProbabilities-Output',
            
            'badcalls_file'              :   'NRResultLog_BadCalls',
            'classification_log_file'    :   'NRResultLog_BadLineClassification',
        }
        
        self.output_file_locations = {key: Path(str(output_file_template).format(name=value)) for key, value in self.output_file_names.items()}
        
        self.logs       = {}
        self.lookups    = NRDefinitions(scope_field=scope_field)
        
        if output_exists and (load_until_state > 0):
            self.reload(load_until_state) # Load from the already-processed database
            self.state = min((self.state, load_until_state))
            print(f"Achieved with state {self.state}")
            
        elif input_exists:
            self.load_cif_db(datewindow=datewindow, filters=filters) # Load from the CIF extract
            self.set_state(0)
        else:
            _  = nr.get_timetable(ttdate=databundle, source='local')
            self.load_cif_db(datewindow=datewindow, filters=filters) # Load from the CIF extract
            
        if (process_until_state > self.state):
            if (self.state < 1) and (process_until_state >= 1):
                print(f"[Process] Processing from CIF...")           
                self.set_scope()            # results in state 1 (finds trips_by_date, trips_by_weekday)
                if save: self.save_db(increment=[1])

            if (self.state < 2) and (process_until_state >= 2):
                self.clean_timetable()      # results in state 2 (recasts self.schedules into PTSP format)
                if save: self.save_db(increment=[2])

            if (self.state < 3) and (process_until_state >= 3):
                print(f"[Process] Enriching with location & geometry data...")
                self.identify_locations()   # results in state 2.1
                self.add_distance()         # results in state 2.2
                self.add_geometry()         # results in state 2.3
                self.add_directions()       # results in state 3
                if save: self.save_db(increment=[3])

            if (self.state < 4) and (process_until_state >= 4):
                print(f"[Process] Gathering calling patterns...")
                self.patterns, self.routes = self.gather_patterns(self.schedules, self.trips_bydate) #
                self.set_state(4)
                if save: self.save_db(increment=[4])

            if (self.state < 5) and (process_until_state >= 5):
                print(f"[Process] Identifying lines...")   
                in_probabilities_file = cfg.cfg['paths']['ref_nr_line_probabilities_file'].format(dprior=prior_probabilities)                
                try:
                    self.lines = self.identify_lines(self.routes, in_probabilities_file)
                    self.lines = self.remap_lines(self.lines)
                    self.set_state(5)
                    self.validate_lines(plot_all=verbose)                
                    if save: 
                        self.save_db(increment=[5])
                        self.save_logs()
                except Exception as err:
                    print(err)

            if (self.state < 6) and (process_until_state >= 6):
                print(f"[Process] Summarizing...")
                try:
                    self.create_summary()
                    #self.create_link_summary()
                    #self.create_volume_summary()
                    #self.create_station_summary()
                    if save: self.save_db(increment=[6])
                except Exception as err:
                    print(err)


    def __repr__(self):
        return f'TimetableNR(databundle={self.databundle},\ndatewindow={self.datewindow},\nstate={self.state})'

    def set_state(self, state):
        # Changes the 'state' of the TimetableNR to reflect what processing has been done
        self.state = state 
        schedules_length = len(self.schedules)
        print(f"Length of schedules was {schedules_length} at state {state}")
         
    def load_cif_db(self, input_db=None, datewindow=None, filters=None):
        # Load the SQL database into a Dataframe
        if input_db is None: input_db = self.input_db
        zd = {}
        print(f"Loading from CIF extract at {self.input_db}")
        with sqlite3.connect(self.input_db) as con:
            #con = create_engine(r"sqlite:///{}".format(input_cif_database))
            
            for table in ['BS', 'BX', 'LO', 'LI', 'LT']:
                print("Reading {}...".format(table))
                zd[table] = pd.read_sql("SELECT * FROM {table}".format(table=table), con)
            
        # 2. Set up the overview of trains
        # Add the basic schedule (overview of trains) into single dataframe
        overview = pd.merge(zd['BS'].set_index('linenum'), 
                            zd['BX'].set_index('linenum'), 
                            left_index=True, 
                            right_on='BS_line')\
                     .set_index('BS_line')
        overview.index.name = 'ScheduleID'
        
        overview['Date Runs From'] = overview['Date Runs From'].astype(str).apply(pd.to_datetime, format='%y%m%d')
        overview['Date Runs To'] = overview['Date Runs To'].astype(str).apply(pd.to_datetime, format='%y%m%d')
    
        if datewindow is not None:
            overview['Date Range Start'] = pd.to_datetime(datewindow[0], yearfirst=True)
            overview['Date Range End'] = pd.to_datetime(datewindow[1], yearfirst=True)
            overview['StartWindow'] = overview[['Date Runs From','Date Range Start']].max(axis=1)
            overview['EndWindow'] = overview[['Date Runs To','Date Range End']].min(axis=1)
            f_cif_datewindow = overview['StartWindow'] <= overview['EndWindow']
            overview = overview[f_cif_datewindow].drop(['Date Range Start','Date Range End',
                                                        'StartWindow','EndWindow'], axis=1)
    
        else:
            overview['StartWindow'] = overview['Date Runs From']
            overview['EndWindow'] = overview['Date Runs To']
            f_cif_datewindow = overview['StartWindow'] <= overview['EndWindow']
            overview = overview[f_cif_datewindow].drop(['StartWindow','EndWindow'], axis=1)
             
        # Filter for specific labels
        if filters is not None:
            f_cif = True
            for key, value in filters.items():
                f_cif_filt = overview[key].isin(value)
                f_cif = f_cif & f_cif_filt
        
            # Apply filters & tidy up
            pre_filter_length = len(overview)
            post_filter_length = len(overview[f_cif])
            
            overview = overview[f_cif]
            print("There are {} out of {} trips within specified parameters.".format(post_filter_length, pre_filter_length))
        
        overview['Bank Holiday Running'] = overview['Bank Holiday Running'].replace({' ':True,'X':False})
        overview['Applicable Timetable Code'] = overview['Applicable Timetable Code'].replace({'Y':True,'N':False})
        
        overview['DaysPerWeek'] = overview['Days Run'].str.count('1')
        
        dropcols = ['Record Identity_y','Spare_x','Spare_y','Transaction Type',
                    'Course Indicator','Reserved field RSID',
                    'Reserved field Data source']
    
        renamecols = {'Record Identity_x':'Record Identity',
                        'Days Run':'DaysOperated',
                        'Speed':'BookedSpeed',
                        'ATOC Code':'Operator',
                        'Train UID':'TrainUID',
                        'Train Service Code':'ServiceGroupID',
                        'Portion Id':'PortionID'}
        
        overview = overview.drop(dropcols, axis=1)\
                           .rename(columns=renamecols)
        
        # Load the detailed calling pattern of each train into a df
        tt = pd.concat([zd['LO'].set_index('linenum'),
                        zd['LI'].set_index('linenum'),
                        zd['LT'].set_index('linenum')])\
               .rename(columns={'BS_line':'ScheduleID'})
        
        tt.index.name='BSSeq'
        
        tt = tt[tt.ScheduleID.isin(overview.index.unique())]
        
        self.schedules = tt 
        self.trips = overview
        self.set_state(0)
        
    def reload(self, max_state):
        #engine = create_engine(r"sqlite:///{}".format(input_database))
        #conn = engine.connect()
        print(f"Loading from already-processed database at {self.output_db}, no further than state {max_state}")
        
        with sqlite3.connect(self.output_db) as conn:
            kwargs = {'con': conn}
            
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", **kwargs)['name'].tolist()
            self.state = pd.read_sql('SELECT * FROM state', **kwargs).squeeze()
            
            if 'summary' in tables:
                state = min((6, max_state))
            elif 'lines' in tables:
                state = min((5, max_state))
            elif 'routes' in tables:
                state = min((4, max_state))
            elif 'trips_byweekday' in tables:
                state = min((2, max_state))
                
            if state != self.state:
                logger.info(f"State clash: self-declared state is {self.state} whilst observed state is {state}")
                self.state = state
            else:
                logger.info(f"State: {self.state}")
                
            if self.state >= 1:
                logger.info("Fetching trips, schedules, trips_bydate and trips_byweekday")
                self.trips           = pd.read_sql('SELECT * FROM trips', index_col='ScheduleID', **kwargs)
                self.trips_bydate    = pd.read_sql('SELECT * FROM trips_bydate', parse_dates='Date', **kwargs).set_index(['TrainUID','Day','Date'])
                self.trips_byweekday = pd.read_sql('SELECT * FROM trips_byweekday', **kwargs).set_index(['TrainUID','Day'])
                self.schedules       = pd.read_sql('SELECT * FROM schedules', **kwargs).set_index(['ScheduleID','CallSeq','EventSeq'])
                self.trips_bydate.columns = self.trips_bydate.columns.astype(int)
                
            if self.state >= 4:
                logger.info("Fetching patterns and routes")
                self.patterns = pd.read_sql('SELECT * FROM patterns', index_col='ScheduleID', **kwargs) # ScheduleID>Pattern,Weight,RouteID
                self.routes   = pd.read_sql('SELECT * FROM routes', **kwargs)\
                                  .groupby('RouteID')\
                                  .agg({'Pattern':tuple,
                                        'Weight':'first'}) # RouteID>Pattern,Weight
            if self.state >= 5:
                logger.info("Fetching lines")
                self.lines = pd.read_sql('SELECT * FROM lines', index_col='RouteID', **kwargs) # RouteID>Line,LinePrediction...

            if self.state >= 6:
                logger.info("Fetching summaries")
                self.summary = pd.read_sql('SELECT * FROM summary', index_col='ScheduleID', **kwargs)
                """
                self.operator_volume_stats = pd.read_sql('SELECT * FROM operator_volume_stats', 
                                                         index_col=['Date', 'Operator', 'Line', 
                                                                    'DirectionRun', 'Origin', 
                                                                    'Destination'], **kwargs)
                self.operator_stn_departures = pd.read_sql('SELECT * FROM operator_stn_departures', 
                                                           index_col='index', **kwargs)
                self.link_departures = pd.read_sql('SELECT * FROM link_departures', 
                                                   index_col=['Line', 'DirectionRun', 
                                                              'DirectionLink', 'LocationID', 
                                                              'CallNext','Platform', 
                                                              'Day', 'UTime'],  **kwargs)
                """
    
    def set_scope(self):
        """
        Determines which trains operate on which dates
        - By date: Working (Permanent) and Planned timetables [0: Planned, 1: Overlay, 2: Actual]
        - By weekday: Working (Permanent) only for dates of interest
        
        """
        
        indicators = {'C':3,'N':2,'O':1,'P':0}
        
        t = Timer("Getting dates of operation")
        # Get dates of operation & turn these into two matrices - a Permanent (Working) Schedule and a Daily Schedule (Planned)
        plandates = self.trips.set_index('TrainUID', append=True)[['Date Runs From','Date Runs To',
                                                                   'DaysOperated','STP Indicator']] #,'Bank Holiday Running']]
        daymatch = (plandates['DaysOperated'].str.split('', expand=True)=='1')#[1:7]
        daymatch.columns.name = 'Weekday'
        daymatch = daymatch.loc[~daymatch.index.duplicated(), 1:7].stack().rename('Operates')
        
        plandates['PlanPriority'] = plandates['STP Indicator'].replace(indicators)
        plandates['Date'] = [pd.date_range(s, e, freq='d') for s, e in zip(plandates['Date Runs From'],
                                                                           plandates['Date Runs To'])]
        plandates = plandates.explode('Date')\
                             .drop(['Date Runs From','Date Runs To','DaysOperated'], axis=1)
        plandates['Day'] = plandates['Date'].dt.weekday+1
        
        plandates = plandates.join(daymatch, on=['ScheduleID','TrainUID','Day'])
        plandates = plandates.loc[plandates.Operates]\
                             .drop(['Operates'], axis=1)\
                             .set_index(['Day','Date'], append=True)\
                             .reorder_levels(['TrainUID','Day','Date','ScheduleID'])\
                             .sort_index()[['PlanPriority']]
        plandates = plandates[plandates.PlanPriority<3]
        
        plandates = plandates.reset_index('ScheduleID')\
                              .set_index('PlanPriority', append=True)\
                              .unstack()\
                              .ffill(axis=1)\
                              .fillna(-1)\
                              .astype(int)\
                              .droplevel(0,1)
        plandates.name='ScheduleID'
        t.stop()
        
        self.trips_bydate = plandates
        
        #Trains in Permanent schedule by weekday, constrained to week immediately following timetable download
        t = Timer("Extracting latest trains in Long Term Plan by weekday")
        idx = pd.IndexSlice[:]
        target_weekdays = pd.date_range(pd.to_datetime(self.datewindow[0], yearfirst=True), periods=7, freq='d')
        
        self.trips_byweekday = plandates.loc[(idx,idx,target_weekdays),0]\
                                         .droplevel('Date')\
                                         .rename('ScheduleID')\
                                         .sort_index()
        
        self.set_state(1)
        """
        target_trains = cif_working_bydate.loc[pd.IndexSlice[:],target_weekdays].unstack('Date').index
        cif_working_byweekday = daymatch.groupby(level=[0,1]).any().loc[target_trains, 1:7]
        cif_working_byweekday.columns.name='Weekday'
        cif_working_byweekday = cif_working_byweekday.replace({True:'P',False:np.nan}).stack()
        cif_working_byweekday.index = cif_working_byweekday.index.swaplevel('Weekday','ScheduleID')
        cif_working_byweekday = cif_working_byweekday.sort_index(0)
        """
        t.stop()
 
    def clean_timetable(self):
        """Cleans the schedules into the PTSP format"""
        if self.state != 1:
            raise ValueError("Timetable has already been cleaned, doing so again will not work")
        
        t = Timer("Cleaning timetable...")
        tt = self.schedules
        tt = tt[['Activity','ScheduleID','Line',
                 'Location (TIPLOC + Suffix)','Platform',
                 'Record Identity', 'Scheduled Arrival',
                 'Scheduled Departure', 'Scheduled Pass']].reset_index()\
               .set_index(['ScheduleID','BSSeq']).sort_index()
        
        tt['EventSeq'] = tt.groupby(level='ScheduleID').cumcount()+1
        tt = tt.set_index('EventSeq', append=True)
        tt = tt.reset_index('BSSeq', drop=True)
        tt = tt.rename(columns={'Activity':         'Activity',
                                'Record Identity':  'EventType',
                                'Location (TIPLOC + Suffix)':'LocationID',
                                'Platform':         'Platform',
                                'Line':             'Track',
                                'Scheduled Arrival':'TimeArr',
                                'Scheduled Departure':'TimeDep',
                                'Scheduled Pass':   'TimePass'})
        
        # Clean up location fields
        tt['LocationID'] = tt['LocationID'].str.strip() # remove blank spaces from this field
        # get the tiploc code for the location, stripping out the '2' suffix which represents a station called at twice
        tt['LocationID'] = tt['LocationID'].mask(tt['LocationID'].str[7] == '2', tt['LocationID'].str[:-1]).str.strip() 
        
        tt['Platform'] = tt['Platform'].str.strip()
        tt['Track']    = tt['Track'].str.strip()
        tt['Activity'] = tt['Activity'].str.strip()
        
        # Convert times from CIF representation to pandas Timedelta
        tt[['TimeArr','TimePass','TimeDep']] = tt[['TimeArr','TimePass','TimeDep']].apply(cif_to_time, out='seconds')
        tt[['TimeArr','TimePass','TimeDep']] = tt[['TimeArr','TimePass','TimeDep']].replace({0: np.nan})
        
        # Fix post-midnight timings
        first_departure = tt.groupby(level='ScheduleID')['TimeDep'].transform('first')
        post_midnight_offset = tt[['TimeArr','TimeDep','TimePass']].lt(first_departure, axis=0)*24*60*60
        tt[['TimeArr','TimeDep','TimePass']] += post_midnight_offset
        
        # Calculate run times
        tt['TimeDep'] = tt['TimeDep'].fillna(tt['TimePass'])
        tt['TimeArr'] = tt['TimeArr'].fillna(tt['TimePass'])
        tt['RunTime'] = tt['TimeArr'].shift(-1) - tt['TimeDep']
        tt['UTime'] = tt['TimeDep'].fillna(tt['TimeArr']).fillna(tt['TimePass'])
        tt['Dwell'] = tt['TimeDep'] - tt['TimeArr']
        
        timediffs = tt.groupby(level='ScheduleID')['UTime'].agg(['first','last']).diff(axis=1)['last']
        print("Start and end time check OK: {}".format((timediffs < 0).sum() == 0))
    
        self.schedules = tt
        self.set_state(2) # Data has been cleaned
        t.stop()

    def identify_locations(self):
        t = Timer("Identifying locations...") #15s
        
        if self.state < 2:
            raise ValueError("Cannot identify locations before timetable has been cleaned (State >=2 required)")
        
        tt = self.schedules
    
        tt['Flags'] = self.produce_flags()

        callflag = tt['Flags'].str.match('.[Cud].')
        
        # Add prev & next location
        tt['LocationNext'] = tt['LocationID'].shift(-1).mask(tt['EventType'] == 'LT')
        tt['LocationPrev'] = tt['LocationID'].shift(1).mask(tt['EventType'] == 'LO')
        
        # Add prev & next calls
        tt['CallNext'] = tt['LocationID'].mask(~callflag).bfill().shift(-1).mask(tt['EventType'] == 'LT')
        tt['CallPrev'] = tt['LocationID'].mask(~callflag).ffill().shift(1).mask(tt['EventType'] == 'LO')
        
        # Add call sequence
        if 'CallSeq' in tt.index.names:
            tt = tt.droplevel('CallSeq')
        tt['CallSeq'] = callflag.groupby(level='ScheduleID').cumsum().astype(int)
        tt.set_index('CallSeq', append=True, inplace=True)
        tt.index = tt.index.swaplevel('EventSeq','CallSeq')
        
        self.schedules = tt
        self.set_state(2.1)
        
        t.stop()
        
    def add_distance(self):
        """ Add available distances """
        
        t = Timer("Adding distances...")
        
        if self.state < 2.1:
            raise ValueError("Cannot add distances before locations have been added (state 3.1 required)")
        
        self.schedules = self.schedules.join(self.lookups['Distances'], on=['LocationID','LocationNext'], how='left')
        self.set_state(2.2)
                               
        t = t.stop()

    def add_geometry(self):
               
        if self.state < 2.2:
            raise ValueError("Cannot add distances before distances have been added (state 3.2 required)")
        
        tt = self.schedules
        if 'easting' in tt.columns:
            tt.drop(['easting','northing'], axis=1)
            
        tt = tt.join(self.lookups['Coordinates'], on='LocationID', how='left')
        
        # Interpolate coordinates where they are unavailable
        coords = pd.concat([tt[['easting','northing']], tt[['easting','northing']].shift(-1)],axis=1).mask(tt['EventType']=='LT')
        coords.columns=['x1','y1','x2','y2']
        xdiffs = coords['x2'] - coords['x1']
        ydiffs = coords['y2'] - coords['y1']
    
        tt['DirectionCompass'] = compass_direction(xdiffs, ydiffs)
        tt['DirectionCardinal'] = compass_to_cardinal(tt['DirectionCompass']).astype(str).replace('nan','')
        
        # Fill distances from adjacent coordinates
        aerial_distance = pythag_distance(xdiffs, ydiffs).where(lambda x: x > 0)
        distance_ratio = (tt['Distance'] / aerial_distance).median()
        tt['Distance'] = tt['Distance'].fillna(aerial_distance*distance_ratio)
        
        # Check where distance looks much shorter than aerial
        wormholes = tt[(tt['Distance'] / aerial_distance) <.5].groupby(['LocationID','LocationNext'])['Distance'].agg(['count','mean']).nlargest(30, 'count')
        print("""Reported link distance looks much shorter than aerial distance (mean = reported link distance), implying either distance or coordinates are wrong.
              No action necessarily needed but these are the30 most common values:\n{}""".format(wormholes))
        
        # Where coordinates unavailable, group into blocks bounded by coordinates, then get group's vector (distance and direction) and speed
        g_coordseq = [tt.index.get_level_values('ScheduleID'), tt.easting.notnull().groupby('ScheduleID').cumsum()]
        coordgroup_diffs = -tt.groupby('ScheduleID')[['easting','northing']].ffill().diff(-1).replace(0,np.nan)
        coordgroup_dist = np.sqrt((coordgroup_diffs**2).sum(axis=1))
        fill_compass = ((-np.rad2deg(np.arctan2(coordgroup_diffs['easting'],coordgroup_diffs['northing']))+90) % 360).groupby('ScheduleID').bfill().groupby('ScheduleID').ffill()
        fill_speeds = coordgroup_dist.groupby(g_coordseq).transform('sum') / tt.groupby(g_coordseq)['RunTime'].transform('sum')
        
        tt['Speed'] = (tt['Distance'] / tt['RunTime']).fillna(fill_speeds)
        
        # Further fill distances based on speeds
        tt['Distance'] = tt['Distance'].fillna(aerial_distance*distance_ratio).fillna(tt['Speed'] * tt['RunTime'])
        
        # Fill coordinates within runs, using filled distances & directions
        tt['DirectionCompass'] = tt['DirectionCompass'].fillna(fill_compass)
        #inverse_direction = (tt['DirectionCompass']+180) % 360
        
        delta_easting =  (tt['Distance']/distance_ratio * np.cos(np.deg2rad(tt['DirectionCompass'])))
        delta_northing = (tt['Distance']/distance_ratio * np.sin(np.deg2rad(tt['DirectionCompass'])))
        
        fill_easting_fwd = (tt.groupby(g_coordseq)['easting'].ffill() + delta_easting.groupby(g_coordseq).cumsum()).shift(1)
        fill_northing_fwd = (tt.groupby(g_coordseq)['northing'].ffill() + delta_northing.groupby(g_coordseq).cumsum()).shift(1)
        g_coordseq_inv = [tt[::-1].index.get_level_values('ScheduleID'), tt[::-1].easting.notnull().groupby('ScheduleID').cumsum()]
        fill_easting_bwd = tt['easting'].shift(-1).groupby(g_coordseq).bfill() - delta_easting[::-1].groupby(g_coordseq_inv).cumsum()[::-1]
        fill_northing_bwd = tt['northing'].shift(-1).groupby(g_coordseq).bfill() - delta_northing[::-1].groupby(g_coordseq_inv).cumsum()[::-1]
        
        tt['easting'] = tt['easting'].fillna(fill_easting_fwd).fillna(fill_easting_bwd)
        tt['northing'] = tt['northing'].fillna(fill_northing_fwd).fillna(fill_northing_bwd)
        
        tt['DirectionCompass'] = tt['DirectionCompass'].mask(tt['EventType']=='LT')
        
        self.schedules = tt
        self.set_state(2.3) # Geometries have been added

    def identify_pax_calls(self):

        """
        Identify passenger calls at stations using the Activity field, 
         and use this to create a Flags field
            # T alone, T followed by space, B or F, nothing else
            # (T[BF\s]?)
            # R either alone, or only followed by white space
        """
        
        pattern = r'(?P<calls>T(?!W))[BF\s]?|(?P<request>R)[^A-Z]|[^-](?P<pickup>U)|[^-](?P<setdown>D)' 
        calls = self.schedules['Activity'].str.extract(pattern)\
                              .fillna('').sum(axis=1)\
                              .replace({'T':'C','':'-','R':'C','U':'u','D':'d'})
        
        return calls

    def validate_calls(self, calls=None):
        if calls is None: calls = self.identify_pax_calls()
        call_counts = self.schedules.groupby([self.schedules['LocationID'], calls]).size()\
                                 .unstack()\
                                 .dropna(subset=['C','d','u'], how='all')
        
        locations_list = self.lookups['Locations'].index
        stations_list = self.lookups['Locations'].query('stationflag').index
        locations_not_in_list = call_counts.loc[call_counts.index.difference(locations_list)]
        
        calls_not_stations = call_counts.loc[call_counts.index.difference(stations_list)].query("C>0")
        calls_not_stations = calls_not_stations.loc[calls_not_stations.index.difference(locations_not_in_list.index)]
        calls_not_in_list = locations_not_in_list.query("C>1")
        
        print("These 'Calls' don't look like NR stations:\n{}.".format(calls_not_stations.sort_values('C',0,False)))
        
        if len(calls_not_in_list) > 0:
            print("""Please update the location lookup table (below) with these new 
                  locations before proceeding. You may need to fetch the latest 
                  NR BPLAN data:\n{}\n{}""".format(cfg.cfg['paths']['ref_nr_definitions_file'], calls_not_in_list))
            raise(ValueError("Not all TIPLOCs can be identified"))
        else:
            print("All call locations available in lookup table")
            
        return calls_not_stations
        
           
    def produce_flags(self):
        calls = self.identify_pax_calls()
        self.calls_not_stations = self.validate_calls(calls)
        
        flags = (  self.schedules['LocationID'].map(self.lookups['Locations']['stationflag']).fillna(False).map({True:'S',False:'-'}) 
                 + calls 
                 + self.schedules['LocationID'].map(self.lookups['Locations']['londonflag']).fillna(False).map({True:'L',False:'-'})
                )
        
        # Alter non-station calls to not show a call (i.e. define a call as only being at a station)
        flags.loc[flags.str.match('-[Cud]')] = flags.loc[flags.str.match('-[Cud]')].str.replace('[Cud]','-', regex=True)
        
        self.schedules['Flags'] = flags
        
        event_counts = self.schedules[['LocationID']].copy()
        event_counts['EventCount'] = 1
        event_counts['StationFlag'] = flags.str.contains('S')
        event_counts['CallFlag'] = flags.str.contains('[Cud]')
        event_counts['LondonFlag'] = flags.str.contains('L')
        
        self.logs['eventcounts'] = event_counts.groupby('LocationID').sum()
        
        print("Stations in NUMBAT NR-ASC list but not in timetable:\n{}".format(self.lookups['NR-ASC'].index.difference(self.logs['eventcounts'].index).values))
        return flags

    def no_direction(self, ttd=None):
        
        if ttd is None:
            ttd = self.schedules[['DirectionLink','EventType','LocationID','LocationNext']]
        """ Links where still unable to deliver a direction """
        print("{:.3f}% of event directions filled".format((1 - (ttd['DirectionLink'].isnull().sum() / len(ttd['DirectionLink'])))*100))
        f_nodirection = ttd.DirectionLink.isnull() & ttd.EventType.ne('LT')
        if f_nodirection.any():
            links_without_direction = ttd.loc[f_nodirection].groupby(['LocationID','LocationNext'])\
                                                           .size()\
                                                           .sort_values(0,False)
            #schedules_with_directionless_links = f_nodirection.groupby(level='ScheduleID').any()
            n_links = len(links_without_direction)
            if n_links>0:
                print("Unable to determine direction for {} links".format(n_links))
                print(links_without_direction)
                return n_links
            else:
                print("Links with no direction exist but cannot locate them.")
                return 0
        else:
            print("All links with EventType LO/LI have a direction")
            return 0

    def check_dir_consistency(self, ttd=None):
        if ttd is None:
            ttd = self.schedules[['LocationID','LocationNext','DirectionLink']]
        dir_uniqueness_check = ttd.groupby(['LocationID','LocationNext','DirectionLink'])\
                                             .size()\
                                             .unstack()
        if (dir_uniqueness_check.notnull().sum(axis=1)>1).any():
            print("Not all Link/Next have a consistent direction:\n")
            print(dir_uniqueness_check[dir_uniqueness_check.notnull().sum(axis=1)>1])
        else:
            print("Direction consistency check complete for links with a direction.")
        return dir_uniqueness_check

    def check_updown(self, ttd, term, up=True):
        if up:
            check_down = ttd.loc[(ttd['EventType'] == 'LT') & term]\
                                       .groupby(['LocationPrev','LocationID','DirectionLink'])\
                                       .size()
            incorrect = check_down[check_down.index.get_level_values('DirectionLink')!='U']
        else:
            check_down = ttd.loc[(ttd['EventType'] == 'LO') & term]\
                                       .groupby(['LocationID','LocationNext','DirectionLink'])\
                                       .size()
            incorrect = check_down[check_down.index.get_level_values('DirectionLink')!='D']

        if len(incorrect)>0:
            print("These London terminal links appear to show the wrong direction:\n{}".format(incorrect))
            print("Correcting...")
            start_terminal = (ttd['EventType'] == 'LO') & term
            ttd.loc[start_terminal, 'DirectionLink'] = 'D'
            return ttd['DirectionLink']
    
        else:
            return ttd['DirectionLink']

    def add_directions(self):
        """Adds directions to the data through a multistage process"""
        if 'DirectionLink' in self.schedules.columns:
            self.schedules = self.schedules.drop('DirectionLink', axis=1)
            self.set_state(2.3)
        
        if self.state < 2.3:
            raise ValueError("Cannot add directions before locations & geometry (State 2.3 required)")
        
        ttd = self.schedules[['EventType','Flags','LocationID','LocationNext','LocationPrev','CallNext','CallPrev']].copy()
        
        px = {}
        for k, od, dn in [
                   ('DirectionLN',['LocationID','LocationNext'], 'direction_initial'),
                   ('DirectionPL',['LocationPrev','LocationID'], 'direction_final'),
                   ('DirectionLC',['LocationID','CallNext'],     'direction_initial'),
                   ('DirectionCL',['CallPrev','LocationID'],     'direction_final')]:
            px[k] = ttd.join(self.lookups['Directions'], how='left', on=od)[dn]
        px = pd.concat(px, axis=1)
        px['Consensus'] = px.bfill(axis=1)['DirectionLN']
        
        # First attempt to fill
        ttd['DirectionLink'] = px['Consensus']
        d0 = self.no_direction(ttd)
        self.check_dir_consistency(ttd)
                
        # If location is London terminus, then fix Up and Down fields
        print("Fix up and down for London termini...")
        term = ttd['LocationID'].isin(self.lookups['Terminals'])
        ttd['DirectionLink'] = self.check_updown(ttd, term, up=True)
        ttd['DirectionLink'] = self.check_updown(ttd, term, up=False)
        
        d0 = self.no_direction(ttd)
        self.check_dir_consistency(ttd)
        
        d1 = None
        #dbreak = d0
        while (d1 != d0):
            d0 = self.no_direction(ttd)
            print("\nFilling by dynamic lookup...")
            ttd['DirectionLink'] = fill_by_dynamic_lookup(ttd, ['LocationID','LocationNext'], 'DirectionLink')
            self.check_dir_consistency(ttd)
            if self.no_direction(ttd) == 0:
                print("Ending direction-fill loop as directions successfully filled...")
                break
            
            print("\nFilling by run direction...")
            ttd['DirectionLink'] = fill_by_run(ttd, 'ScheduleID', 'DirectionLink')
            self.check_dir_consistency(ttd)
            if self.no_direction(ttd) == 0:
                print("Ending direction-fill loop as directions successfully filled...")
                break
            
            print("\nForward filling through pass locations from call locations...")
            # Forward fill through passing locations from calling locations
            ttd['DirectionLink'] = ttd.groupby(level=['ScheduleID','CallSeq'])['DirectionLink'].ffill()
            if self.no_direction(ttd) == 0:
                print("Ending direction-fill loop as directions successfully filled...")
                break
        
            print("\nFill where neighbouring events are both same direction...")
            # Fill where neighbouring events are both the same direction
            beforeafter = (ttd['DirectionLink'].shift(1)==ttd['DirectionLink'].shift(-1))
            notterm = ttd['EventType'].ne('LT')
            isblank = self.schedules['DirectionLink'].isnull()
            ttd['DirectionLink'] = ttd['DirectionLink'].mask(beforeafter & notterm & isblank, 
                                                                                   ttd['DirectionLink'].shift(-1))
            if self.no_direction(ttd) == 0:
                print("Ending direction-fill loop as directions successfully filled...")
                break
        
            # Backward fill through calls from subsequent passes
            print("\nBackward filling through call locations from subsequent passes...")
            ttd['DirectionLink'] = ttd.groupby(level=['ScheduleID','CallSeq'])['DirectionLink'].bfill()
            if self.no_direction(ttd) == 0:
                print("Ending direction-fill loop as directions successfully filled...")
                break
        
            print("\nFill terminating events from departure direction of penultimate event...")
            ttd.loc[ttd.EventType=='LT', 'DirectionLink'] = ttd['DirectionLink'].ffill()
            if self.no_direction(ttd) == 0:
                print("Ending direction-fill loop as directions successfully filled...")
                break
            
            d1 = self.no_direction(ttd)
            print("\nDirection-filling loop carried out: d1:{}, d0:{}".format(d1,d0))
            self.check_dir_consistency(ttd)
        
        # Forward fill remaining events
        ttd['DirectionLink'] = ttd.groupby(level=['ScheduleID'])['DirectionLink'].ffill()
        d0 = self.no_direction(ttd)
        self.check_dir_consistency(ttd)
        
        # Get an overall direction
        rundirs = ttd[ttd['Flags'].str.match('S[Cud].')].pivot_table(index='ScheduleID',columns='DirectionLink',values='LocationID',aggfunc='count')
        rundirs = rundirs.div(rundirs.sum(axis=1), axis=0)
        dircommon = rundirs.idxmax(axis=1).where(rundirs.max(axis=1)>0.65)
        dircommon.name='CommonDirection'
        ttd['DirectionRun'] = dircommon.reindex(ttd.index, level='ScheduleID')
        
        # Force cardinal directions for these locations
        t = Timer("Force cardinal directions for services which use a link that specifies a cardinal direction...")
        
        service_contains_cardinals = ttd.loc[ttd['DirectionLink'].isin(['N','S','E','W']),
                                                        'DirectionLink']\
                                                    .groupby(['ScheduleID'])\
                                                    .value_counts().unstack()
        service_contains_cardinals = service_contains_cardinals[service_contains_cardinals.notnull().sum(axis=1)==1].stack().reset_index('DirectionLink')['DirectionLink']
        ttd['DirectionRun'] = service_contains_cardinals.reindex(ttd.index, level='ScheduleID')\
                                .fillna(ttd['DirectionRun']).fillna('')
        
        self.schedules[['DirectionLink','DirectionRun']] = ttd[['DirectionLink','DirectionRun']]
        self.set_state(3)
        t.stop()
    
    def gather_patterns(self, schedules, trips_bydate):
        calls   = schedules['Flags'].str.match('.[Cud].')
        weights = trips_bydate[2]\
                      .value_counts()\
                      .drop(-1, errors='ignore')\
                      .rename('Weight')
        patterns = schedules[calls]\
                            .join(weights, on='ScheduleID')\
                            .groupby('ScheduleID')\
                            .agg(Pattern=('LocationID', tuple), 
                                 Weight=('Weight','sum'),
                                 Direction=('DirectionRun', 'first'))

        patterns['RouteID'] = patterns['Pattern'].apply(hash)
    
        routes = patterns.groupby(['RouteID','Pattern','Direction'])['Weight'].sum()\
                         .reset_index(['Pattern','Direction'])
                         
        return patterns, routes
        #self.set_state(4) (Don't set this here as it will get re-done later after summarising)
        
    def identify_lines(self, routes, prior_probabilities_file, certainty_thresh=.9, zero_replacement=0.00001):
        
        """
        tt_calls = self.schedules['Flags'].str.match('.[Cud].')
                                       
        schedule_calls = self.schedules.loc[tt_calls, ['LocationID']]\
                                       .reset_index(['CallSeq','EventSeq'], drop=True)
        """                               
        schedule = 'RouteID'
        location = 'LocationID'
        schedule_calls = routes['Pattern'].explode().to_frame('LocationID')
        #weights = self.unique_patterns['Weight']

        # Read prior probabilities
        lineprob = pd.read_csv(prior_probabilities_file, index_col=0, header=0).fillna(0)
        anti_lineprob = 1 - lineprob
        
        # Get the prior probabilities for each station call and set as the first Line Prediction
        #classifier_scorer = lambda x: x.product() 
        classifier_scorer = lambda x: np.exp(np.log(x).sum()) # same as x.product() but reduces risk of buffer underrun errors due to tiny values
        #classifier_scorer = lambda x: np.log(np.exp(x).sum()) - np.log(np.exp(0)*x.count()) # Gives a stronger probability difference & may change order slightly
        
        # Score using positive probabilities (probability that this IS the Line for this Schedule)
        #  Replace any zeroes (trains of Line X never call at Location Y) with imagined occurence value 0.001 in case a new locations appears in Line X
        print("Estimating prior probabilities...")
        priors_probability_matrix = schedule_calls.join(lineprob, on=location).set_index('LocationID', append=True)
        priors_probability_results = priors_probability_matrix.dropna().replace(0,zero_replacement).groupby(level=schedule).apply(classifier_scorer)
        priors_probability_results = priors_probability_results.div(priors_probability_results.sum(axis=1), axis=0) # Normalise the weights
        
        # Score using inverse probabilities (probability that this is NOT the Line for this Schedule)
        print("Estimating complement probabilities...")
        priors_complement_matrix = schedule_calls.join(anti_lineprob, on=location).set_index('LocationID', append=True)
        priors_complement_results = 1 - priors_complement_matrix.dropna().groupby(level=schedule).apply(classifier_scorer)
        priors_complement_results = priors_complement_results.div(priors_complement_results.sum(axis=1), axis=0) # Normalise the weights
        
        print("Combine into prior probability matrix...")
        priors = priors_complement_results * priors_probability_results
        priors = priors.div(priors.sum(axis=1), axis=0)
        
        if (priors.max(axis=1)==0).any():
            print("Could not comput priors for some schedules (result was a zero probability of any line).")
            
        # Get highest probability answer from the two methods
        tt_lineprob = pd.DataFrame()
        tt_lineprob['LinePrediction1Score'] = priors.max(axis=1)
        tt_lineprob['LinePrediction1'] = priors.idxmax(axis=1)
        certainty_level = (tt_lineprob['LinePrediction1Score']==1).sum()/len(tt_lineprob)
        near_certainty_level = (tt_lineprob['LinePrediction1Score']>certainty_thresh).sum()/len(tt_lineprob)
        print("Prior probabilities are certain for {:.1f}% of calling patterns & near-certain for {:.1f}%".format(certainty_level*100,near_certainty_level*100))
        
        # Get a count of which locations each Schedule calls at to feed to the classifier
        locn_count = schedule_calls.groupby([schedule,location]).size() #*weights #numeric_only=True
        locn_count = locn_count.unstack(level=location).fillna(0)
        
        print("Carrying out Multinomial Naive Bayes classification...")
        # Set up a Multinomial Naive Bayes classifer to reclassify the data
        from sklearn.naive_bayes import MultinomialNB
        clf = MultinomialNB(alpha=0.01)
        clf.fit(locn_count, tt_lineprob['LinePrediction1']) # data = [n_samples, n_features]; classes = [n_samples]
        
        # Carry out a prediction of the line using the new classifier
        tt_lineprob['LinePrediction2'] = clf.predict(locn_count)
        tt_lineprob2_result_scores = pd.DataFrame(index=locn_count.index, columns=clf.classes_, data=clf.predict_proba(locn_count))
        tt_lineprob2_result_scores_filtered = tt_lineprob2_result_scores.mask(tt_lineprob2_result_scores < 0.001)#.stack()
        tt_lineprob2_result_scores_filtered.name = 'MultinomialNB_Scores'
        tt_lineprob['LinePrediction2Score'] = tt_lineprob2_result_scores_filtered.max(axis=1)
            
        # Where certainty is very high, use this as the final prediction
        filt_certain = (tt_lineprob['LinePrediction1'].eq(tt_lineprob['LinePrediction2']))
        filt_probable = (tt_lineprob['LinePrediction2Score'] > certainty_thresh) & ~filt_certain
        tt_lineprob[['Line','LineScore']] = tt_lineprob[['LinePrediction2','LinePrediction2Score']].where(filt_certain | filt_probable)
        
        # For lower certainty, use highest probability result from the three methods (prior/negative, prior/positive, post/Naive Bayes classifier)
        prediction_uncertain_scores = tt_lineprob.loc[tt_lineprob['Line'].isnull(), tt_lineprob.columns.str.contains('Score')]
        prediction_uncertain_items = tt_lineprob.loc[tt_lineprob['Line'].isnull(), ~tt_lineprob.columns.str.contains('Score')]
        prediction_uncertain_best = prediction_uncertain_scores.idxmax(axis=1).str.replace("Score","")
        
        tt_lineprob.loc[tt_lineprob['Line'].isnull(), 'LineScore'] = prediction_uncertain_scores.lookup(prediction_uncertain_scores.index, prediction_uncertain_best+"Score")
        tt_lineprob.loc[tt_lineprob['Line'].isnull(), 'Line'] = prediction_uncertain_items.lookup(prediction_uncertain_scores.index, prediction_uncertain_best)
        
        uncertain = len(tt_lineprob[~filt_certain & ~filt_probable]) / len(tt_lineprob)

        print("Some uncertainty on {:.1f}% of patterns".format(uncertain * 100))
        
        return tt_lineprob
        
    def remap_lines(self, lines):
        
        print("Forcibly remapping lines where specified in Definitions table...")
        remapping = self.lookups['LineRemapping'][::-1]\
                        .set_index(['CallsAt','IfLine'])\
                        .to_dict()['MapLineTo']
        
        # Forced remapping - for trains which call at KEY and and have LINE, their service group should be mapped to NEWLINE {KEY:{LINE:NEWLINE}}
        for (stn, linefrom), lineto in remapping.items():
            print("Replacing {} at {} with {}".format(linefrom, stn, lineto))
            patterns_toforce = self.routes[['Pattern']].explode('Pattern')\
                                   .query("Pattern=='{stn}'".format(stn=stn))\
                                   .index.unique()
            lines.loc[patterns_toforce, 'Line'] = lines.loc[patterns_toforce, 'Line']\
                                                           .replace(linefrom, lineto)
            lines.loc[patterns_toforce, 'LineScore'] = np.nan
        
        # Test results in London
        sch_with_london = self.schedules['Flags'].str.match('S[Cud]L')\
                              .groupby(level='ScheduleID').any()
        pat_with_london = self.patterns.loc[sch_with_london, 'RouteID'].unique()
        london_lines    =  lines.reindex(pat_with_london)['Line'].value_counts()
        print(f"Test results for services calling in Oyster area:\n{london_lines}")
        
        return lines

    def validate_lines(self, plot_all=False):
        
        line_s = self.patterns.join(self.lines['Line'], on='RouteID')['Line']
        tts = self.schedules.join(line_s, on='ScheduleID')\
                            .join(self.trips['ServiceGroupID'], on='ScheduleID')
        
        filt_allcalls = tts['Flags'].str.match('S[Cud].')
        filt_londoncalls = tts['Flags'].str.match('S[Cud]L')
        
        filt_bad_london_matches = (filt_londoncalls & 
                                (tts.Line.str.startswith('X') | tts.Line.str.contains('Rural')) & 
                                ~((tts.LocationID=='RDNGSTN') & (tts.Line=='XC Intercity')))
                                
        # Lookups for train plotting
        
        if filt_bad_london_matches.any():
            bad_london_results = tts.loc[filt_bad_london_matches].groupby(['Line','LocationID']).size()
            
            print("Mismatched services call in London at:\n{}".format(bad_london_results))
            
            badmatching_servicegroups = tts.loc[filt_bad_london_matches, 'ServiceGroupID'].unique()
            badmatching_ttslice = tts.loc[tts['ServiceGroupID'].isin(badmatching_servicegroups) & filt_allcalls]
            badmatching_locations = badmatching_ttslice.groupby(['ServiceGroupID','Line','LocationID']).size()
            
            print("Relevant service groups are:\n{}".format(badmatching_locations.sort_values(0,False).groupby(level=[0,1]).head(5)))
            self.plot_trains(badmatching_ttslice, frame='national')
            
            # Output the London sample of 'confused trains' to CSV log
            self.badmatching_locations = badmatching_locations
        
        else:
            print("Forced remapping was successful")
        
        # Look for outliers where a station is only served in a limited way by a Line
        outliers = tts.loc[filt_londoncalls]\
                                 .groupby(['LocationID','Line']).size()\
                                 .groupby(level='LocationID')\
                                 .apply(lambda x: x/x.sum())
        print("""These calls are outliers (top 50).
            You can use the LineRemapping table in the Definitions to forcibly 
            remap these if needed, although they could also be genuine but infrequent services.\n{}""".format(outliers.nsmallest(50)))
        
        available_lines = self.lines.Line.unique()
        
        # Plot where the Lines are as a final visual check
        lines_byregion = {
            'London': [l for l in available_lines if ("LO" in l or "EL" in l or "Metro" in l)],
            'Inner' : [l for l in available_lines if (("EL" in l or "Metro" in l or "TL" in l) and (l[0] != 'X'))],
            'Outer' : [l for l in available_lines if (("EL" in l or "Regional" in l or "TL" in l) and (l[0] != 'X'))]
        }
        
        if plot_all:
            for region, plotarea in [('London','london'),('Inner','lse'),('Outer','lse')]:
                for l in lines_byregion[region]:
                    if l in available_lines:
                        tt_slice = tts.loc[filt_allcalls & (tts.Line==l)]
                        self.plot_trains(tt_slice, frame=plotarea, title="{}: ".format(region)+l, figwidth=6)
                    else:
                        print("{} not in available Lines\n{}".format(l, available_lines))

    def create_summary(self):
        tt = self.schedules
        if self.state >= 5:
            line_s = self.patterns.join(self.lines['Line'], on='RouteID')[['RouteID','Line']]
            run_summary = tt.groupby('ScheduleID').agg({
                                    'Distance':'sum',
                                    'RunTime':'sum',
                                    'Dwell':'sum',
                                    'DirectionCompass':'mean',
                                    'DirectionRun':'first'})
            run_summary[['Origin','Destination']] = tt.groupby('ScheduleID')['LocationID'].agg(['first','last'])
                
            run_summary['Calls'] = tt['Flags'].str.match('.C.').groupby('ScheduleID').sum()
            run_summary['RunTime'] = run_summary['RunTime']+run_summary['Dwell']
            run_summary = run_summary.drop('Dwell', axis=1)
            
            run_summary['DistanceAerial']   = tt.groupby('ScheduleID')\
                                                .agg({'easting':['first','last'],'northing':['first','last']})\
                                                .assign(DistanceAerial=lambda x: np.sqrt((x[('easting','first')]-x[('easting','last')])**2 + (x[('northing','first')]-x[('northing','last')])**2))['DistanceAerial']
            run_summary['DistanceSpread']   = run_summary['Distance'] / run_summary['DistanceAerial']
            run_summary['AvgSpeed']         = run_summary['Distance'] / run_summary['RunTime']
            run_summary['InterStopDistance']= run_summary['Distance'] / (run_summary['Calls']-1)
            run_summary['InterStopRunTime'] = run_summary['RunTime'] / (run_summary['Calls']-1)
            run_summary['DirectionSpread']  = tt['DirectionCompass'].diff()\
                                                                .mask(tt['EventType']=='LT')\
                                                                .mask(lambda x: x > 180, lambda x: 360-x)\
                                                                .mask(lambda x: x < -180, lambda x: 360+x)\
                                                                .groupby('ScheduleID').sum()
            
            run_summary['CallsLondon']      = tt['Flags'].str.match('.CL').groupby('ScheduleID').sum()
            
            run_summary['CircularService']  = (run_summary['Origin']==run_summary['Destination']) | (run_summary['DistanceSpread']>50)
            run_summary['CircularDirection']= (run_summary['DirectionSpread']<0).map({True:'O',False:'I'})
            run_summary['DirectionRun']     = run_summary['DirectionRun'].replace('',np.nan).mask(run_summary['CircularService']).fillna(run_summary['CircularDirection'])
            
            self.schedules['DirectionRun'] = self.schedules[['LocationID']].join(run_summary['DirectionRun'])['DirectionRun']
            self.patterns, self.routes = self.gather_patterns(self.schedules, self.trips_bydate) # Re-do the patterns as the Direction will have changed for some trips

            if self.state >= 5:
                run_summary[['RouteID','Line']] = line_s
                run_summary['LineCorridor']     = run_summary['Line'].str[0:2]
                run_summary['LineType']         = run_summary['Line'].str.extract('(Metro)|(Regional)|(Intercity)|(Suburban)|(Rural)')\
                                                            .stack()\
                                                            .dropna()\
                                                            .reset_index(level=1, drop=True)\
                                                            .mask(run_summary.LineCorridor=='LO','Metro')\
                                                            .mask(run_summary.LineCorridor.isin(['TL','EL']), 'MetroRegional')
        else:
            print("Insufficient state to create a summary. State is {} and should be 4 or more".format(self.state))
        
        self.summary = run_summary
        #self.set_state(6)
        return self.summary

    def create_link_summary(self, save=False):
        
        """ Produce a departure summary by Line, Station, Next Call & Direction, for departures per hour """
        
        tt_perm = self.full_tt()

        f_london = (tt_perm.Flags.str.contains('L') | 
                    tt_perm.groupby(level='ScheduleID')['Flags']\
                           .shift(-1).str.contains('L'))
        
        link_departures={}
        for day, code in {'We':3,'Sa':6,'Su':7}.items():
            #filt_day = tt_perm.index.get_level_values('Weekday') & filt_london_any
            hrs = (tt_perm[f_london].xs(code, axis=0, level='Day')['UTime'] // 3600).astype(int)
            link_departures[day] = tt_perm[f_london].xs(code, axis=0, 
                                                        level='Day')\
                                        .groupby(['Line','DirectionRun',
                                                  'DirectionLink','LocationID',
                                                  'CallNext','Platform',
                                                  hrs]).size()
        
        link_departures = pd.concat(link_departures, axis=1)
        link_departures.columns = pd.Categorical(link_departures.columns, 
                                                 categories=['We','Sa','Su'], 
                                                 ordered=True)
        link_departures.columns.name = 'Day'
        link_departures = link_departures.stack()
        link_departures.index = link_departures.index.swaplevel('Day','UTime')
        link_departures.sort_index(inplace=True)
        link_departures.name = 'Departures'
        
        if save:
            link_departures.to_csv(self.output_file_locations['output_summary_link_frequency'], header=True)

        self.link_departures = link_departures

        return link_departures

    def create_station_summary(self, weekday=3, save=False):
        tt_perm = self.full_tt()
        operator_stn_departures = tt_perm[tt_perm.Flags.str.match('SC.').fillna(False)]\
                                    .xs(weekday, axis=0, level='Day')\
                                    .join(self.trips['Operator'])\
                                    .groupby(['LocationID','Operator'])\
                                    .size()
        operator_stn_departures.name = 'DailyDepartures'
        operator_stn_departures = operator_stn_departures.reset_index()
        operator_stn_departures = operator_stn_departures.join(self.lookups['Tiplocs'][['NLC4','NR_3alpha','NameFull']], on='LocationID')
        
        if save:
            operator_stn_departures.to_csv(self.output_file_locations['output_stn_departures'], header=True)
        
        self.operator_stn_departures = operator_stn_departures
        return operator_stn_departures
    
    def create_volume_summary(self, save=False):
        operator_volume_stats = self.trips_bydate.loc[:, 2]\
                            .to_frame('ScheduleID')\
                            .join(self.trips, on='ScheduleID')\
                            .join(self.summary, on='ScheduleID')\
                            .groupby(['Date','Operator','Line','DirectionRun',
                                      'Origin','Destination'])\
                            .agg({'ScheduleID':'count','Distance':'sum',
                                  'Calls':'sum','CallsLondon':'sum',
                                  'RunTime':['sum','mean']})
        operator_volume_stats.columns= ['NumDepartures','TotalKMs','TotalCalls','TotalLondonCalls','TotalRunTime','MeanRunTime']
        if save:
            operator_volume_stats.to_csv(self.output_file_locations['output_operated_volumes'], header=True)
        
        self.operator_volume_stats = operator_volume_stats
        return operator_volume_stats
        
    def save_logs(self):
        # Output event count
        if hasattr(self, 'logs'):
            if 'eventcounts' in self.logs:
                self.logs['eventcounts'].to_csv(self.output_file_locations['eventcounts'])
        
        # Output line probabilities
        self.schedules[self.schedules['Flags'].str.match('S[Cud].')]\
            .join(self.patterns['RouteID'], on='ScheduleID')\
            .join(self.lines['Line'], on='RouteID')\
            .groupby(['LocationID','Line'])\
            .size()\
            .unstack([1])\
            .apply(lambda x: x / x.sum(), axis=1)\
            .to_csv(self.output_file_locations['output_line_probabilities_file'], index=True)

        if not hasattr(self, 'calls_not_stations'):
            self.calls_not_stations = self.validate_calls()
        self.calls_not_stations.to_csv(self.output_file_locations['badcalls_file'])

        if hasattr(self, 'lines'):
            self.lines.to_csv(self.output_file_locations['output_line_identification_results'])

        if hasattr(self, 'badmatching_locations'):
            self.badmatching_locations.to_csv(self.output_file_locations['classification_log'], header=True)

        return None
    
    def save_csvs(self):
        
        t = Timer("Saving to disk...")
        # A. CIF/ScheduleID Summary: output_uids_summary
        self.trips\
            .join(self.summary)\
            .to_csv(self.output_file_locations['output_uids_summary'], 
                    chunksize=50000)
                 
        # B. Train Lists: output_uids_perm_byweekday, output_uids_perm_bydate, output_uids_planned_bydate
        run_summary_subset = self.summary[['Origin','Destination','Distance','RunTime','DirectionRun','Calls','CallsLondon','Line','PatternID']]

        # B1. Working/Permanent/LTP Trains By Weekday (-> NUMBAT)
        self.trips_byweekday.to_frame()\
                             .join(run_summary_subset, on='ScheduleID')\
                             .to_csv(self.output_file_locations['output_uids_byweekday'], chunksize=50000)
        # B2. LTP and STP Trains By Date
        self.trips_bydate.rename(columns={0:'LTP',1:'LTP_Overlay',2:'STP'})\
                         .stack()\
                         .to_frame('ScheduleID')\
                         .join(run_summary_subset, on='ScheduleID')\
                         .to_csv(self.output_file_locations['output_uids_bydate'], chunksize=50000)
        t.stop()
        
        # D. All Schedules: output_timetable_allbyuid
        t = Timer("Writing all schedules...")
        self.schedules.to_csv(self.output_file_locations['output_timetable_allbyuid'], chunksize=50000)
        t.stop()
        
        # C. Complete Perm Sched: output_timetable_permbyday
        t = Timer("Writing permanent timetable by weekday...")
        self.full_tt().to_csv(self.output_file_locations['output_timetable_permbyday'], chunksize=50000)
        t.stop()

    def save_db(self, output_db=None, increment=None):
        #engine = create_engine(r"sqlite:///{}".format(output_database))
        #conn = engine.connect()
        #conn = sqlite3.connect(self.output_database)
        if output_db is None: output_db = self.output_db
        
        if increment is not None:
            if not isinstance(increment, list):
                increments = [increment]
            else:
                increments = increment
        else:
            increments = [x for x in [1,4,5,6] if self.state >= x]
        
        with sqlite3.connect(output_db) as conn:
            kwargs = {'con': conn, 'if_exists': 'replace', 'index': False}
            pd.Series(self.state).to_sql('state', **kwargs)
            
            if any([y in increments for y in [1,2,3,5]]):
                print("Saving schedules [1-5]")
                # resave schedules in all cases except 4 and 6
                self.schedules.reset_index().to_sql('schedules', **kwargs)
                
            if 1 in increments:
                print("Saving trips [1]")
                self.trips.reset_index().to_sql('trips', **kwargs)
                self.trips_bydate.reset_index().to_sql('trips_bydate', **kwargs)
                self.trips_byweekday.reset_index().to_sql('trips_byweekday', **kwargs)
            if 4 in increments:               
                print("Saving patterns [4]")
                self.patterns.drop('Pattern', axis=1).reset_index().to_sql('patterns', **kwargs)
                self.routes.explode('Pattern').reset_index().to_sql('routes', **kwargs)
            if 5 in increments:
                print("Saving lines [5]")
                self.lines.reset_index().to_sql('lines', **kwargs)
            if 6 in increments:
                print("Saving summary [6]")
                self.summary.reset_index().to_sql('summary', **kwargs)
                try:
                    if hasattr(self, 'operator_volume_stats'):
                        self.operator_volume_stats.reset_index().to_sql('operator_volume_stats', **kwargs)
                    else:
                        print("No operator_volume_stats table - run create_volume_summary to produce this")
                    if hasattr(self, 'operator_stn_departures'):
                        self.operator_stn_departures.reset_index().to_sql('operator_stn_departures', **kwargs)
                    else:
                        print("No operator_stn_departures table - run create_station_summary to produce this")
                    if hasattr(self, 'link_departures'):
                        self.link_departures.reset_index().to_sql('link_departures', **kwargs)
                    else:
                        print("No link_departures table - run create_link_summary to produce this")
                except Exception as err:
                    print(f"Could not save summaries to {output_db} because:\n {err}\nOther tables have been saved")

    # Convenience functions for extracting portions of the data
    def full_tt(self):
        line_s = self.patterns.join(self.lines, on='RouteID')['Line']
        all_trips = self.trips_byweekday.join(line_s, on='ScheduleID')
        tt_perm = all_trips.join(self.schedules\
                                     .reset_index(['CallSeq','EventSeq']), 
                                                  on='ScheduleID')\
                           .set_index(['ScheduleID','CallSeq','EventSeq'], 
                                      append=True)
        return tt_perm

    def get_line_timetable(self, line):
        #for line in tt_nr.summary.line.unique():
        trips_line = self.summary.query(f"Line=='{line}'")
        trips_line_weekly = self.trips_byweekday.to_frame().join(trips_line, on="ScheduleID", how='inner')
        result = trips_line_weekly[['ScheduleID','Origin','Destination']]\
                    .join(self.schedules.reset_index(['CallSeq','EventSeq']), on='ScheduleID')\
                    .set_index(['ScheduleID','CallSeq','EventSeq'], append=True)
        result[result.Flags=='SCL'].groupby(['Day','DirectionRun','LocationID','CallNext',(result.UTime//900).astype(int)])\
                 .size()\
                 .unstack(['Day','UTime'])\
                 .sort_index(axis=1)\
                 .fillna(0)\
                 .astype(int)
                 #.to_csv(Path(r'M:\12 Library\04 Timetable analysis\_outputs\nr\20210417\{line}_frequency_check.csv').as_posix().format(line=line))
        return result
    
    def daily_scheduled_services_plot(self, line, direction):
        
        trips_line = self.summary.query(f"Line=='{line}'")[['Line','DirectionRun','Origin','Destination','RouteID']]
        trips_line_daily = self.trips_bydate[['2']]\
            .join(trips_line, on='2',how='inner')\
        	.join(self.routes[['Pattern']],on='RouteID')\
        	.sort_index(level='Date')
        
        # Get a chart of daily trips by date for Up trains for each OD pair
        od_line_daily = trips_line_daily.groupby(['Date','DirectionRun','Origin','Destination'])\
        	.size()\
        	.unstack(['DirectionRun','Origin','Destination'])\
        	.loc[:, direction]
        
        fig, ax = plt.subplots(figsize=(15,8), )
        od_line_daily.plot(cmap='Dark2', linewidth=2, ax=ax, ylim=(0))
        ax.set_ylabel('Daily services')
        
        return od_line_daily

    def trains_calling_at(self, location):
        schedules_at = self.schedules[(self.schedules.LocationID==location) & self.schedules.Flags.str.match('SC.')].index.get_level_values('ScheduleID').unique()
        patterns_at = self.patterns.reindex(schedules_at).join(self.lines['Line'], on='RouteID')
        return patterns_at

    def plot_trains(self, tt_slice, frame='london', xy_cols=['easting','northing'], figwidth=10, title=None):
        points = self.refpoints[frame][0]
        frame = self.refpoints[frame][1]
        coords = self.lookups['Coordinates'].reindex(self.schedules.LocationID.unique())
         
        # Extract a dataframe of unique links
        result = tt_slice[xy_cols].dropna()
        result.columns = ['x0','y0']
        result[['x1','y1']] = result.groupby(level='ScheduleID').shift(-1)
        result = result.dropna().groupby(['x0','y0','x1','y1']).size().to_frame('count').reset_index()
        
        # Extract the coordinates and turn into matplotlib segments
        s = result['count']/7500*100
        segs = result[['x0','y0','x1','y1']].values.reshape((-1,2,2))
        
        # Set the figure size
        ptwidth =  frame[0][1]-frame[0][0]
        ptheight = frame[1][1]-frame[1][0]
        figheight = ptheight/ptwidth*figwidth
        
        fig, ax = plt.subplots(figsize=(figwidth,figheight))
        ax.set_xlim(frame[0][0],  frame[0][1])
        ax.set_ylim(frame[1][0],  frame[1][1])
        
        line_segments = LineCollection(segs, linewidths=s, linestyle='solid')
        ax.add_collection(line_segments)
        
        # Add reference points
        coords.loc[points].plot.scatter(ax=ax, x=xy_cols[0], y=xy_cols[1], color='black', alpha=0.5)  
        for i, point in coords.loc[points].iterrows():
            ax.text(point[xy_cols[0]], point[xy_cols[1]], i)
        
        ax.set_title(title)
        plt.show()
   
class TimetableTfL:

    def __init__(self, databundle, modes=['u','d','t','b','f','c']):
        self.state = None
        
        self.databundle = databundle
        self.input_db  = cfg.cfg['paths']['tfl_extracted_file'].format(databundle=self.databundle)
        self.output_db = cfg.cfg['paths']['tfl_processed_file'].format(databundle=self.databundle)
        
        self.transxchange = self.load(modes)

        self.trips = self.get_trips()

        services_byweekday = self.service_operating_days(self.transxchange['Services_RegularDayType_DaysOfWeek'])
        services_bydate = self.service_operating_dates(self.transxchange['Services'], services_byweekday)

        self.trips_byweekday = self.trip_operating_days(services_byweekday)
        self.trips_bydate = self.trip_operating_dates(services_bydate)
        
        self.schedules = self.get_schedules()
        
    def save_csv(self, output_folder=None):

        if output_folder is None:
            output_folder = Path(cfg.cfg['paths']['nr_output_folder'].format(databundle=self.databundle))
        
        if output_folder.exists():
            logging.info(f"Output folder (for transformed timetable for {self.databundle}) already exists")
        else:
            output_folder.mkdir(parents=True)
            
        self.trips.to_csv(output_folder / Path('TfLTripList_{}.csv'.format(self.databundle)))
        self.services_byweekday.to_csv(output_folder / Path('TfLServicesByWeekday_{}.csv'.format(self.databundle)))
        self.services_bydate.to_csv(output_folder / Path('TfLServicesByDate_{}.csv'.format(self.databundle)))
        self.schedules.to_csv(output_folder / Path('TfLSchedules_{}.csv'.format(self.databundle)))
        
    def save_db(self, output_db=None, increment=None):

        if output_db is None: output_db = self.output_db
        if increment is None: increment = [2,4,5,6]
        
        with sqlite3.connect(output_db) as conn:
            kwargs = {'con': conn, 'if_exists': 'replace', 'index': False}
            pd.Series(self.state).to_sql('state', **kwargs)
            
            if 2 in increment:               
                self.trips.reset_index().to_sql('trips', **kwargs)
                self.schedules.reset_index().to_sql('schedules', **kwargs)
                self.trips_bydate.reset_index().to_sql('trips_bydate', **kwargs)
                self.trips_byweekday.reset_index().to_sql('trips_byweekday', **kwargs)
            if 4 in increment:               
                self.patterns.drop('Pattern', axis=1).reset_index().to_sql('patterns', **kwargs)
                self.routes.explode('Pattern').reset_index().to_sql('routes', **kwargs)
            if 5 in increment:
                self.lines.reset_index().to_sql('lines', **kwargs)
            if 6 in increment:
                self.summary.reset_index().to_sql('summary', **kwargs)
                self.operator_volume_stats.reset_index().to_sql('operator_volume_stats', **kwargs)
                self.operator_stn_departures.reset_index().to_sql('operator_stn_departures', **kwargs)
                self.link_departures.reset_index().to_sql('link_departures', **kwargs)

    def load(self, modes):
        # Services: Service -> generic stuff (i.e. "Tram")
        # Routes: Origin to Destination routes with description and RouteSectionRef
        # RouteLinks: Links within each Route (RouteSectionRef->RouteSections)
        # JourneyPatterns: link a JourneyPattern to a RouteRef and JourneyPatternSectionRef
        # JourneyPatternTimingLinks: link JourneyPatternSections to Links and run times (to nearest minute)
        # VehicleJourneys: links JourneyCodes to Service, LineRef, JourneyPatternRef and has DepartureTime

        with sqlite3.connect(self.input_db) as con:
            cur = con.cursor()
            get_tables = cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
            get_tables = pd.Series(get_tables).explode().unique()
    
            tx = {key: [] for key in get_tables}
            for table in get_tables:
                logger.info(f"Reading {table}")
                q = pd.read_sql_query(f"SELECT * FROM '{table}' WHERE Mode IN {str(tuple(modes))}", con).drop('index', axis=1)
                tx[table].append(q)
            
        
        for t, ldf in tx.items():
            tx[t] = pd.concat(ldf)
            if t != 'Routes':
                tx[t] = tx[t].drop(['Line','Mode'], axis=1)
        
        indices = {'JourneyPatterns':'JourneyPattern',
                   'Routes':'Route',
                   'RouteLinks':'RouteLink',
                   'JourneyPatternTimingLinks': ['JourneyPatternSections','JourneyPatternTimingLink'],
                   'Services':'ServiceCode',
                   'VehicleJourneys':'VehicleJourneyCode'}
        
        for t, v in indices.items():
            tx[t] = tx[t].set_index(v)
        
        return tx

    def service_operating_days(self, servicedays):
        # Load TfL timetables
        
        operatingdays = servicedays.set_index('Services')['DaysOfWeek']\
                                   .map({
                'Sunday':           '......1', 
                'MondayToSunday':   '1111111', 
                'Tuesday': '.1.....', 'Weekend': '.....11', 'Monday': '1......',
                'Friday': '....1..', 'Saturday': '.....1.', 'Wednesday': '..1....', 
                'Thursday': '...1...', 'MondayToFriday': '11111..',
                'MondayToSaturday': '111111.'})

        operatingdays = operatingdays.str.split('',expand=True)\
                                     .replace({'.':False,'1':True})\
                                     .groupby('Services').any()
        operatingdays.index.name='ServiceGroupID'
        operatingdays.columns.name='Weekday'
        operatingdays = operatingdays.loc[:,1:7].replace({False:None}).stack()
        operatingdays.name = 'Operating'
        
        return operatingdays

    def trip_operating_days(self, operatingdays):
        
        busdaymap = {'MFSc':[1,2,3,4,5], 'Sa':[6],
       'Su':[7], 'MFHo':[1,2,3,4,5], 'MF':[1,2,3,4,5], 'Fr':[5], 'Mo':[1], 'TWTh':[2,3,4], 'MT':[1,2], 'SuNt':[7], 'FrNt':[5],
       'SaNt':[6], 'MTNt':[1,2], 'MTSc':[1,2], 'FrSc':[5], 'Tu':[2], 'We':[3], 'Th':[4], 'MoSc':[1],
       'ThSc':[4], 'WeSc':[3], 'TuSc':[2], 'MSa':[1,2,3,4,5,6], 'SMNt':[7,1], 'TWTsc':[2,3,4], 'MoNt':[1], 'TuNt':[2],
       'WeNt':[3], 'ThNt':[4], 'xMoSc':[2,3,4,5,6,7]}
        busdays = pd.Series(busdaymap).apply(pd.Series).stack().astype(int).reset_index(-1, drop=True).rename('BusWeekday')
        
        tripdays = self.trips[['ServiceGroupID']].join(operatingdays.reset_index('Weekday'), on='ServiceGroupID')\
                                            .drop('ServiceGroupID', axis=1)
        tripdays['BusDay'] = tripdays.index.str.split('-').str[-1]
        tripdays = tripdays.join(busdays, on='BusDay')
        tripdays['Operating'] = (tripdays.BusWeekday==tripdays.Weekday) | tripdays.BusWeekday.isnull()
        tripdays['SchoolDaysOnly'] = tripdays['BusDay'].str.contains('Sc|sc') & tripdays['BusDay'].notnull()
        
        return tripdays.loc[tripdays.Operating, ['Weekday','Operating','SchoolDaysOnly']]
        
    def service_operating_dates(self, services, operatingdays):
        operatingdates = services.apply(lambda x: pd.Series(pd.date_range(x['OpPeriod_StartDate'], x['OpPeriod_EndDate'])), axis=1)\
                                           .stack()\
                                           .to_frame('Date')\
                                           .reset_index(-1, drop=True)
        operatingdates['Weekday'] = operatingdates['Date'].dt.weekday+1
        operatingdates.index.name = 'ServiceGroupID'
        operatingdates = operatingdates.join(operatingdays, on=['ServiceGroupID','Weekday'])\
                                       .dropna(subset=['Operating'])\
                                       .set_index('Date', append=True)
        
        return operatingdates
   
    def trip_operating_dates(self, operatingdates):
        tripdates = self.trips[['ServiceGroupID']].join(operatingdates.reset_index('Date'), on='ServiceGroupID')\
                                            .drop('ServiceGroupID', axis=1)
        return tripdates

    def get_trips(self):
        # Process departures
        logger.info("Processing departures...")
        
        routes   = self.transxchange['Routes']
        services = self.transxchange['Services']\
                       .drop(['OpPeriod_StartDate','OpPeriod_EndDate'], axis=1)\
                       .rename(columns={'StandardService_Origin':'Origin',
                                        'StandardService_Destination':'Destin'})
        
        linecode = services.index.str.split('-', expand=True).to_frame().set_index(services.index)
        
        """
        modemap = {  1: 'u', # Underground
                    35: 'f', # Woolwich Ferry
                    30: 'f', 31: 'f', 32: 'f', # Ferry Tour
                    33: 'f', 
                    25: 'd',
                    63: 't',
                    71: 'c'} # Cable Car
        """
        
        linemap = {'NTN':'NOR', 'TR':'TRM'}

        services['Line'] = linecode[1].replace(linemap) #.mask(linecode[0]='99','RRB')
        services['ServiceGroupSeq'] = linecode[4].astype(int)
        
        # ServiceGroupID (LineRef/ServiceRef) e.g. 1-NTN-_-y05-1756276 (Mode-Line-(Normal)-y05-ServiceGroupSeq)
        # ScheduleID (VehicleJourney) e.g.VJ_1-NTN-_-y05-1756276-1-UP (as above + TripSeq + Direction)

        # RouteID (Route) e.g. R_1-NTN-_-y05-1756276-O-1 (ServGrp+DirectionBinary-RouteSeq)
        # RouteLinkID (RouteLinkRef) e.g. R_1-NTN-_-y05-1756276-O-1-1 (ServGrp+DirectionBinary-RouteSeq-LinkSeq)
        # PatternIDRef (JourneyPatternRef) e.g. JPS_1-NTN-_-y05-1756276-1-O-1 (ServGrp + PatSeq-DirBin-RouteSeq)
        # PatternIDSec (JourneyPatternSection) e.g. JPS_1-NTN-_-y05-1756276-1-1-O (ServGrp + PatSeq-RouteSeq-DirBin)
        # LinkID (JourneyPatternTimingLink) e.g. JPL_1-NTN-_-y05-1756276-1-O-1-2  (ServGrp+PatSeq-DirBin-RouteSeq-LinkSeq)
        
        trips = self.transxchange['VehicleJourneys']
        trips.index.name = 'ScheduleID'
        trips = trips.drop(['PrivateCode','ServiceRef'], axis=1)
        trips = trips.join(services, on='LineRef', how='right')
        trips = trips.rename(columns={'JourneyPatternRef': 'PatternID',
                                      'LineRef':'ServiceGroupID',
                                      'DepartureTime':'OriginDepSch'})
        trips['OriginDepSch'] = pd.to_timedelta(trips['OriginDepSch'])
        
        tripseq = trips.index.str.rsplit('-',n=2, expand=True).to_frame().set_index(trips.index)#['Trips']
        trips['DirectionRun'] = tripseq.iloc[:, -1]
        trips['TripSeq'] = tripseq.iloc[:, -2].astype(int)
         
        trips = trips.join(self.transxchange['JourneyPatterns'], on='PatternID')
        trips = trips.rename(columns={'Direction':'DirBin',
                                      'RouteRef':'RouteID',
                                      'JourneyPatternSectionRefs':'PatternIDSec'})
        trips['DirBin'] = trips['DirBin'].str[0].str.upper()
        trips['RouteSeq'] = trips['RouteID'].str.rsplit('-',n=1).str[-1]
        trips['PatternSeq'] = trips['PatternIDSec'].str.rsplit('-',n=3).str[-3]
        trips = trips.join(routes['Mode'], on='RouteID')
                
        return trips
                
    def get_schedules(self):
        logger.info("Converting run and wait times to right format...")
        # All nodes: tflnet_timinglinks['From_StopPointRef'].unique(), tflnet_timinglinks['To_StopPointRef'].unique()
        links = self.transxchange['JourneyPatternTimingLinks']
        links['RunTime'] = pd.to_timedelta(links['RunTime'].str[2:])
        links['WaitTime'] = pd.to_timedelta(links['WaitTime'].fillna('PT0M').str[2:])
        links['RunTimeS'] = links['RunTime'].astype('timedelta64[s]') 
        links['WaitTimeS'] = links['WaitTime'].astype('timedelta64[s]') 
        #links = links.reset_index().set_index('JourneyPatternSections')

        tmr = Timer("Joining stopping patterns and journey departures together...")
        schedules = self.trips[['PatternIDSec','OriginDepSch']]\
                          .join(links.reset_index(-1), on='PatternIDSec')\
                          .drop(['PatternIDSec'], axis=1)\
                          .rename(columns={'JourneyPatternTimingLink':'TimingLinkID',
                                           'From_SequenceNumber':'LinkSeqFrom',
                                           'From_Activity':'ActivityFrom', 
                                           'From_StopPointRef':'LocationID', 
                                           'To_SequenceNumber':'LinkSeqTo',
                                           'To_Activity':'ActivityTo', 
                                           'To_StopPointRef':'LocationNext',
                                           'RouteLinkRef':'RouteLinkID'})
        schedules.index.name='ScheduleID'
        schedules['EventSeq'] = schedules.groupby('ScheduleID').cumcount()+1
        schedules = schedules.set_index('EventSeq', append=True)
        tmr.stop()
        
        tmr = Timer("Calculating cumulative run times...")
        schedules['CmlRunTimeS'] = schedules.groupby(level=0)['RunTimeS'].cumsum()
        schedules['CmlWaitTimeS'] = schedules.groupby(level=0)['WaitTimeS'].cumsum()
        schedules['CmlToArr'] = schedules['CmlRunTimeS'] + schedules.groupby(level=0)['CmlWaitTimeS'].shift(1).fillna(0)
        tmr.stop()
        
        tmr = Timer("Converting to departure times...")
        schedules['TimeArrNext'] = schedules['OriginDepSch'] + pd.to_timedelta(schedules['CmlToArr'], 's')
        schedules['TimeDepNext'] = schedules['TimeArrNext'] + schedules['WaitTime']
        schedules['TimeArr'] = schedules.groupby(level=0)['TimeArrNext'].shift(1)
        schedules['TimeDep'] = schedules.groupby(level=0)['TimeDepNext'].shift(1)
        schedules.loc[schedules['TimeDep'].isnull(), 'TimeDep'] = schedules.loc[schedules['TimeDep'].isnull(), 'OriginDepSch']
        tmr.stop()
        
        tmr = Timer("Categorising departures by time band...")
        
        schedules['TimeBand'] = pd.cut(schedules['TimeDep'],
                   [pd.Timedelta('7 hours'), pd.Timedelta('10 hours'), 
                    pd.Timedelta('16 hours'), pd.Timedelta('19 hours'), 
                    pd.Timedelta('22 hours')], labels=['AM','IP','PM','EV'])
        
        tmr.stop()
        
        activity_map = {'pickUp':'SuL', 
                        'pickUpAndSetDown':'SCL', 
                        'pass':'S-L', 
                        'setDown':'SdL'}
        
        schedules['ActivityFrom'] = schedules['ActivityFrom'].map(activity_map)
        schedules['ActivityTo'] = schedules['ActivityTo'].map(activity_map)
        schedules['EventType'] = 'LI'
        
        lt = schedules.groupby('ScheduleID').tail(1).copy().reset_index('EventSeq')
        lt[['ActivityFrom','LinkSeqFrom','LocationID','TimeArr']] = lt[['ActivityTo','LinkSeqTo','LocationNext','TimeArrNext']]
        lt[['ActivityTo','LinkSeqTo','LocationNext','TimeArrNext']] = np.nan
        lt[['TimingLinkID','RouteLinkID']] = np.nan
        lt['EventType'] = 'LT'
        lt['EventSeq'] = lt['EventSeq']+1
        lt = lt.set_index('EventSeq', append=True)
        
        schedules = schedules.append(lt)
        schedules.loc[schedules.index.get_level_values('EventSeq')==1, 'EventType'] = 'LO'
        schedules = schedules.drop(['OriginDepSch','ActivityTo','LinkSeqTo',
                                    'TimeArrNext','TimeDepNext',
                                    'CmlRunTimeS','CmlWaitTimeS',
                                    'CmlToArr','RunTime',
                                    'WaitTime'], axis=1)
        schedules = schedules.rename(columns={'ActivityFrom':'Activity',
                                              'LinkSeqFrom':'LinkSeq',
                                              'WaitTimeS':'DwellTimeS'})
        schedules = schedules.sort_index()
        
        return schedules

    def get_frequencies(self,days=[3,5,6,7]):

        
        def intervalfunc(x):
            return x.diff().median()
        def effintervalfunc(x):
            return (((x / pd.Timedelta('1s')).diff()**2).sum() / (x / pd.Timedelta('1s')).diff().sum()) / 60
        
        
        tmr = Timer("Getting frequencies for each day & time band...")
        
        d = {}
        for days_n in days:
            trips_inscope = self.trips_byweekday.query(f"Weekday=={days_n}").index
            d[days_n] = self.schedules.loc[trips_inscope, 
                                           ['LocationID','LocationNext','TimeBand','TimeDep']]\
                        .join(self.trips['Mode'])\
                        .sort_values('TimeDep')\
                        .groupby(['Mode','LocationID','LocationNext','TimeBand'])['TimeDep']\
                        .agg(['size','min','max',intervalfunc,effintervalfunc])

        freq = pd.concat(d, keys='weekday').reset_index()
        
        tmr.stop()
        
        tmr = Timer("Calculating anicillary statistics...")
        
        freq = freq.rename(columns = {'Mode':'mode',
                                      'LocationID':'location',
                                      'LocationNext':'call_next',
                                      'TimeBand':'timeband',
                                      'size':'count',
                                      'min':'first',
                                      'max':'last',
                                      'intervalfunc':'interval_median',
                                      'effintervalfunc':'interval_effective'})
        
        freq['service_duration'] = (freq['last'] - freq['first'] + freq['interval_median']) / pd.Timedelta('1 hour')
        freq['period_duration'] = freq['timeband'].map({'AM':3,'IP':6,'PM':3,'EV':3})
        
        freq['availability'] = freq['service_duration'] / freq['period_duration']
        
        freq['tph_periodmean'] = freq['count'] / freq['period_duration']
        freq['tph_servicehours_mean'] = (freq['count'] - 1) / freq['service_duration']
        freq['tph_servicehours_mean'] = freq['tph_servicehours_mean'].fillna(freq['tph_periodmean'])
        
        freq['interval_median'] = freq['interval_median'] / pd.Timedelta('1 min')
        freq['tph_median'] = 60 / freq['interval_median']
        freq['tph_effective'] = 60 / freq['interval_effective']
        freq['tph_effective'] = freq['tph_effective'].fillna(freq['tph_periodmean'])
        freq['tt_evenness'] = freq['tph_effective'] / freq['tph_servicehours_mean']
        
        tmr.stop()
        
        # Geometry
        """
        tmr = Timer("Adding geometry...")
        stops = pd.read_csv(str(naptan_stops), encoding='windows-1252')
        stops = stops.set_index('ATCOCode')[['CommonName','Easting','Northing','StopType','AdministrativeAreaCode']]
        
        stopareas = pd.read_csv(str(naptan_stop_areas), encoding='windows-1252')
        stopareas = stopareas.set_index('StopAreaCode')[['Name','Easting','Northing','StopAreaType','AdministrativeAreaCode']]
        stopareas = stopareas.rename(columns={'Name':'CommonName','StopAreaType':'StopType'})
        
        ferrypiers = pd.read_csv(str(naptan_piers), encoding='windows-1252')
        ferrypiers = ferrypiers.set_index('AtcoCode')[['Name','Easting','Northing']]
        ferrypiers = ferrypiers.rename(columns={'Name':'CommonName'})
        
        stops = pd.concat([stops, stopareas, ferrypiers])
        
        # Hard fixes
        stops.loc['9400ZZLUWIG3'] = stops.loc['9400ZZLUWIG1']

        #freq = freq.iloc[:,0:18]
        
        freq = freq.join(stops[['CommonName','Easting','Northing']], on='location')
        freq = freq.join(stops[['CommonName','Easting','Northing']], on='call_next', rsuffix='_to')
        
        freq = freq.rename(columns={'CommonName':'label_from',
                                    'CommonName_to':'label_to',
                                    'Easting':'easting_from',
                                    'Northing':'northing_from',
                                    'Easting_to':'easting_to',
                                    'Northing_to':'northing_to'
                                    })
        
        #freq = freq.drop(['label_from','label_to','easting_from','easting_to','northing_from','northing_to','linkgeom'], axis=1)
        freq = freq.reset_index(drop=True)      
            
            
        freq['linkgeom'] = pd.Series(['LINESTRING(']*len(freq)).str.cat([
                                     freq['easting_from'].fillna(0).astype(int).astype(str),
                                     pd.Series([' ']*len(freq)),
                                     freq['northing_from'].fillna(0).astype(int).astype(str), 
                                     pd.Series([', ']*len(freq)),
                                     freq['easting_to'].fillna(0).astype(int).astype(str),
                                     pd.Series([' ']*len(freq)),
                                     freq['northing_to'].fillna(0).astype(int).astype(str), 
                                     pd.Series([')']*len(freq))
                                     ], na_rep='0')
        
        freq.loc[freq[['easting_from','easting_to']].isnull().any(axis=1), 'linkgeom'] = 'LINESTRING(540000 180000, 540000 180000)'
        
        tmr.stop()
        """
        
        return freq

# %%