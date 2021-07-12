#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 17:42:13 2018

@author: DavidArquati
"""

import numpy as np
from pathlib import Path
import pandas as pd
import time

importdate = '20181117'

#working_folder = '//onelondon.tfl.local/shared/London Rail Projects/12 Library/33 NR Open Data/Analysis Tools/NRTT Loader'
working_folder = Path('/Users/DavidArquati/Documents/GIS/Timetable analysis/National Rail')

numbat_lookup_file = working_folder / Path('reference/network/numbat-stations.xlsx')
node_lookup_file = working_folder / Path('reference/network/model-nodes.xlsx')

input_file = working_folder / Path('outputs/{d}/NRTimetable_{d}.csv'.format(d=importdate)) # 2018-04-06, 2017-07-19, 2016-11-04
#timetable_intermediate_summary_file = working_folder / Path('inputs/{d}/NRTimetable_{d}_summary.csv'.format(d=importdate))

output_file = working_folder / 'outputs/{d}/NRFrequencies_{d}.csv'.format(d=importdate)


pd.set_option('display.float_format', lambda x:'%f'%x) # Workaround for bug in Spyder

class Timer(object):
    def __init__(self, tag=None):
        if tag is not None:
            print(tag, flush=True)
        else:
            print("Timing...", flush=True)
        self.start = time.time()

    def stop(self):

        self.end = time.time()
        self.duration = self.end - self.start
        print("Completed in {:.1f} seconds.".format(self.duration), flush=True)

# Reorder columns for clarity
def reorder_list(listobj, firstitems, lastitems=[]):
    firstitems = [x for x in firstitems if x in listobj]
    notfirstitems = [x for x in listobj if x not in firstitems]
    middleitems = [x for x in notfirstitems if x not in lastitems]
    newlist = firstitems + middleitems + lastitems
    return newlist


#%%
t = Timer("Loading data...")

#tt_summary = pd.read_csv(str(timetable_intermediate_summary_file), index_col=['uid','subid'])
tt = pd.read_csv(str(input_file), index_col=['uid','subid','eventid'])

# Fix time representations after loading CSV
#timecols = ['keytime','origin_dep','destin_arr','runtime','halflife','linktime_mean','linktime_median','linktime_mean_london','runtime_london']
#tt_summary[timecols] = tt_summary[timecols].replace(np.inf,np.nan).apply(pd.to_timedelta)

timecols = ['utime','arr','dep','pass','dwell','runtimetonext','key_time','key_event_time','key_time_calls']
tt[timecols] = tt[timecols].apply(pd.to_timedelta)
tt['utime'] = tt['utime'] / pd.Timedelta('1 second')

# Fix non-strings
nonstr = (tt.platform.apply(type) == float) & ~tt.platform.isnull()
tt.loc[nonstr, 'platform'] = tt.loc[nonstr, 'platform'].astype(int)
tt['platform'] = tt['platform'].astype(str)
tt.loc[tt.platform == 'nan', 'platform'] = None

# Fix bools
tt['londoncall'] = tt['londonflag'] & tt['callflag']
tt[['londonflag','callflag']] = tt[['londonflag','callflag']].astype('bool')

tt['days'] = tt['days'].astype('str').str.zfill(7)

t.stop()

node_lookup_file_str = Path(working_folder).joinpath(node_lookup_file).as_posix()
l_tiplocs = pd.read_excel(node_lookup_file_str, sheetname='all-tiplocs', index_col='TIPLOC') # Used to convert TIPLOCs to NLCs (note: not a 1-to-1 relationship)
l_tiplocs = l_tiplocs[~l_tiplocs.index.duplicated()]
l_tiplocs['londonflag'] = l_tiplocs['londonflag'].fillna(0).astype(bool)
l_tiplocs_l = l_tiplocs[['NLC6','londonflag','nrstationflag',
                         'easting','northing',
                         'servicegroup','srs_code',
                         'score']]

#%%

from shapely.geometry import LineString

tt['link'] = tt.groupby(level=['uid','subid'])['callflag'].cumsum().astype(int)
tt = tt.set_index('link', append=True)

tt = tt.join(l_tiplocs_l[['easting','northing']],on='call_next',rsuffix='j')

def geomfunc(link):
    linkview = link[(link.record != 'LT') & ~link.easting.isnull()]
    if len(linkview) > 0:
        pointset = linkview[['easting','northing']].values.tolist()
        pointset.append(linkview[['eastingj','northingj']].values[-1].tolist())
        return LineString(pointset).wkt

#tt.loc['C25400'].groupby(['subid','link']).apply(geomfunc)

res = tt[tt.londonflag].groupby(level=['uid','subid','link']).apply(geomfunc)
res.name = 'linkgeom'

tt = tt.reset_index('eventid').join(res).set_index('eventid', append=True)
tt.loc[~tt.callflag, 'linkgeom'] = np.nan

latetrains = tt.xs(1, level='eventid').query('utime < 4*60*60').index.droplevel(2).to_frame()

tt = tt.reset_index(['link','eventid'])
tt.loc[latetrains.index, ['arr','dep','pass']] += '1 day'
tt.loc[latetrains.index, ['utime']] += 24*60*60
tt = tt.set_index(['link','eventid'], append=True)

tt['timeperiod'] = None
tt.loc[tt.utime.between(7*3600,10*3600), 'timeperiod'] = 'AM'
tt.loc[tt.utime.between(10*3600,16*3600), 'timeperiod'] = 'IP'
tt.loc[tt.utime.between(16*3600,19*3600), 'timeperiod'] = 'PM'
tt.loc[tt.utime.between(19*3600,22*3600), 'timeperiod'] = 'EV'

linkeventcount = tt[tt.londonflag].groupby(level=['uid','subid','link'])['stationflag'].sum()
linkeventcount.name = 'linkeventcount'
tt = tt.reset_index('eventid').join(linkeventcount).set_index('eventid', append=True)


#%%

d = {}
dmatch =  {'We':'..1....','Fr':'....1..','Sa':'.....1.','Su':'......1'}
modefunc = lambda x: x.mode().iloc[0]

def intervalfunc(x):
    return x.diff().median()
def effintervalfunc(x):
    return (x.diff()**2).sum() / x.diff().sum()

aggs = {'utime':['size','min','max',intervalfunc,effintervalfunc],
        'linkgeom':modefunc,
        'linkeventcount': 'mean'}

tt = tt.sort_values('utime')
for key, value in dmatch.items():
    d[key] = tt[tt.londoncall & tt.days.str.match(value)].groupby(['timeperiod','location','call_next']).agg(aggs)

freq = pd.concat(d).reset_index()

freq.columns = ['day','TimeBand','LocationFrom','LocationTo',
                'count','first','last',
                'interval_median','interval_effective','linkgeom',
                'skippedstns']

freq['skippedstns'] -= 1

freq.loc[freq['first'] == freq['last'], 'count'] = 1

freq['service_duration'] = (freq['last'] - freq['first'] + freq['interval_median'])
freq['period_duration'] = freq['TimeBand'].map({'AM':3*3600,'IP':6*3600,'PM':3*3600,'EV':3*3600})

freq['availability'] = freq['service_duration'] / freq['period_duration']

freq['tph_periodmean'] = freq['count'] / freq['period_duration'] * 3600
freq['tph_servicehours_mean'] = (freq['count'] - 1) / freq['service_duration'] * 3600
freq['tph_servicehours_mean'] = freq['tph_servicehours_mean'].fillna(freq['tph_periodmean'])

freq['tph_median'] = 3600 / freq['interval_median']
freq['tph_effective'] = 3600 / freq['interval_effective']
freq['tph_effective'] = freq['tph_effective'].fillna(freq['tph_periodmean'])
freq['tt_evenness'] = freq['tph_effective'] / freq['tph_servicehours_mean']

freq.to_csv(str(output_file), sep=';')

