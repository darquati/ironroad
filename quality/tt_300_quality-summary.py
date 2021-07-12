#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 17:54:04 2019

@author: DavidArquati
"""

import pandas as pd
from pathlib import Path
import time

tfl_path = Path('/Users/DavidArquati/Documents/GIS/Timetable analysis/TfL API/outputs/20181122')
nr_path = Path('/Users/DavidArquati/Documents/GIS/Timetable analysis/National Rail/outputs/20181117')
working_folder = Path(r'/Users/DavidArquati/Documents/GIS/Timetable analysis/Unified Quality/outputs')

importdate = '20181122'

tt_files = {'nr': nr_path / Path('NRTimetable_20181117_LinkFormat.csv'), 
            'lu': tfl_path / Path('TfL_LU_Timetable_20181122.csv'), 
            'dlr': tfl_path / Path('TfL_DLR_Timetable_20181122.csv'), 
            'trams': tfl_path / Path('TfL_Tram_Timetable_20181122.csv')}
            #'buses': tfl_path / Path('TfL_Bus_Timetable_20181122.csv'), 

outputfile = working_folder / Path(r'{}/RailFrequencies_{}.csv'.format(importdate,importdate)) 

ref_folder = Path(r'/Users/DavidArquati/Documents/GIS/Station database')
#ref_folder = Path(r'\\onelondon.tfl.local\shared\City Planning\07 Transport Strategy\14 Public Transport\10 Data\Reference\Stations & Stops')
stnlist_file = ref_folder / Path(r'PTSPAllStationList_NumbatV16a_All.csv')
naptan_to_numbat_file = ref_folder / Path(r'naptan-numbat-v4.csv')

naptan_stops = ref_folder / Path(r'naptan/Stops.csv')
naptan_stop_areas = ref_folder / Path(r'naptan/StopAreas.csv')
naptan_piers = ref_folder / Path(r'naptan/FerryReferences.csv')

class Timer():
    def __init__(self, label=None):
        self.a = time.time()
        if label is not None:
            print(label)
    def stop(self):
        self.b = time.time()
        self.duration = self.b - self.a
        print("{:.2f} seconds elapsed".format(self.duration))
        return self.duration

# Reorder columns for clarity
def reorder_list(listobj, firstitems, lastitems=[], dropitems=[]):
    firstitems = [x for x in firstitems if x in listobj and x not in dropitems]
    middleitems = [x for x in listobj if x not in firstitems and x not in lastitems and x not in dropitems]
    lastitems = [x for x in lastitems if x not in dropitems]
    newlist = firstitems + middleitems + lastitems                  
    return newlist

#%%
data = pd.concat([pd.read_csv(str(p)) for m, p in tt_files.items()])

data['UTime'] = data['UTime'].fillna(data['FromDep'])
data['DaysOperating'] = data['DaysOperating'].astype(str).str.zfill(7)
data['ServiceGroupID'] = data['ServiceGroupID'].fillna(0).astype(int).astype(str)

# Merge ScheduleID_CIF and RunID (RunID should be ScheduleID_TfL)
data['ScheduleID'] = data['ScheduleID'].fillna(data['ScheduleID_CIF'])

data = data.set_index(['ScheduleID','CallSeq','EventSeq'])

# Add call sequence for TfL (done in calls-only section)
callflag = (data.FlagsFrom.str.match('.[CudR].') | (data.EventType=='LT'))
data['CallNext'] = data.loc[data.FlagsTo.str.match('.[CudR].'), 'LocationTo']
data['CallNext'] = data['CallNext'].bfill()

data['RunTime'] = data['ToArr'] - data['FromDep']

data['EventType'] = data['EventType'].fillna(data['Event'])

newcols = reorder_list(data.columns, ['Mode','Line','Operator','DaysOperating',
                                    'ServiceGroupID','DirectionRun','ScheduleID','VehicleType','EventType',
                                    'LocationFrom','LocationTo','DirectionLink',
                                    'FlagsFrom','FlagsTo',
                                    'FromDep','ToArr','UTime','DwellTimeFrom','RunTime'],
                                    dropitems=['CallPrev','LocationPrev','ScheduleID_Headcode','ScheduleID_CIF','RunID','DwellTimeTo','Event'])
data = data.loc[:, newcols]


#%%
# Collapse data to calls only & at least one-end in 'London'
data_calls = data[callflag]

data_calls['LocationTo'] = data_calls['CallNext']
data_calls['FlagsTo'] = data_calls.groupby(level='ScheduleID')['FlagsFrom'].shift(-1).fillna(data.FlagsTo)

london_eitherend = (data_calls.FlagsFrom.str.contains('L') | data_calls.FlagsTo.str.contains('L'))
london_bothends = (data_calls.FlagsFrom.str.contains('L') & data_calls.FlagsTo.str.contains('L'))

data_calls = data_calls.reset_index('EventSeq').set_index('CallSeq', append=True)

data_calls['LinkDistance'] = data.groupby(level=['ScheduleID','CallSeq'])['LinkDistance'].sum()


newcols = reorder_list(data_calls.columns, ['Mode','Line','Operator','DaysOperating',
                                    'ServiceGroupID','DirectionRun','EventType',
                                    'LocationFrom','LocationTo','DirectionLink',
                                    'FlagsFrom','FlagsTo',
                                    'FromDep','ToArr','UTime'],
                                    dropitems=['DwellTimeFrom','RunTime','VehicleType','CallNext'])

data_calls = data_calls.loc[london_eitherend, newcols]

#%%
naptan_to_numbat = pd.read_csv(str(naptan_to_numbat_file)).set_index('location').drop('mode',axis=1)

data_calls = data_calls.join(naptan_to_numbat[['NumbatTLC']], on='LocationFrom')
data_calls.rename(columns={'NumbatTLC':'LocationFromGroup'}, inplace=True)
data_calls = data_calls.join(naptan_to_numbat[['NumbatTLC']], on='LocationTo')
data_calls.rename(columns={'NumbatTLC':'LocationToGroup'}, inplace=True)

data_calls['LocationFromGroup'] = data_calls['LocationFromGroup'].fillna(data_calls['LocationFrom'])
data_calls['LocationToGroup'] = data_calls['LocationToGroup'].fillna(data_calls['LocationTo'])

#data_calls = data_calls.reset_index(drop=True)

#%% Get frequencies by time band

tmr = Timer("Categorising departures by time band...")

data_calls['TimeBand'] = pd.cut(data['FromDep'],
           [7*3600, 
            10*3600, 
            16*3600, 
            19*3600, 
            22*3600], 
            labels=['AM','IP','PM','EV'])

tmr.stop()

#%%
tmr = Timer("Getting frequencies for each day & time band...")
# 430 seconds

d = {}
dmatch =  {'We':'..1....','Fr':'....1..','Sa':'.....1.','Su':'......1'} # 'Mo':'1......','Tu':'.1.....','Th':'...1...',

def effintervalfunc(x):
    return ((x**2).sum() / x.sum())

data_calls = data_calls.sort_values('FromDep')

data_calls['PrevHeadway'] = None

# Create a new dataframe with a separate entry for each day, and just the relevant info for analysis
for key, value in dmatch.items():
    print("  Getting {}".format(key))
    d[key] = data_calls.loc[data_calls.DaysOperating.str.match(value), ['Mode','Line','LocationFromGroup','LocationToGroup','TimeBand','FromDep']]
    d[key]['Day'] = key
    
departures = pd.concat(d).reset_index()
departures['PrevHeadway'] = departures.groupby(['Day','Mode','Line','LocationFromGroup','LocationToGroup'], sort=False)['FromDep'].transform(pd.Series.diff)

freq = departures.groupby(['Mode','Line','LocationFromGroup','LocationToGroup','Day','TimeBand'], sort=False).agg({'FromDep': ['size','min','max'],'PrevHeadway':['sum','mean','median',effintervalfunc]})
freq = freq.reset_index()

tmr.stop()

tmr = Timer("Calculating ancillary statistics...")

freq.columns = [' '.join(col).strip() for col in freq.columns.values]
freq = freq.rename(columns = {'FromDep size':'count',
                              'FromDep min':'first',
                              'FromDep max':'last',
                              'PrevHeadway sum':'service_duration',
                              'PrevHeadway mean':'interval_servicemean',
                              'PrevHeadway median':'interval_median',
                              'PrevHeadway effintervalfunc':'interval_effective'})

freq['period_duration'] = freq['TimeBand'].map({'AM':3*3600,'IP':6*3600,'PM':3*3600,'EV':3*3600})
#freq['service_duration'] = freq['last'] - freq['first']

freq['interval_periodmean'] = freq['period_duration'] / freq['count']
#freq['interval_servicemean'] = freq['service_duration'] / (freq['count'] - 1)

freq['availability'] = (freq[['service_duration','period_duration']].min(axis=1) / freq['period_duration'])

freq['tph_periodmean'] = 3600 / freq['interval_periodmean'] 
freq['tph_servicemean'] = 3600 / freq['interval_servicemean'] 
freq['tph_median'] = 3600 / freq['interval_median']
freq['tph_effective'] = 3600 / freq['interval_effective']

freq['tt_evenness'] = freq['tph_effective'] / freq['tph_servicemean']

freq = freq.set_index(['Mode','LocationFromGroup','LocationToGroup','Day','TimeBand'])

tmr.stop()

#%%
# Geometry
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
### Up to here

freq = freq.reset_index()
freq = freq.join(stops[['CommonName','Easting','Northing']], on='LocationFromGroup')
freq = freq.join(stops[['CommonName','Easting','Northing']], on='LocationToGroup', rsuffix='_to')

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

t = Timer("Writing outputs to disk...")

#freqoutfile_weekday = workingfolder / Path(r'TfLFrequencies_{}_MF.csv') 
#freqoutfile_sat = workingfolder / Path(r'TfLFrequencies_{}_Sa.csv') 
#freqoutfile_sun = workingfolder / Path(r'TfLFrequencies_{}_Su.csv') 
    
freq.to_csv(str(outputfile), sep=';')

#freq[freq.day == 'We'].to_csv(str(freqoutfile_weekday), sep=';')
#freq[freq.day == 'Sa'].to_csv(str(freqoutfile_sat), sep=';')
#freq[freq.day == 'Su'].to_csv(str(freqoutfile_sun), sep=';')
   
t.stop()