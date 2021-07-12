#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 23:29:49 2018

@author: DavidArquati
"""

#%%
# SETUP
# System imports
import pandas as pd
from pathlib import Path
import time

# Working folders
#working_folder = Path(r'//onelondon.tfl.local/shared/London Rail Projects/12 Library/04 Timetable analysis/TfL API/')
working_folder = Path(r'/Users/DavidArquati/Documents/GIS/Timetable analysis/TfL API/')

importdate = '20181122'

inputfile = working_folder / Path('outputs/{}/TfLTimetable_{}.feather'.format(importdate, importdate))
outputfile = working_folder / Path(r'outputs/{}/TfLFrequencies_{}.csv'.format(importdate,importdate)) 

naptan_stops = working_folder / Path(r'reference/naptan/Stops.csv')
naptan_stop_areas = working_folder / Path(r'reference/naptan/StopAreas.csv')
naptan_piers = working_folder / Path(r'reference/naptan/FerryReferences.csv')

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

#%% Load data
        
tmr = Timer("Loading data from disk...")
tflnet_full = pd.read_feather(str(inputfile))
#tflnet_full = pd.read_csv(str(inputfile))

tflnet_full['SeqFrom'] = tflnet_full['SeqFrom'].astype(int)
tflnet_full['SeqTo'] = tflnet_full['SeqTo'].astype(int)

tmr.stop()

#%% Get frequencies by time band
# This takes about 20 minutes on a Macbook

tmr = Timer("Categorising departures by time band...")

tflnet_full['TimeBand'] = pd.cut(tflnet_full['FromDep'],
           [7*3600, 
            10*3600, 
            16*3600, 
            19*3600, 
            22*3600], 
            labels=['AM','IP','PM','EV'])

tmr.stop()

d = {}
dmatch =  {'We':'..1....','Fr':'....1..','Sa':'.....1.','Su':'......1'} # 'Mo':'1......','Tu':'.1.....','Th':'...1...',

def intervalfunc(x):
    return x.diff().median()
def effintervalfunc(x):
    return ((x.diff()**2).sum() / x.diff().sum())


tmr = Timer("Getting frequencies for each day & time band...")
# 430 seconds

tmr2 = Timer("  Sorting by departure time...")
# Original 'index': Mode Line Pattern Run DCode SeqFrom
tflnet_full = tflnet_full.sort_values('FromDep')
tmr2.stop()

for key, value in dmatch.items():
    print("  Getting {}".format(key))
    d[key] = tflnet_full[tflnet_full.DaysF.str.match(value)].groupby(['Mode','LocationFrom','LocationTo','TimeBand'])['FromDep'].agg(['size','min','max',intervalfunc,effintervalfunc])

print("Concatenating results...")
freq = pd.concat(d).reset_index()

tmr.stop()

tmr = Timer("Calculating ancillary statistics...")

freq = freq.rename(columns = {'level_0':'day',
                              'size':'count',
                              'min':'first',
                              'max':'last',
                              'intervalfunc':'interval_median',
                              'effintervalfunc':'interval_effective'})

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

freq = freq.join(stops[['CommonName','Easting','Northing']], on='LocationFrom')
freq = freq.join(stops[['CommonName','Easting','Northing']], on='LocationTo', rsuffix='_to')

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