# -*- coding: utf-8 -*-
"""
Extract a station-by-station Effective Headway from TfL & NR data as 
one unified figure for the week.

"""

import pandas as pd
from pathlib import Path

#working_folder = Path('\\onelondon.tfl.local\shared\City Planning\07 Transport Strategy\14 Public Transport\10 Data\GIS\03 Projects\Timetable quality')
working_folder = Path('/Users/DavidArquati/Documents/Work/Timetable Quality')

input_file = working_folder / Path('outputs/Unified_WeekLowestEffectiveInterval.csv')

ref_folder = Path(r'/Users/DavidArquati/Documents/Work/Station database')
#ref_folder = Path(r'\\onelondon.tfl.local\shared\City Planning\07 Transport Strategy\14 Public Transport\10 Data\Reference\Stations & Stops')
stnlist_file = ref_folder / Path(r'PTSPAllStationList_NumbatV15_All.csv')
naptan_to_numbat_file = ref_folder / Path(r'naptan-numbat-v3.csv')

output_folder = working_folder
output_file = output_folder / Path('outputs/Unified_WeekLowestEffectiveInterval_ByStation.csv')

#%%
data = pd.read_csv(str(input_file), index_col=0)
data = data[data['mode'].isin(['NR','LU','DLR','Tram'])]

# Get the worst interval for each time period for each link
link_max_effint = data.groupby(['mode','location','call_next'])['interval_effective'].max()

#%%
# Out of that, get the best interval for each departure station
stn_max_effint = link_max_effint.groupby(level=['mode','location']).min()
stn_max_effint = stn_max_effint.reset_index()

#naptan_stopstoareas = pd.read_csv(str(naptan_stopstoareas_file))[['StopAreaCode','AtcoCode']].set_index('AtcoCode')
#naptan_railreferences = pd.read_csv(str(naptan_railreferences_file),index_col=0).drop_duplicates()[['AtcoCode','TiplocCode']].set_index('TiplocCode')

naptan_to_numbat = pd.read_csv(str(naptan_to_numbat_file)).set_index('location').drop('mode',axis=1)

stn_max_effint = stn_max_effint.join(naptan_to_numbat, on='location')

if stn_max_effint.NumbatTLC.isnull().sum() > 0:
    print("Could not match some locations:")
    print(stn_max_effint[stn_max_effint.NumbatTLC.isnull()])

stn_max_effint.AtcoCode = stn_max_effint.AtcoCode.fillna('location')

#%%
stnlist = pd.read_csv(str(stnlist_file)).set_index('FinalTLC')
hub_coords = stnlist.groupby('Hub')[['Easting','Northing']].mean()

stn_max_effint = stn_max_effint.join(stnlist[['UniqueStationName','Hub']], on='NumbatTLC')

stn_max_effint['FlagTfL'] = ~stnlist.loc[stn_max_effint.NumbatTLC, ['Flag_LO','Flag_EL']].isnull().all(axis=1).values
stn_max_effint.loc[stn_max_effint['mode'].isin(['Bus','DLR','Tram','LU']), 'FlagTfL'] = True

stn_max_effint_summary = stn_max_effint.set_index(['Hub','NumbatTLC','mode','FlagTfL']).groupby(level=[0,2,3])['interval_effective'].min().unstack(level=[2,1])

stn_max_effint_summary = stn_max_effint_summary.join(stnlist['UniqueStationName'])
stn_max_effint_summary = stn_max_effint_summary.join(hub_coords)
    
stn_max_effint_summary = stn_max_effint_summary.rename(columns={(False, 'NR'): 'NR_noTfL',
                                                                (True, 'DLR'): 'DLR',
                                                                (True, 'NR'): 'NR_withTfL',
                                                                (True, 'LU'): 'LU',
                                                                (True, 'Tram'): 'Tram'})

stn_max_effint_summary['BestTfL'] = stn_max_effint_summary[['LU','DLR','NR_withTfL','Tram']].min(axis=1)
stn_max_effint_summary['BestNR'] = stn_max_effint_summary[['NR_withTfL','NR_noTfL']].min(axis=1)
stn_max_effint_summary['Best'] = stn_max_effint_summary[['LU','DLR','NR_withTfL','Tram','NR_noTfL']].min(axis=1)
    
    
stn_max_effint_summary.to_csv(str(output_file))
