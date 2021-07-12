# -*- coding: utf-8 -*-
"""
Unify the TfL and NR effective headway datasets and output as 
Bus and UnifiedRail datasets

"""

import pandas as pd
from pathlib import Path

#working_folder = Path('\\onelondon.tfl.local\shared\City Planning\07 Transport Strategy\14 Public Transport\10 Data\GIS\03 Projects\Timetable quality')
working_folder = Path('/Users/DavidArquati/Documents/Work/Timetable Analysis')
output_folder = working_folder


# Read the data in
nr_data = pd.read_csv(str(working_folder / Path('National Rail/outputs/20180929/NRFrequencies_20180929.csv')),
                   sep=';', index_col=0)
tfl_data = pd.read_csv(str(working_folder / Path('TfL API/outputs/20181122/TfLFrequencies_20181122.csv')),
                   sep=';', index_col=0)

# Align the two datasets by columns
nr_data['mode'] = 'NR'
nr_data = nr_data.rename(columns={'linkgeom_cmn':'linkgeom','mode':'Mode'})

tfl_data['skippedstns'] = 0
tfl_data = tfl_data.drop(['easting_from','northing_from','easting_to','northing_to'], axis=1)

all_data = pd.concat([tfl_data,nr_data])

rail_data = all_data[all_data['Mode'].isin(['LU', 'DLR', 'Tram' ,'NR'])]
bus_data = all_data[all_data['Mode'].isin(['Bus', 'Bus_OutsideLondon'])]

bus_data.to_csv(str(output_folder / Path('BusFrequencies_2018_AllWeek.csv')))
rail_data.to_csv(str(output_folder / Path('UnifiedRailFrequencies_2018_AllWeek.csv')))

#%%

# Calculate weekly stats using Mo-Sa 0700-2200 and Su 1000-2200
filt = (all_data['day'].isin(['We','Sa']) & all_data['TimeBand'].isin(['AM','IP','PM','EV'])) | ((all_data.day == 'Su') & all_data['TimeBand'].isin(['IP','PM','EV']))

all_link_max_effint = all_data[filt].groupby(['Mode','LocationFrom','LocationTo','linkgeom'])['interval_effective'].max()
all_link_max_effint = all_link_max_effint.reset_index()

all_link_max_effint.to_csv(str(output_folder / Path('Unified_WeekLowestEffectiveInterval.csv')))




