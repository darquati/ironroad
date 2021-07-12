#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IronRoad TfL Timetable Loader

Retrieves a local or remote TfL timetable ZIP and converts it into an SQLite Database

@author: DavidArquati


"""
#%%
import datetime
import logging
import pandas as pd
from pathlib import Path
import sqlite3
import zipfile

from . import transxchange as txtt
from ..tools import create_onelondon_session
from .. import cfg

#%%
group_mapper = {'1': 'u', '25': 'd', '30':'f','31':'f','32':'f','33':'f', '35': 'f', '63': 't', '71': 'c', '*':'b'}

logger = logging.getLogger(__name__)

def download(url=cfg.cfg['remote_resources']['tfl_api_tt_url']):

    databundle = datetime.date.today().strftime("%Y%m%d")
    destination_file = Path(cfg.cfg['paths']['tfl_downloaded_file'].format(databundle=databundle))
    
    if not destination_file.parent.exists():
        destination_file.parent.mkdir(parents=True)
    
    logger.info(f"Downloading from {url}")
    session = create_onelondon_session()
    r = session.get(url)
    
    with open(destination_file, "wb") as zip:
        zip.write(r.content)
    
    logger.info(f"Downloaded result to folder {destination_file.parent}")
    
    return destination_file, databundle

def transxchange_to_db(input_dir, output_db, group_mapper=group_mapper):
    """
        This script is used for extracting the full set of features from the
        TransXChange timetable data.

        This function calls a number of functions defined in a custom
        TfLTimetable Class based on the lxml ElementTree.
        For that Class definition see TfLTimetable.py.

        This source file is invoked with two arguments:

            input:    The folder location of the XML files for parsing. eg ../1_data/1_1_raw_data/timetables/data

            output:   The file location for an SQLite database to store the outputs
            
        Code is based on source from https://github.com/ruaridhw/london-tube/blob/master/2_analysis/python/XMLParsing.py

    #' The Operating Period and Profile information is scraped separately
    #' from the `TfLTimetable.get_df` function due to the extra level of
    #' sophistication required.
    #' These nodes contain a varying number of child nodes where the data is
    #' contained in the tag name (rather than the tag text) and the tag name
    #' is not known in advance.

    #' Due to the fact that the **StopPoints** and **NptgLocalities**
    #' tables are the only two that contain duplicate information *across*
    #' tube lines, these are held in memory throughout the iteration in
    #' `output_tables_common` without being written to disk after each line
    #' and dumped to a single file with the prefix "ALL" in place of the line code.
    """

    # Pattern match timetable files
    xml_files = pd.DataFrame.from_dict({f.stem: f for f in input_dir.glob('*.xml')}, orient='index', columns=['Path']) #pd.Series({f.stem: f for f in input_dir.glob('*.xml')})   
    xml_files.index = pd.MultiIndex.from_frame(xml_files.index.str.extract("tfl_([0-9]{1,2})-([0-9A-Z]{1,4})-[^.]+").rename(columns={0:'Group',1:'Line'}))
    xml_files['Mode'] = xml_files.index.get_level_values('Group').map(group_mapper).fillna('b')
    xml_files = xml_files.set_index('Mode', append=True)

    # Create SQLite database to receive outputs
    if output_db.exists():
        output_db.unlink() # Overwrite any existing database so we can append incrementally
    con = sqlite3.connect(output_db)
    
    # Iterate over files to extract contents into tables
    i=0
    for (group, line, mode), path in xml_files['Path'].iteritems():
        i+=1
        n = path.stem
        logger.info(f"File {i} of {len(xml_files)}: {n}...")

        timetable = txtt.TransXTimetable(str(path))

        for table, paths in timetable.required_xpaths.items():
            timetable.get_df(paths)\
                    .assign(Mode=mode, Line=line)\
                    .drop_duplicates()\
                    .to_sql(table, con, if_exists='append')

        # Parse Operating Periods for VehicleJourneys and Services
        for op_prof_path in timetable.op_prof_paths:
            op_prof_tables = timetable.get_varying_child_tags(op_prof_path)

            for table, data in op_prof_tables.items():
                data.assign(Mode=mode, Line=line)\
                    .drop_duplicates()\
                    .to_sql(table, con, if_exists='append')

    con.close()
    
    logger.info("Database written.")
    return
        
def extract_to_db(databundle):

    # Set up input and output folders
    databundle_folder = Path(cfg.cfg['paths']['tfl_input_folder'].format(databundle=databundle))
    if not databundle_folder.exists():
        raise ValueError(f"Databundle folder does not exist at {databundle_folder}")

    input_file = Path(cfg.cfg['paths']['tfl_downloaded_file'].format(databundle=databundle))
    output_db = Path(cfg.cfg['paths']['tfl_extracted_file'].format(databundle=databundle))

    if output_db.exists():
        print("Looks like timetable has already been extracted and transformed into dataframes")
        #return output_db
    elif not output_db.parent.exists():
       output_db.parent.mkdir()
 
    try:
        input_file = list(input_file.parent.glob('*.zip'))[0]
    except:
        raise ValueError(f"Could not find a downloaded Journey Planner zip file in {input_file.parent}")

    xml_dir = input_file.parent/'xml'
    xml_dir.mkdir()
        
    # Extract the downloaded file
    with zipfile.ZipFile(input_file, 'r') as download_zip:
        for z in download_zip.infolist():
            download_zip.extract(z, xml_dir)
            
    nested_zip = True
    while nested_zip:
        start_files = list(xml_dir.glob('*'))
        any_zip_files = any(map(zipfile.is_zipfile, start_files))
        
        if any_zip_files:
            for i in start_files:
                if zipfile.is_zipfile(i):
                    print(f"Extracting {i.stem}")
                    with zipfile.ZipFile(i) as zz:
                        zz.extractall(xml_dir)
                    i.unlink()
        else:
            nested_zip = False

    # Convert the extracted files to dataframes in feather format
    transxchange_to_db(xml_dir, output_db)
    
    # Clean up the extracted files
    for x in xml_dir.glob('*'):
        x.unlink()
    xml_dir.rmdir()
    
    return output_db

def get_timetable(ttdate=None,
                  source='remote',
                  extract=None
                  ):
    """Obtains the TfL timetable in gzipped format and returns 
    the location, extracting it into an SQLite DB if requested.
    
    It can choose either a local file, 
    the latest file from the TfL API (no auth required).
    
    Parameters
    ----------  
    ttdate : str or None, optional
        Date (yyyymmdd) to search for, or None for the latest file online 
        (default is None)
        
    source:     str
                    local : already-downloaded local ZIP or extracted DB
                    latest: current file from TfL (ignores date)
         
    extract : bool or None (default)
        Whether to transform a local downloaded timetable file into a database or not
        None will decide based on the source (extract=True if source='remote')

    Returns
    -------
    Path object
        The local path to the resulting file as specified (local zip or database)

    """        

    source = source.lower()
    locate_success = False

    if source not in ['local','remote']:
        raise ValueError("Expected one of 'local' or 'remote' as the source")

    if source == 'local':
        if extract is None: extract = False
        downloaded_file = Path(cfg.cfg['paths']['tfl_downloaded_file'].format(databundle=ttdate))

        if not downloaded_file.exists():
            logger.warning(f"Could not find local file matching date {ttdate} at {downloaded_file}")
        else:
            locate_success = True
            databundle = ttdate
            logger.info(f"Local file located at {downloaded_file}")
            
    if source == 'remote':
        if extract is None: extract = True

        try:
            downloaded_file, databundle = download()
            locate_success = True
            logger.info(f"Remote file downloaded to {downloaded_file}")
        except Exception as err:
            raise ValueError(f"Could not retrieve & offer TfL timetable file: {err}")
    
    if locate_success and extract:
        extracted_file = extract_to_db(databundle)
        out_file = extracted_file
    elif locate_success and not extract:
        out_file = downloaded_file
    else:
        raise ValueError("Could not locate the TfL timetable file")

    return out_file, databundle

