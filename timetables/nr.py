# -*- coding: utf-8 -*-
"""
IronRoad NRTT downloader

This module retrieves an NR CIF timetable file from a local or remote source, and can convert it to a SQLite Database

Currently using mirrors at
https://networkrail.opendata.opentraintimes.com/mirror/schedule/cif/

Otherwise
https://datafeeds.networkrail.co.uk/ntrod/CifFileAuthenticate?type=CIF_ALL_FULL_DAILY&day=toc-full.CIF.gz

"""

import getpass
import gzip
import logging
import pickle
import re
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from requests.auth import HTTPBasicAuth

from ..tools import create_onelondon_session
from .. import cfg

#%%
logger = logging.getLogger(__name__)

def get_databundle_from_filename(filename):
    return re.findall(r'\d{8}', filename)[0]

def get_local_file(ttdate=None, extract=False):
    if ttdate is not None:
        searchdate = ttdate+'*'
    else:
        searchdate = '*'

    if extract:
        file_location = Path(cfg.cfg['paths']['nr_downloaded_file'].format(date=searchdate))
        processstr='downloaded'
    else:
        file_location = Path(cfg.cfg['paths']['nr_extracted_file'].format(databundle=searchdate))
        processstr='extracted'

    logging.info(f"Looking for {processstr} local file for date {ttdate} in {file_location}")
        
    input_cif = sorted([x for x in file_location.parent.glob(file_location.name)]) # gets first matching file

    if len(input_cif) > 0:
        file_location = input_cif[-1]
        databundle = get_databundle_from_filename(file_location.name)
        logger.info(f"Found {processstr} local file for {databundle} at {file_location}")
        if extract:
            extract_cif(file_location)
    else:
        logger.warning(f"Did not find {processstr} local file at {file_location}")
        file_location, databundle = None, None
    
    return file_location, databundle

def get_file_name(response):
    if 'content-disposition' in response.headers.keys():
        name = response.headers['content-disposition']
    else:
        name = response.url.split(r'/')[-1]
    return name

def download(url=cfg.cfg['remote_resources']['nr_api_tt_url'], 
                    session=None,
                    username=None, password=None):
    logging.info(f"Looking for remote file at {url}")
    
    if session is None:
        session = create_onelondon_session()
    
    if username is not None and password is not None:
        auth = HTTPBasicAuth(username, password)
        databundle = date.today().strftime('%Y%m%d')
    else:
        auth = None
        databundle = None
    
    r = session.get(url, auth=auth, stream=True)
        
    if 'text' in r.headers['Content-Type']:
        raise ValueError("Did not receive a file in response:\n{}".format(r.text))
    
    if databundle is None:
        filename = get_file_name(r)
        if '.gz' not in filename.lower():
            filename+='.gz'
    else:
        filename = databundle+'.cif.gz'
    
    destination_file = Path(cfg.cfg['paths']['nr_input_folder'])/filename

    if destination_file.parent.exists():
        with open(destination_file, 'wb') as out:
            for chunk in r.raw.stream(1024, decode_content=False):
              if chunk: 
                 out.write(chunk)    
    else:
        raise FileNotFoundError(f"{destination_file.parent} does not exist, cannot download to this location")
    
    return destination_file, databundle

def get_mirror_file_url(url=cfg.cfg['remote_resources']['nr_mirror_url'], 
                        ttdate=None):

    session = create_onelondon_session()

    rx = session.get(url, verify=False)

    logger.info("NB that PTSP are aware of the unverified certificate error but cannot solve it")

    files = sorted(re.findall(r"[0-9]{12}_full.gz", rx.text))

    if date is None and len(files)>0:
        tt = files[-1]
        outurl = url+tt
    elif len(files)>0:
        tt = [f for f in files if ttdate in f][0]
        outurl = url+tt
    else:
        outurl = None
        logger.warning(f"No file found matching {ttdate} at mirror URL {url}")
    return outurl, session

def get_bsline(BSlines, zdf, t):
    entry = zdf[t].index.tolist()
    idx = np.searchsorted(BSlines, entry, side='left') - 1
    return zdf['BS'].index[idx]
    
def extract_cif(input_cif):
    # structure = {table: fieldlengths}
    with open(Path(cfg.cfg['paths']['ref_nr_data_structure_folder'])/'cif_structure.pkl', 'rb') as f: structure = pickle.load(f) 
    #fields = {table: fieldnames}
    with open(Path(cfg.cfg['paths']['ref_nr_data_structure_folder'])/'cif_fields.pkl', 'rb') as f: fields = pickle.load(f) 
    #tables = {table: description}
    #with open(working_folder/lookup_folder/Path('cif_tables.pkl'), 'rb') as f: tables = pickle.load(f) 
    
    # Load data from CIF->text converted files, into a dict of dataframes
    zd = {key: [] for key in structure.keys()}
    zdf = {}
    
    print("Loading data from CIF file into dataframes...")
    with gzip.open(input_cif, 'rt') as f:
        for index, line in enumerate(f):
            tag = line[0:2]
            struct = structure[tag]
            #tablename = tables[tag]
            #fieldnames = fields[tag]
            linesep = [index]
            c1 = 0
            for c in struct:
                linesep.append(line.strip()[c1:c1+c])
                c1 += c
            
            zd[tag].append(linesep)
    
    for key in zd.keys():
        if len(zd[key]) > 0:
            print(key)
            col_length = np.max([len(x) for x in zd[key]])
            col_names = (['linenum']+fields[key])[0:col_length]
            zdf[key] =  pd.DataFrame(data=zd[key],
                                     columns=col_names
                                    ).set_index('linenum')
    
    # Get the date of the timetable extract
    tt_date_output = pd.to_datetime(zdf['HD']['Date of Extract'][0], dayfirst=True).strftime('%Y%m%d')
    
    # Create a reference to link together all items in a Run
    BSlines = zdf['BS'].index.tolist()
    
    for t in ['BX','LO','LI','LT','CR']:
        zdf[t]['BS_line'] = get_bsline(BSlines, zdf, t)
    
    nrtt_db_file = Path(cfg.cfg['paths']['nr_extracted_file'].format(databundle=tt_date_output))
    con = sqlite3.connect(nrtt_db_file)
    for key, df in zdf.items():
        df.to_sql(key, con, if_exists='replace') 
    con.close()
    print(f"Wrote to {nrtt_db_file}")

    return nrtt_db_file

def get_timetable(ttdate=None,
                  source='local',
                  fallback=True,
                  extract=None,
                  username=None,
                  password=None
                  ):
    """Obtains the National Rail timetable in gzipped CIF format and returns 
    the location, extracting it into an SQLite DB if requested.
    
    It can choose either a local file, 
    a specific date from the OpenTrainTimes mirror site, or
    the latest file from either OpenTrainTimes or the NR data feed 
    (username and password required).
    
    Parameters
    ----------  
    ttdate : str or None, optional
        Date (yyyymmdd) to search for, or None for the latest file online 
        (default is None)
        NB If mirrorfile and localfile are both False, then date is ignored 
           and the current timetable is requested from the NR API
        
    source:     str
                    local : already-downloaded local CIF or processed DB
                    mirror: full CIF from OpenTrainTimes
                    nr    : current CIF from Network Rail (ignores date)
         
    fallback:   bool
        Allow fallback to other sources if file cannot be found, in the order 
        of local > mirror > nr
    
    extract : bool or None (default)
        Whether to transform a local CIF file into a database or not
        None will decide based on the source (extract=True if local=False, others=True)
                
    username : str, optional
        Username for access to the Network Rail data feeds. 
        Required if localfile is None and mirrorfile is False.
                
    password : str, optional
        Password for access to the Network Rail data feeds

    Returns
    -------
    Path object
        The local path of the existing or newly-downloaded CIF file

    """        

    source = source.lower()
    locate_success = False

    if source not in ['local','mirror','nr']:
        raise ValueError("Expected one of 'local', 'mirror' or 'nr' as the source")
    
    if source =='local':
        if extract is None: extract = False
        downloaded_file, databundle = get_local_file(ttdate, extract)

        if downloaded_file is None:
            logger.warning(f"Could not find local file matching date {ttdate}")
        else:
            locate_success = True
    
    if (source=='mirror') or ((source =='local') and fallback and not locate_success):
        if extract is None: extract = True
        try:
            remote_url, session = get_mirror_file_url(ttdate=ttdate)
            downloaded_file, databundle = download(remote_url, session)
            locate_success = True
        except Exception as err:
            logger.warning(f"Could not find mirror file matching date {ttdate}: {err}")
            databundle = ttdate
    
    if (source =='nr') or (fallback and not locate_success):
        if extract is None: extract = True
        # Fetch latest file from Network Rail
        if username is None: username = input("A username is required for Network Rail Data Feeds: ")
        if password is None: password = getpass.getpass("A password is required for Network Rail Data Feeds: ")
    
        try:
            downloaded_file, databundle = download(username=username, password=password)
            locate_success = True
            logger.info(f"Remote file downloaded to {downloaded_file}")
        except Exception as err:
            raise ValueError(f"Could not retrieve & offer Network Rail timetable file: {err}")
    
    if locate_success and extract: 
        extracted_file = extract_cif(downloaded_file)
        out_file = extracted_file
    elif locate_success and not extract:
        out_file = downloaded_file
    else:
        raise ValueError("Could not locate the NR timetable file")

    return out_file, databundle


# %%
