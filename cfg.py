import configparser
import logging
import os
from pathlib import Path

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Gather configuration values from config.ini in same folder as this script
cfg = configparser.ConfigParser()
cfg.read(os.path.join(__location__,'config.ini'))

# Establish logging to info level to the console only
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)