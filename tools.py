#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 12:07:29 2021

@author: davearquati
"""

import configparser
import getpass
import logging
import os
import time

import numpy as np
import pandas as pd
import requests

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
config = configparser.ConfigParser()
config.read(os.path.join(__location__,'config.ini'))

proxy_required = config['connections'].getboolean('proxy_required')
proxy_auth_required = config['connections'].getboolean('proxy_auth_required')

proxies = {k: v for k, v in config['proxies'].items()}

def create_onelondon_session(proxy_required=proxy_required, 
                             auth_required=proxy_auth_required, 
                             proxies=proxies):

    logging.getLogger(__name__).info("Setting up a OneLondon proxy session...")
    session = requests.Session()   
    if auth_required:
        proxy_username = input("Please confirm your OneLondon username to send to the TfL proxy server: ")
        proxy_password = getpass.getpass("OneLondon password: ")
        session.proxies = {k: v.format(auth=f'{proxy_username}:{proxy_password}@') for k, v in proxies.items()}
    elif proxy_required:
        session.proxies = {k: v.format(auth='') for k, v in proxies.items()}
    return session
    
class Timer():
    def __init__(self, label=None, accuracy=4):
        self.a = time.time()
        self.accuracy=accuracy
        if label is not None:
            print(label)
    def stop(self):
        self.b = time.time()
        self.duration = self.b - self.a
        elapsis = r"{:."+str(self.accuracy)+"}"
        print((elapsis+" seconds elapsed").format(self.duration))
        return self.duration

def compass_direction(xdiffs, ydiffs):
        return (-np.rad2deg(np.arctan2(xdiffs, ydiffs)) + 90) % 360
    
def pythag_distance(xdiffs, ydiffs):
    return np.sqrt(xdiffs**2 + ydiffs**2)

def compass_to_cardinal(compass_direction):
    cardinals =  pd.cut(compass_direction.squeeze(), 
                        bins=np.arange(-22.5,360+22.5*2,45), 
                        labels=['N','NE','E','SE','S','SW','W','NW','N2'])\
                    .replace({'N2':'N'})
    return cardinals

