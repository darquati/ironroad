# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:48:32 2018

@author: benjamingreen
"""
import requests
import json
import os
import json
import pdb
import time
import re
import socket
import matplotlib.pyplot as plt
import networkx as nx





###
###Config section
###

#Please change to your personal API credentials 
api_credentials = "app_id=8268063a&app_key=14f7f5ff5d64df2e88701cef2049c804"

#Script will save the results in a file so there's no need to pull data each time
pull_from_file_if_present = True    

#List of modes that you want to include in the graph (note: bus is pretty big)
modes_to_pull = [
    #"bus", 
    "dlr",
    "national-rail",
    "overground",
    "tflrail",
    "tram",
    "tube"]







###
###Main code section
###

##Annoyingly, on a onelondon machine you need to specify a proxy to connect to the internet
if socket.getfqdn()[-20:]=='.onelondon.tfl.local':
    proxy = 'https://webproxy.onelondon.tfl.local:8080'
    
    proxy_dict = {
        'http' : proxy,
        'https' : proxy
    }
else:
    proxy_dict = {}


#wrapping the api pull in a function
#Looks a bit complicated but it's so I can try a second time if it fails, cache a local copy of the API response
def pull_from_api(url, credential_string, pull_from_file, retries=3):
    full_url = url +'?'+api_credentials
    filename = re.sub(r"[^A-Za-z0-9]+", '', url)

    if pull_from_file==True and os.path.exists('api_cache\\'+filename+".json"):
        with open('api_cache\\'+filename+".json", 'r') as infile:
            api_response = json.load(infile)
            return api_response
    ##stop trying once retries gets low
    elif retries>0:
        try:
            api_response = requests.get(full_url, proxies = proxy_dict)
            if api_response.status_code != 200:
                raise
            ##write the api response to a file so it's quick next time
            if not os.path.exists('api_cache'):
                os.makedirs('api_cache')
            with open('api_cache\\'+filename+".json", 'w+') as outfile:
                json.dump(api_response.json(), outfile)
            return api_response.json()
        except Exception as e:
            time.sleep(5)
            return pull_from_api(url, credential_string, False, retries-1)
    else:
        print("Couldn't pull "+full_url)
        pdb.set_trace()


#join the modes into the right format to be used as an API input
mode_string = '%2C'.join(modes_to_pull)
#pull a list of lines for the specified modes
line_api_url = 'https://api.tfl.gov.uk/Line/Mode/'+mode_string
api_line_datas = pull_from_api(line_api_url, api_credentials, pull_from_file_if_present)


#create a list of all the lines we want to pull
lines = [x['id'] for x in api_line_datas]    


print('Pulled a list of lines, now to pull all the data for each line from the api')
#create empty data structures for nodes, edges (and a progress counter)
master_nodes = {}
master_edges= []
progress_counter = 0


##both directions because there are some single-direction edges in tram / bus
for direction in ['inbound', 'outbound']:
    #for each of the lines
    for line in lines:
        ##try to do a bunch of stuff for that line, print errors otherwise
        try:
            line_url = 'https://api.tfl.gov.uk/Line/'+line+'/Route/Sequence/'+direction
            api_line_data = pull_from_api(line_url, api_credentials, pull_from_file_if_present)
            
            
            ##add each station (or bus stop ) to the nodes with some info
            for station in api_line_data['stations']:
                                
                station_id = station['id']
                station_info = station
                del station_info['$type']
                station_info['children'] = []
                station_info['lines'] = [x['id'] for x in station_info['lines']]
                station_info['lines'].sort()
                
                if station_id in master_nodes:
                    #check the dicts agree everywhere except children
                    if {x:master_nodes[station_id][x] for x in master_nodes[station_id] if x!='children'} !=  {x:station_info[x] for x in station_info if x!='children'}:
                        print('The same station has different info in two places')
                        pdb.set_trace()
                else:
                    master_nodes[station_id] = station_info
            
            
            
            line_id = api_line_data['lineId']
            mode = api_line_data['mode']
            #A line might have multiple stop point sequences (e.g. where the central line branches)
            #Loop over them all and add the edges to our master edges data
            for stop_point_sequence in api_line_data['stopPointSequences']:
                for i in range(len(stop_point_sequence['stopPoint'])-1):
                    start_id = stop_point_sequence['stopPoint'][i]['topMostParentId']
                    end_id = stop_point_sequence['stopPoint'][i+1]['topMostParentId']
                    
                    
                    ##for some reason this particular station isn't getting mapped up to the Naptan Hub that I think it should be
                    if start_id == '9400ZZLUBNK8':
                        start_id = 'HUBBAN'
                    
                    if end_id == '9400ZZLUBNK8':
                        end_id = 'HUBBAN'
                    
                    if start_id in master_nodes.keys() and end_id in master_nodes.keys() and start_id!=end_id:
                        master_edges.append({'start_id':start_id, 'end_id':end_id, 'mode':mode, 'line_id': line_id, 'direction':direction})
                            
                
        except Exception as inst:
            print('Error! Error!')
            print(line)
            print(inst)
    
        
        progress_counter += 1
        if progress_counter % 5 ==0:
            print(str(progress_counter)+'/'+str(len(lines)*2)+' completed')
        




###
###End of main code, couple of quick examples of how to use the data
###

print('finished collecting data')


##draw as a bunch of lines with matplotlib
print("Drawing the network geographically with matplotlib")


for edge in master_edges:

    y = [master_nodes[edge['start_id']]['lat'], master_nodes[edge['end_id']]['lat']]
    x = [master_nodes[edge['start_id']]['lon'], master_nodes[edge['end_id']]['lon']]
    plt.plot(x, y, color='b')

    
plt.show

plt.figure()
##Convert to network x structure (with all the attributes in place)
print("converting the data to a Network X graph (with all the attribute data)")

Graph = nx.MultiDiGraph()

for key in master_nodes.keys():
    Graph.add_node(key, attr_dict = master_nodes[key])


for row in master_edges:
    Graph.add_edge(row['start_id'], row['end_id'], attr_dict = {x: row[x] for x in row.keys() if x not in ['start_id', 'end_id']})

print("Drawing the network X graph")
nx.drawing.draw(Graph, arrows=True, with_labels=False, node_size=30, alpha=0.5)

