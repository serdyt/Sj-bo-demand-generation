#!/usr/bin/env python
# coding: utf-8

# In[1]:


import geopandas
import pandas as pd
import numpy as np
import random
import math
from math import sqrt
import json
from shapely.geometry import Point


# In[2]:


pesrons = pd.read_feather('data/trips_others.feather')


# In[3]:


pesrons['home'] = pesrons.home.apply(lambda x: Point(x))
pesrons['coords'] = pesrons.coords.apply(lambda x: [Point(y) for y in x])


# In[4]:


pesrons


# In[5]:


time_type = pd.read_feather('data/time-distr-type.feather')
time_type.columns = time_type.columns.astype(int)


# In[6]:


time_type


# In[7]:


def stop_coords(stop_id):
    lat = stops.loc[stop_id]['stop_lat']
    lon = stops.loc[stop_id]['stop_lon']
    latlon = (lat,lon)
    return latlon


# In[8]:


class Population:
    def __init__(self):
        self.persons = []
    def to_JSON(self):
        return json.dumps(self, default=lambda o: _try(o), sort_keys=True, indent=4, separators=(',',':'))


class Person:
    def __init__(self):
        self.activities = []
        
#    def toJSON(self):
#        return json.dumps(self, default=lambda o: o.__dict__, 
#            sort_keys=True, indent=4)
        
#     def to_JSON(self):
#         return json.dumps(self, default=lambda o: _try(o), sort_keys=True, indent=4, separators=(',',':'))
        
def _try(o):
    try:
        if o.__class__ == ActivityType:
            raise Exception()
        return o.__dict__
    except:
        return str(o)

from enum import Enum
class ActivityType(Enum):
    HOME = 'HOME'
    WORK = 'WORK'

    def __str__(self):
        return self.name
    
class Activity(object):
    """Activity that is performed by a person.
    Person moves to a next activity after end_time is reached.

    Parameters
    ----------
    type_ : <ActivityType>
    coord : <Coord>
    start_time : <int> seconds from 00:00
    end_time : <int> seconds from 00:00
    """

    def __init__(self, type_, coord, start_time=None, end_time=None, zone=None):
        """docstring"""
        if start_time is None and end_time is None:
            raise Exception("Sanity check: both activity times are None")
        self.type = type_
        self.coord = coord
        self.start_time = start_time
        self.end_time = end_time
        self.zone = zone

    def __str__(self):
        return 'An ' + str(self.type) + ' at ' + str(self.coord)


# In[9]:


def form_datetime(x):
    if type(x) in [np.float64, float, int, np.int64, np.int32]:
#         try:
#             x2 = math.trunc(x)
#         except TypeError:
#             x2 = x
        
        m = (x - math.trunc(x)) * 60
        s = math.trunc((m - math.trunc(m)) * 60)
        m = math.trunc(m)
        h = math.trunc(x)
        return datetime(year = 1970, month = 1 ,day = 1, hour = h, minute=m, second = s)
    elif type(x) == time:
        return datetime(year = 1970, month = 1 ,day = 1, hour = x.hour, minute=x.minute, second = x.second)
    
def form_td(x):
    if type(x) == np.float64:
        m = (x - math.trunc(x)) * 60
        s = math.trunc((m - math.trunc(m)) * 60)
        m = math.trunc(m)
        h = math.trunc(x)
        return td(hours = h, minutes=m, seconds = s)
    elif type(x) == time:
        return td(hours = x.hour, minutes=x.minute, seconds = x.second)


# In[10]:


class Coord(object):
    """Coordinate.

    Parameters
    ----------
    lat : <float> latitude
    lon : <float> longitude
    latlon : <list> list with both lat and long. Latitude first!
    """
    def __init__(self, lat=None, lon=None, latlon=None):
        if latlon is not None:
            if len(latlon) != 2:
                raise Exception("Wrong coordinate latlon format. Should be a list of two floats.")
            self.lat = latlon[0]
            self.lon = latlon[1]
        elif lat is None or lon is None:
            raise Exception("Coordinates not provided")
        else:
            self.lat = lat
            self.lon = lon

    def to_json(self):
        return json.dumps(self, default=lambda o: self._try(o), sort_keys=True, indent=4, separators=(',', ':'))

    @staticmethod
    def _try(o):
        try:
            if o.__class__ == Coord:
                raise Exception()
            return o.__dict__
        except:
            return str(o)
        
    def __str__(self):
        return str(self.lat) + ',' + str(self.lon)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.lat == other.lat and self.lon == other.lon

    def __hash__(self):
        return hash((self.lat, self.lon))


# In[11]:


kommun = geopandas.read_file('data/kommuns.geojson')
kommun = kommun.to_crs('EPSG:4326')


# # persons to trips

# In[12]:


pesrons.head()


# In[13]:


pid = 0
population = Population()
for person in pesrons.itertuples():
    
    if len(person.trip_chain) == 0:
        continue
        
    for o, d, otype, dtype in zip(person.coords, person.coords[1:], [1] + person.trip_chain, person.trip_chain):

        trip = Person()
        trip.id = pid
        pid += 1
        trip.attributes = {'age': person.age}
        
        trip_start = time_type[dtype].sample(1, weights=time_type[dtype].values).index.values[0]*60

        act = Activity(otype,
                       Coord(latlon=[o.coords[0][1], o.coords[0][0]]),
                       start_time=0,
                       end_time=trip_start,
                       zone = int(kommun[kommun.contains(o)].kommun.values[0])
                      )
        trip.activities.append(act)
        
        act = Activity(dtype,
                       Coord(latlon=[d.coords[0][1], d.coords[0][0]]),
                       start_time=trip_start,
                       end_time=86400,
                       zone = int(kommun[kommun.contains(d)].kommun.values[0])
                      )
        trip.activities.append(act)
        population.persons.append(trip)


# In[14]:


j = json.loads(population.to_JSON())


# In[15]:


j


# In[16]:


with open('data/population_sjobo_others.json', 'w') as outfile:
    json.dump(j, outfile)


# In[ ]:




