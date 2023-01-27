#!/usr/bin/env python
# coding: utf-8

# In[1]:


import geopandas
import pandas as pd
import numpy as np
import random
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


# # Data

# In[2]:


buildings = geopandas.read_file('data/by_get.shp')
buildings = buildings.to_crs("EPSG:4326")
buildings.geometry = buildings.geometry.centroid


# In[3]:


ruta1000 = geopandas.read_file('data/B1 ruta 1000 difference with 250.shp')
ruta1000 = ruta1000.to_crs("EPSG:4326")

ruta250 = geopandas.read_file('data/B1 ruta 250.shp')
ruta250 = ruta250.to_crs("EPSG:4326")

rutas = geopandas.GeoDataFrame(pd.concat([ruta250, ruta1000], axis=0))

rutas['Ruta'] = rutas.Ruta.astype(int)
rutas = rutas.set_index('Ruta')


# In[4]:


houses = buildings[buildings.ANDAMAL_1 < 200]
workplaces = geopandas.read_file('data/workplaces_OSM_SCB_merged.geojson')


# In[ ]:


houses_in_ruta = []
workplaces_in_ruta = []
columns = rutas.reset_index().columns
r = rutas
for ruta in r.itertuples():
    ruta_repeated = geopandas.GeoDataFrame(np.repeat([ruta], len(houses.index), axis=0), columns=columns, geometry='geometry', crs="EPSG:4326")
    ruta_houses_ind = houses[ruta_repeated.contains(houses.geometry, align=False).values].index
    houses_in_ruta.append(ruta_houses_ind.to_list())
    
    ruta_repeated = geopandas.GeoDataFrame(np.repeat([ruta], len(workplaces.index), axis=0), columns=columns, geometry='geometry', crs="EPSG:4326")
    ruta_workplaces_ind = workplaces[ruta_repeated.contains(workplaces.geometry, align=False).values].index
    workplaces_in_ruta.append(ruta_workplaces_ind.to_list())


# In[74]:


r['houses_in_ruta'] = houses_in_ruta
r['workplaces_in_ruta'] = workplaces_in_ruta


# In[75]:


r


# In[36]:


r.drop('geometry', axis=1).to_feather('data/rutas_with_buildings.feather')


# In[38]:


pd.read_feather('data/rutas_with_buildings.feather')


# In[ ]:




