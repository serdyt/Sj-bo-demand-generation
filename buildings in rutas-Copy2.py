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


# # Data

# In[2]:


ruta1000 = geopandas.read_file('data/B1 ruta 1000 difference with 250.shp')
ruta1000 = ruta1000.to_crs("EPSG:4326")

ruta250 = geopandas.read_file('data/B1 ruta 250.shp')
ruta250 = ruta250.to_crs("EPSG:4326")

rutas = geopandas.GeoDataFrame(pd.concat([ruta250, ruta1000], axis=0))

rutas['Ruta'] = rutas.Ruta.astype(int)
rutas = rutas.set_index('Ruta')


# In[3]:


rutas


# # Other types

# In[5]:


healthcare = geopandas.read_file('data/healthcare.geojson')
school = geopandas.read_file('data/schools.geojson')
pickup_kids = geopandas.read_file('data/pickup_kids.geojson')
shopping_livsmedel = geopandas.read_file('data/shopping_livsmedel.geojson')
shopping_other = geopandas.read_file('data/shopping_other.geojson')
bank_post_mndighet = geopandas.read_file('data/bank_post_mndighet.geojson')
sport_outdoor = geopandas.read_file('data/sport_outdoor.geojson')
association_activity = geopandas.read_file('data/association_activity.geojson')
leisure = geopandas.read_file('data/leisure.geojson')
waste = geopandas.read_file('data/waste.geojson')
goods = geopandas.read_file('data/goods.geojson')


# In[40]:


healthcare_in_ruta = []
school_in_ruta = []
pickup_kids_in_ruta = []
shopping_livsmedel_in_ruta = []
shopping_other_in_ruta = []
bank_post_mndighet_in_ruta = []
sport_outdoor_in_ruta = []
association_activity_in_ruta = []
leisure_in_ruta = []
waste_in_ruta = []
goods_in_ruta = []

arrays = (healthcare_in_ruta, school_in_ruta, pickup_kids_in_ruta, shopping_livsmedel_in_ruta,
shopping_other_in_ruta, bank_post_mndighet_in_ruta, sport_outdoor_in_ruta, association_activity_in_ruta,
leisure_in_ruta, waste_in_ruta, goods_in_ruta)

sources = (healthcare, school, pickup_kids, shopping_livsmedel ,shopping_other,
bank_post_mndighet, sport_outdoor , association_activity, leisure, waste, goods)

columns = rutas.reset_index().columns
r = rutas
i = 0
for ruta in r.itertuples():
    print(i)
    i += 1
    for source, array in zip(sources, arrays):
        ruta_repeated = geopandas.GeoDataFrame(np.repeat([ruta], len(source.index), axis=0), columns=columns, geometry='geometry', crs="EPSG:4326")
        ruta_source_ind = source[ruta_repeated.contains(source.geometry, align=False).values].index
        array.append(ruta_source_ind.to_list())


# In[41]:


names = ('healthcare_in_ruta', 'school_in_ruta', 'pickup_kids_in_ruta', 'shopping_livsmedel_in_ruta',
'shopping_other_in_ruta', 'bank_post_mndighet_in_ruta', 'sport_outdoor_in_ruta', 'association_activity_in_ruta',
'leisure_in_ruta', 'waste_in_ruta', 'goods_in_ruta')

for name, array in zip(names, arrays):
    r[name] = array


# In[42]:


r.drop(['geometry', 'Ald0_6','Ald7_15','Ald16_19','Ald20_24','Ald25_44','Ald45_64','Ald65_W','Totalt'], axis=1).to_feather('data/rutas_with_other_buildings.feather')


# In[ ]:




