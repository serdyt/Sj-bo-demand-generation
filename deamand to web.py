#!/usr/bin/env python
# coding: utf-8

# In[31]:


import geopandas
import pandas as pd
import numpy as np
import random
import math
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from shapely.geometry import Point


# In[32]:


persons_df = pd.read_feather('data/trips_others.feather')


# In[33]:


persons_df.chain_length.sum()


# In[34]:


import folium
import geopandas
from shapely.geometry import Point


# In[35]:


# persons_df['home'] = persons_df.home.apply(lambda x: Point(x))
# persons_df['coords'] = persons_df.coords.apply(lambda x: [Point(y) for y in x])


# In[38]:


persons_df


# In[39]:


kommun = geopandas.read_file('data/kommuns.geojson')
kommun = kommun.to_crs('EPSG:4326')


# In[40]:


# persons_df = persons_df.sample(100000)


# In[41]:


import itertools
flat_chain = list(itertools.chain(*persons_df.coords.values.tolist()))
# flat_chain = list(set(flat_chain))


# In[42]:


len(flat_chain)


# In[43]:


coords_kommun = []
for coord, i in zip(flat_chain, range(len(flat_chain))):
    coords_kommun.append(kommun[kommun.contains(Point(coord))].index[0])


# In[44]:


pd.DataFrame(coords_kommun, columns=['kommun']).to_feather('data/coords_kommun.feather')


# In[45]:


coords_kommun = pd.read_feather('data/coords_kommun.feather').kommun.to_list()


# In[46]:


len(coords_kommun)


# In[47]:


ks = (x for x in coords_kommun)
kf = (x for x in coords_kommun[1:])
dd = {}
for person in persons_df.itertuples():
#     print(person.id, person.coords)
    for sd, fd, skomm, fkomm in zip(person.coords, person.coords[1:], ks, kf):
#         print(sd, fd)
        if (skomm,fkomm) in dd:
            dd[(skomm,fkomm)] += 1
        elif (fkomm,skomm) in dd:
            dd[(fkomm,skomm)] += 1
        else:
            dd[(skomm,fkomm)] = 1
    try:
        next(ks)
        next(kf)
    except StopIteration:
        print('finnished')
        pass


# In[54]:


dd


# In[67]:


summ = sum([d for d in dd.values()])


# In[71]:


c=500


# In[72]:


COORD = [55.63135, 13.70622]
m = folium.Map(location=COORD, zoom_start=8)

for k,tr in dd.items():
    if -1 in k:
        continue
    s = kommun.loc[k[0]].geometry.centroid
    e = kommun.loc[k[1]].geometry.centroid
    if s == e:
        folium.Circle((s.coords[0][1], s.coords[0][0]), weight=tr/summ*c, tooltip=str(tr)).add_to(m)
    else:
        folium.PolyLine([(x.coords[0][1], x.coords[0][0]) for x in [s,e]], weight=tr/summ*c, tooltip=str(tr)).add_to(m)
m.save('data/demand.html')


# In[ ]:




