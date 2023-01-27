#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import folium
import geopandas
from shapely.geometry import Point


# In[2]:


def draw_population(df, weights):
    df2 = df.loc[np.repeat(df.index.values, weights)]
    return df2


# # Data

# In[3]:


buildings = geopandas.read_file('data/by_get.shp')
buildings = buildings.to_crs("EPSG:4326")
buildings.geometry = buildings.geometry.centroid


# In[4]:


ruta1000 = geopandas.read_file('data/B1 ruta 1000 difference with 250.shp')
ruta1000 = ruta1000.to_crs("EPSG:4326")

ruta250 = geopandas.read_file('data/B1 ruta 250.shp')
ruta250 = ruta250.to_crs("EPSG:4326")

rutas = geopandas.GeoDataFrame(pd.concat([ruta250, ruta1000], axis=0))

rutas['Ruta'] = rutas.Ruta.astype(int)
rutas = rutas.set_index('Ruta')


# In[5]:


houses = buildings[buildings.ANDAMAL_1 < 200]
workplaces = geopandas.read_file('data/workplaces_OSM_SCB_merged.geojson')


# In[6]:


resfil_raw = pd.read_csv('data/RVU_resfil.csv')
individfil = pd.read_csv('data/RVU_individfil.csv')

resfil_raw = resfil_raw[resfil_raw['dagtyp'] == 1]
individfil = individfil[individfil['dagtyp'] == 1]


# In[7]:


resfil_raw['rf3a_ärende'] = resfil_raw.rf3a_ärende.replace(88.0, 14.0)
resfil_raw['rf3a_ärende'] = resfil_raw.rf3a_ärende.fillna(14.0)
resfil_raw['rf3a_ärende'] = resfil_raw['rf3a_ärende'].astype(int)


# # Change age grouping

# In[8]:


bins = [6, 15, 19, 24, 44, 64, 999]
age_groups = ['Ald7_15','Ald16_19','Ald20_24','Ald25_44','Ald45_64','Ald65_W'] # 'Ald0_6' are excluded
resfil_raw['age_scb'] = pd.cut(resfil_raw.alder_tab, bins=bins, labels=age_groups)
individfil['age_scb'] = pd.cut(individfil.alder_tab, bins=bins, labels=age_groups)


# # Find P(trip_chain|age)

# In[9]:


# freq_age = []
# for age in age_groups:
#     trip_chains = {}  
    
#     # no trips in a day for age bin
#     trip_chains[()] = individfil[(individfil['resa.antal'] == 0) & (individfil['age_scb'] == age)].individvikt.sum()
        
#     for pid in resfil_raw[resfil_raw.age_scb == age].Id.unique():    
#         trips = resfil_raw[resfil_raw.Id == pid]
#         chain = tuple(trips['rf3a_ärende'].to_list())
        
#         if any([math.isnan(x) for x in chain]):
#             chain = tuple(14.0 if math.isnan(x) else x for x in chain)
#         chain = tuple(14 if x==88 else x for x in chain)
        
#         if chain in trip_chains.keys():
#             trip_chains[chain] += trips['individvikt'].iloc[0]
#         else:
#             trip_chains[chain] = trips['individvikt'].iloc[0]
#     freq_age.append(pd.DataFrame.from_dict(trip_chains, orient='index',columns=[age]))
    
# freq_age = freq_age[0].join(freq_age[1], how='outer').join(freq_age[2], how='outer').join(freq_age[3], how='outer').join(freq_age[4], how='outer').join(freq_age[5], how='outer').fillna(0)


# In[10]:


# freq_age


# In[11]:


# freq_age = freq_age/freq_age.sum()


# In[12]:


# freq_age


# # distances between rutas

# In[13]:


rutas_distance = pd.read_feather("data/rutas_distance_matrix.feather")


# In[14]:


rutas_distance.columns = rutas_distance.columns.astype(int)
rutas_distance.index = rutas_distance.columns


# In[15]:


rutas_distance.head()


# In[16]:


rutas_with_buildings = pd.read_feather('data/rutas_with_buildings.feather')


# In[17]:


rutas_with_buildings['index'] = (rutas_with_buildings.reset_index().Ruta.astype(str).values + rutas_with_buildings.Rutstorl.astype(str).values).astype(int)
rutas_with_buildings = rutas_with_buildings.reset_index().set_index('index')


# In[18]:


rutas_with_buildings.head()


# In[19]:


rutas['index'] = (rutas.reset_index().Ruta.astype(str).values + rutas.Rutstorl.astype(str).values).astype(int)
rutas = rutas.reset_index().set_index('index')


# In[20]:


rutas.head()


# In[21]:


building_columns = ['houses_in_ruta', 'workplaces_in_ruta']


# In[22]:


rutas[building_columns] = rutas_with_buildings[building_columns]


# In[23]:


rutas.head()


# # Sampling (home location|ruta) (trip chain|age)

# In[24]:


columns = rutas.columns


# In[25]:


# persid = 0
# persons_df = None
# skipped_rutas = []
# for _,ruta in rutas.sample(1).iterrows():   
#     ruta = rutas.loc[3937506190000250]
# #     ruta_repeated = geopandas.GeoDataFrame(np.repeat([ruta.values], len(houses.index), axis=0), columns=columns, geometry='geometry', crs="EPSG:4326")
# #     ruta_buildings = houses[ruta_repeated.contains(houses.geometry)]
    
#     ruta_buildings = houses.loc[ruta.houses_in_ruta]    
    
#     ages = np.repeat(ruta[age_groups].index, ruta[age_groups].values)
#     ids = range(persid, persid + ruta.Totalt - ruta.Ald0_6)
#     persid += ruta.Totalt
# #     name is the 
#     ruta_df = np.array([ruta.Ruta]).repeat(len(ids))
#     rutstorl_df = np.array([ruta.Rutstorl]).repeat(len(ids))
    
#     # to make DF with person ID and ages
#     persons = pd.DataFrame({'id': ids,
#                             'age': ages,
#                            'Ruta': ruta_df,
#                            'Rutstorl': rutstorl_df}
#     )
    
#     #sample home location
#     if len(ruta_buildings.index) == 0:
#         # if ruta has no buildings, it can be outside skåne
#         skipped_rutas.append(str(ruta.Ruta))
#         continue
#         #         if ruta has no buildings, take its centroid
# #         persons['home'] = [ruta.geometry.centroid]*len(persons.index)
#     else:
#         persons['home'] = ruta_buildings.sample(n=len(persons.index), replace=True).geometry.values
    
#     #sample trip chain | age
#     chains = []
#     for n,age in zip(ruta[age_groups], age_groups):
#         chains.append(freq_age[age].sample(n=n, weights=freq_age[age].values, replace=True))
#     persons["trip_chain"] = pd.concat(chains).index
    
#     # save to dict
#     if persons_df is None:
#         persons_df = persons
#     else:
#         persons_df = pd.concat([persons_df,persons], axis=0, ignore_index=True)       
        
#     break
# print(skipped_rutas)


# In[26]:


# persons_df


# # Trip chain to trip pairs

# In[27]:


# persons_df['chain_length'] = persons_df.trip_chain.apply(len)


# In[28]:


# persons_df_repeated = persons_df.loc[np.repeat(persons_df.index.values, persons_df.chain_length)]


# In[29]:


# persons_df_repeated


# In[30]:


# trip_pairs = []
# for row in persons_df.itertuples():
#     chain = row.trip_chain
#     for f,s in zip([1.0] + [i for i in chain], chain):
#         trip_pairs.append((f,s))


# In[31]:


# len(trip_pairs)


# In[32]:


# persons_df_repeated['trip_pair'] = trip_pairs


# In[33]:


# persons_df_repeated


# # prepare trip length distributions

# In[34]:


upsampled_resfil = draw_population(resfil_raw[['rf7_km', 'age_scb', 'rf3a_ärende']], resfil_raw.individvikt)
upsampled_resfil = upsampled_resfil.dropna(subset=['rf7_km'])


# In[35]:


distance = np.linspace(1, 500, 500)


# In[36]:


# fig, ax = plt.subplots(nrows=6, ncols=17, figsize=(300,70))

# prob_age_type = {}

# for age,axj in zip(age_groups, range(0,7)):
#     loc_resfil = upsampled_resfil[upsampled_resfil['age_scb'] == age]
    
# #     fig, ax = plt.subplots(nrows=18, figsize=(7,30))
    
#     axi = loc_resfil.rf3a_ärende.unique().astype(int) - 1
#     axi.sort()
#     loc_resfil.hist(column=['rf7_km'], by=['rf3a_ärende'],bins=range(0,100, 1), density=True, ax=ax[axj][axi])

#     for axm, ind in zip(ax[axj].flat, range(1,18)):
#         df = loc_resfil[loc_resfil['rf3a_ärende']==ind]
#         if len(df.index) == 0:
#             continue
        
#         kde = KernelDensity(kernel='gaussian', bandwidth=5).fit(df['rf7_km'][:, np.newaxis])
#         log_dens = kde.score_samples(distance[:, np.newaxis])
#         probability = np.exp(log_dens)
        
#         prob_age_type[(age, ind)] = probability
        
#         axm.plot(distance, probability)
# fig.savefig('data/age-type-legth-hist.pdf')


# In[37]:


# prob_age_type = pd.DataFrame(prob_age_type)
# prob_age_type.index = distance.astype(int)
# prob_age_type.head()


# In[38]:


# prob_age_type = prob_age_type.reset_index(drop=True)
# prob_age_type.columns = pd.MultiIndex.from_tuples(prob_age_type.columns.to_series().apply(lambda x: (str(x[0]), str(x[1]))).values)
# prob_age_type.to_parquet('data/prob_age_type.parquet')


# In[39]:


prob_age_type = pd.read_parquet('data/prob_age_type.parquet')
prob_age_type.columns = pd.MultiIndex.from_tuples(prob_age_type.columns.to_series().apply(lambda x: (str(x[0]), int(x[1]))).values)
prob_age_type.index = distance.astype(int)
prob_age_type.head()


# In[40]:


# prob_type = {}
# fig, ax = plt.subplots(nrows=17, figsize=(7,30))
# upsampled_resfil.hist(column=['rf7_km'], by=['rf3a_ärende'],bins=range(0,100, 1), density=True, ax=ax)

# for axi, ind in zip(ax.flat, range(1,18)):
#     df = upsampled_resfil[['rf7_km', 'rf3a_ärende']][upsampled_resfil['rf3a_ärende']==ind]
#     kde = KernelDensity(kernel='gaussian', bandwidth=5).fit(df['rf7_km'][:, np.newaxis])
#     log_dens = kde.score_samples(distance[:, np.newaxis])
#     probability = np.exp(log_dens)

#     prob_type[ind] = probability

#     axi.plot(distance, probability)
# fig.savefig('data/type-legth-hist.pdf')


# In[41]:


# prob_type = pd.DataFrame(prob_type)
# prob_type.index = distance.astype(int)
# prob_type


# In[42]:


# prob_type = prob_type.reset_index(drop=True)
# prob_type.columns = prob_type.columns.astype(str)
# prob_type.to_parquet('data/prob_type.parquet')

prob_type = pd.read_parquet('data/prob_type.parquet')
prob_type.columns = prob_type.columns.astype(int)
prob_type.index = distance.astype(int)
prob_type.head()


# In[43]:


# prob_age = {}
# fig, ax = plt.subplots(nrows=6, figsize=(7,15))
# upsampled_resfil.hist(column=['rf7_km'], by=['age_scb'],bins=range(0,100, 1), density=True, ax=ax)

# for axi, age in zip(ax.flat, age_groups):
#     df = upsampled_resfil[['rf7_km', 'age_scb']][upsampled_resfil['age_scb']==age]
#     kde = KernelDensity(kernel='gaussian', bandwidth=5).fit(df['rf7_km'][:, np.newaxis])
#     log_dens = kde.score_samples(distance[:, np.newaxis])
#     probability = np.exp(log_dens)

#     prob_age[age] = probability

#     axi.plot(distance, probability)
# fig.savefig('data/age-legth-hist.pdf')


# In[44]:


# prob_age = pd.DataFrame(prob_age)
# prob_age.index = distance.astype(int)
# prob_age


# In[45]:


# prob_age.to_parquet('data/prob_age.parquet')


# In[46]:


prob_age = pd.read_parquet('data/prob_age.parquet')
prob_age.head()


# In[47]:


upsampled_resfil.hist(column=['rf7_km'], bins=range(0,500, 1), density=True)


# # Add buildings of other types

# In[48]:


rutas_other_buildings = pd.read_feather('data/rutas_with_other_buildings.feather')


# In[49]:


rutas_other_buildings = rutas_other_buildings.reset_index()


# In[50]:


rutas_other_buildings.index = (rutas_other_buildings.Ruta.astype(str).values + rutas_other_buildings.Rutstorl.astype(str).values).astype(int) 


# In[51]:


rutas_other_buildings


# In[52]:


print(rutas.loc[3495006225750250].geometry.centroid)


# In[53]:


rutas = pd.concat([rutas, rutas_other_buildings], axis=1)


# # building coordinates

# In[54]:


healthcare = geopandas.read_file('data/healthcare.geojson')
healthcare = healthcare.geometry

school = geopandas.read_file('data/schools.geojson')
school = school.geometry

pickup_kids = geopandas.read_file('data/pickup_kids.geojson')
pickup_kids = pickup_kids.geometry

shopping_livsmedel = geopandas.read_file('data/shopping_livsmedel.geojson')
shopping_livsmedel = shopping_livsmedel.geometry

shopping_other = geopandas.read_file('data/shopping_other.geojson')
shopping_other = shopping_other.geometry

bank_post_mndighet = geopandas.read_file('data/bank_post_mndighet.geojson')
bank_post_mndighet = bank_post_mndighet.geometry

sport_outdoor = geopandas.read_file('data/sport_outdoor.geojson')
sport_outdoor = sport_outdoor.geometry

association_activity = geopandas.read_file('data/association_activity.geojson')
association_activity = association_activity.geometry

leisure = geopandas.read_file('data/leisure.geojson')
leisure = leisure.geometry

waste = geopandas.read_file('data/waste.geojson')
waste = waste.geometry

goods = geopandas.read_file('data/goods.geojson')
goods = goods.geometry


# In[55]:


building_coordinates = {
    1: buildings,                           # 1: home
    2: workplaces,                           # 2: work
    3: buildings,                    # 3: tjänst
    4: school,                    # 4: education
    5: pickup_kids,               # 5: pickup kids
    6: shopping_livsmedel,        # 6: shopping livsmedel         
    7: shopping_other,            # 7: other shopping
    8: healthcare,                # 8: hospital
    9: bank_post_mndighet,        # 9: visit bank/post/myndighet
    10: sport_outdoor,            # 10: motion
    11: association_activity,     # 11: föreningsaktivitet
    12: leisure,                  # 12: leisure
    13: buildings,                   # 13: visit friends/relatives
    14: buildings,                   # 14: other
    15: waste,                    # 15: soptippen
    16: goods,                    # 16: pickup goods
    17: buildings,                   # 17: summer house
}


# # Assign buildings

# In[56]:


# persons_df['home_ruta_storlek'] = (persons_df.Ruta.astype(str).values + persons_df.Rutstorl.astype(str).values).astype(int)


# In[57]:


ellipse_trips = [2, 4, 8]
trip_type_to_name = {
    1: 'houses_in_ruta',                    # 1: home
    2: 'workplaces_in_ruta',                # 2: work
    3: 'houses_in_ruta',                    # 3: tjänst
    4: 'school_in_ruta',                    # 4: education
    5: 'pickup_kids_in_ruta',               # 5: pickup kids
    6: 'shopping_livsmedel_in_ruta',        # 6: shopping livsmedel         
    7: 'shopping_other_in_ruta',            # 7: other shopping
    8: 'healthcare_in_ruta',                # 8: hospital
    9: 'bank_post_mndighet_in_ruta',        # 9: visit bank/post/myndighet
    10: 'sport_outdoor_in_ruta',            # 10: motion
    11: 'association_activity_in_ruta',     # 11: föreningsaktivitet
    12: 'leisure_in_ruta',                  # 12: leisure
    13: 'houses_in_ruta',                   # 13: visit friends/relatives
    14: 'houses_in_ruta',                   # 14: other
    15: 'waste_in_ruta',                    # 15: soptippen
    16: 'goods_in_ruta',                    # 16: pickup goods
    17: 'houses_in_ruta',                   # 17: summer house
                 }

# 88: error   - should be filtered out already


# In[58]:


# persons_df_save = persons_df.copy()
# persons_df_save['home'] = persons_df_save.home.apply(lambda x: x.coords[0])
# persons_df_save.to_parquet('data/persons_df.parquet')


# In[59]:


p = pd.read_parquet('data/persons_df.parquet')
p['home'] = p.home.apply(lambda x: Point(x))


# In[60]:


persons_df = p


# In[61]:


def get_building_coords(trip_type, building_id):
    b = building_coordinates[trip_type].loc[building_id]
    if type(b) == pd.core.series.Series:
        b = b.geometry
    return b


# In[69]:


persons_chains = []
persons_coords = []
trip_distances = []
# sample = persons_df.sample(100)
sample = persons_df
for person in sample.itertuples():
    
#     if person.trip_chain == ():
#         continue
    retry_person = True
    while retry_person:
        retry_person = False
    
        chain_ruta = []
        chain_building = []
#         trip_distance_log = []
#         rutas_within_trip_distance_ind_log = []
#         chain_build_id = []

        if person.home_ruta_storlek not in rutas_distance.index:
                print('TDM is all wrong!')
                raise Exception

        ellipse_type = any([x in person.trip_chain for x in ellipse_trips])

        if ellipse_type:
            # take first hit from ellipse_trips as the maindestination
            # it will be the second focus of the ellipse
            # home location is the first ellipse
            main_purpose = [p for p in ellipse_trips if p in person.trip_chain][0]

            found = False
            # lc = 0
            while not found:
                # lc += 1            
                trip_distance_main = prob_age_type[(person.age, main_purpose)].sample(1, weights=prob_age_type[(person.age, main_purpose)].values).index[0]
                trip_distance_main = trip_distance_main * 1000

                home_distance_row = rutas_distance[person.home_ruta_storlek]
                dist_interval = 1000 #+ trip_distance_main * 0.1
                rutas_within_trip_distance_ind = home_distance_row[(home_distance_row.values < trip_distance_main) & (home_distance_row.values > trip_distance_main - dist_interval)].index
                rutas_within_trip_distance = rutas.loc[rutas_within_trip_distance_ind]

                rutas_within_trip_distance['weight'] = rutas_within_trip_distance[trip_type_to_name[main_purpose]].apply(len)
                if rutas_within_trip_distance['weight'].sum() == 0:
                    continue
                rutas_within_trip_distance['weight'] = rutas_within_trip_distance['weight']/rutas_within_trip_distance['weight'].sum()
                main_ruta = rutas_within_trip_distance.sample(1, weights=rutas_within_trip_distance['weight'])
                main_building_ind = np.random.choice(main_ruta[trip_type_to_name[main_purpose]].values[0])

                trip_distance_main = max(1000, home_distance_row.loc[main_ruta.index].values[0])

                main_distance_row = rutas_distance[main_ruta.index[0]]
                ellipse_distances = home_distance_row + main_distance_row
                # a = sqrt((trip_distance_main/2)**2 + (trip_distance_main/2)**2)
                rutas_ellipse_ind = ellipse_distances[ellipse_distances.values < 2*trip_distance_main]
                habitable_rutas = rutas.loc[rutas_ellipse_ind.index]

                found = True
                # print(lc)

        else: # just a circle around home location
            trip_distance_main = prob_age_type[(person.age, 1)].sample(1, weights=prob_age_type[(person.age, 1)].values).index[0]
            trip_distance_main = trip_distance_main * 1000 * 1.1

            home_distance_row = rutas_distance[person.home_ruta_storlek]
            rutas_within_trip_distance_ind = home_distance_row[home_distance_row.values < trip_distance_main].index
            rutas_within_trip_distance = rutas.loc[rutas_within_trip_distance_ind]

            habitable_rutas = rutas_within_trip_distance
            main_purpose = 1

        if len(habitable_rutas.index) == 0:
            raise Exception('!!')

        chain_building.append(person.home)
#         chain_build_id.append(None)
        chain_ruta.append(person.home_ruta_storlek)
#         trip_distance_log.append(0)

        # TODO: add special (1,1) case
        # TODO: skip a trip if there if cannot find a building for some time (or use all buildings)

        for trips, tripe in zip([1] + person.trip_chain.tolist(), person.trip_chain):
#             print((trips, tripe))
            if retry_person:
                break
            
            if (trips, tripe) == (1,1):
                trips = 14

            if tripe == main_purpose and ellipse_type == True:
                # TODO: buildings here won't work for schools at least
                # try main_ruta[trip_type_to_name[main_purpose]]
                chain_building.append(get_building_coords(tripe, main_building_ind)) 
#                 chain_build_id.append(None)
                chain_ruta.append(main_ruta.index[0])
#                 print('main location special case')
                continue

            if tripe == 1:
                chain_building.append(person.home) 
#                 chain_build_id.append(None)
                chain_ruta.append(person.home_ruta_storlek)
#                 print('return home special case')
                continue

            origin_ruta = chain_ruta[-1]

            found = False
            lc2 = 0
#             print('location loop')
            while not found:
                lc2 += 1
                if lc2 == 50:
#                     print('could not find a trip {} for {}'.format(tripe, str(person)))
#                     print('tring home trip instead')
                    tripe = 1
                elif lc2 == 100:
                    retry_person = True
#                     print('could not find a trip {} for {}'.format(tripe, str(person)))
#                     print('retrying the whole person')
#                     print(lc2)
                    break
                    # raise Exception
                reasonable_distance_range = max(1, int(trip_distance_main/1000*1.1))

                if (person.age, tripe) in prob_age_type.columns:
                    trip_distance = prob_age_type[(person.age, tripe)].loc[:reasonable_distance_range].sample(1, weights=prob_age_type[(person.age, tripe)].loc[:reasonable_distance_range].values).index[0]
                else:
                    trip_distance = prob_type[tripe].loc[:reasonable_distance_range].sample(1, weights=prob_type[tripe].loc[:reasonable_distance_range].values).index[0]
                trip_distance = trip_distance * 1000

                if trip_distance > trip_distance_main * 0.8:
                    # print('trip much longer than main trip')
                    continue

                origin_distance_row = rutas_distance[origin_ruta]
                dist_interval = 1000 #+ trip_distance * 0.1
                rutas_within_trip_distance_ind = origin_distance_row[(origin_distance_row.values < trip_distance) & (origin_distance_row.values > trip_distance - dist_interval)].index
                rutas_within_trip_distance = habitable_rutas.loc[habitable_rutas.index.intersection(rutas_within_trip_distance_ind)]

                rutas_within_trip_distance['weight'] = rutas_within_trip_distance[trip_type_to_name[tripe]].apply(len)
                if rutas_within_trip_distance['weight'].sum() == 0:
                # print('cannot find rutas within trip distance')
                    continue
                rutas_within_trip_distance['weight'] = rutas_within_trip_distance['weight']/rutas_within_trip_distance['weight'].sum()

                destination_ruta = rutas_within_trip_distance.sample(1, weights=rutas_within_trip_distance['weight'])
                
                destination_building_ind = np.random.choice(destination_ruta[trip_type_to_name[tripe]].values[0])

                chain_ruta.append(destination_ruta.index[0])
                chain_building.append(get_building_coords(tripe, destination_building_ind))

#                 trip_distance_log.append(trip_distance)
#                 rutas_within_trip_distance_ind_log.append(rutas_within_trip_distance_ind)

                found = True
                # print(lc2)
    persons_chains.append(chain_ruta)
    persons_coords.append(chain_building)
#     trip_distances.extend(trip_distance_log)
    
#     ax = habitable_rutas.plot()
#     geopandas.GeoDataFrame(chain_building, columns=['geometry']).plot(ax=ax, color='y')


sample['coords'] = [tuple([(y.coords[0]) for y in x]) for x in persons_coords]


# In[ ]:


sample['home'] = sample.home.apply(lambda x: x.coords[0])


# In[ ]:


sample = sample.reset_index()


# In[ ]:


sample.to_feather('data/trips_others.feather')


# In[ ]:


# sample_df = pd.read_feather('data/trips_others_sample.feather')


# In[ ]:




