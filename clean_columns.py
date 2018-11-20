import numpy as np
import pandas as pd
import json

pd.set_option('display.max_columns', 500)

d = pd.read_csv("train.csv")

# Fields in json objects to keep, others are "not included in the demo dataset"
device_list = ['browser', 'operatingSystem', 'isMobile', 'deviceCategory']
geo_nets = ['continent', 'subContinent', 'country', 'region', 'city', 'networkDomain']

# Populate lists in dicts and then copy to dataframe for faster wrangling
t_dict = {'visits': np.zeros((len(d), 1)).astype(int), 
          'hits': np.zeros((len(d), 1)).astype(int), 
          'pageviews': np.zeros((len(d), 1)).astype(int), 
          'newVisits': np.zeros((len(d), 1)).astype(int), 
          'transactionRevenue': np.zeros((len(d), 1)).astype(int), 
          'bounces': np.zeros((len(d), 1)).astype(int)}
d_dict = {'browser': [], 'operatingSystem': [], 'isMobile': [], 'deviceCategory': []}
g_dict = {'continent': [], 'subContinent': [], 'country': [], 'region': [], 'city': [], 'networkDomain': []}

for i, stats in enumerate(d.totals):
    stats = json.loads(stats)
    for k in stats.keys():
        t_dict[k][i] = stats[k]

for key in t_dict.keys():
    d[key] = t_dict[key]

for i, stats in enumerate(d.device):
    stats = json.loads(stats)
    for k in device_list:
        d_dict[k].append(stats[k])

for key in d_dict.keys():
    d[key] = d_dict[key]

for i, stats in enumerate(d.geoNetwork):
    stats = json.loads(stats)
    for k in geo_nets:
        g_dict[k].append(stats[k])

for key in g_dict.keys():
    d[key] = g_dict[key]

d.drop(['geoNetwork','device','totals','trafficSource','socialEngagementType', 'visits'], axis=1, inplace=True)
d.to_csv('train_clean.csv', index=False)
print(d.head())


