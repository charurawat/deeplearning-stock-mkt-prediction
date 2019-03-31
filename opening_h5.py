import pandas as pd
import os 

#%% Set path
path = 'C://Users/Elena/Desktop/Spring 19/ML/Project/Phase2'
os.chdir(path)

#%% Read in data
reread = pd.HDFStore('book_events_total_view_2017-01-09_new.h5')

# list data keys
keys = list(reread.keys())
pd.DataFrame(reread[keys[0]][:]).head()

#%% Extract Bank of America data
BAC = reread['/BAC'][['timestamp' ,'order_id', 'book_event_type', 'price', 'quantity', 'side']]
# print(BAC.shape) #(757503, 6)

#%% Write data to csv
BAC.to_csv('bac_new.csv', sep=',')
