# SYS6016 - Project Stage II - Charu Rawat (cr4zy) and Elena Gillis (emg3sc)

# Initial look at the data
# loading the packages
import pandas as pd
import datetime

#%% reading in the data
w_df = pd.read_csv('data/data_raw.csv')
print(w_df.shape)

#%% min and max prices
print('Price min:')
print(w_df['price'].min())
print('')
print('Price max:')
print(w_df['price'].max())
# prices are multiplied by 10000 to remove decimals


#%% Data preprocessing

# edit fields
w_df['book_event_type'] = w_df['book_event_type'].astype(str)
w_df['side'] = w_df['side'].astype(str)
w_df['side']  = w_df['side'].apply(lambda x :  x[2:3])
w_df['book_event_type']  = w_df['book_event_type'].apply(lambda x :  x[2:3])

# dropping cancellations
df1 = w_df[w_df['book_event_type'] != 'C']

# remove executed trades since they won't have a bid/ask
df1 = df1[df1['side'] != 'U']
print(df1['book_event_type'].value_counts())

# convert timestamp from UTC to Eastern
df1['timestamp_east'] = pd.to_datetime(df1['timestamp']).dt.tz_localize('UTC').dt.tz_convert('US/Eastern').dt.tz_localize(None)

# round up timestamp to the second
df1['date_second'] = df1['timestamp_east'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour,dt.minute,dt.second))
df1['date_hour'] = df1['timestamp_east'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour))
df1['date'] = df1['timestamp_east'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day))

#df1.drop(columns=['timestamp'], inplace= True)
df1 = df1.drop(['timestamp','timestamp_east','aux_quantity','aux1','aux2'], axis=1)

# count number of instances for each day
print(df1.groupby('date').size())

#group by date and pick max and min time
print(df1.groupby(['date'],sort = True)['date_second'].min())
print(df1.groupby(['date'],sort = True)['date_second'].max())

#%% import packages for plotting

# for plotting
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pylab
import matplotlib.dates as mdates

#%% plots to check health of data
# Event counts on a daily basis by the hour
sns.set_style("darkgrid")
fig = plt.figure(figsize=(20,8))
ax = fig.add_subplot(111)

df_plot = df1.groupby(by=['date_hour']).size().reset_index(name="counts")
df_plot = df_plot.sort_values(by=['date_hour'])
#df_plot.head()

# plot the trends
plt.plot('date_hour', 'counts', data = df_plot, color='mediumvioletred')

# format plots
fig.suptitle('Book Events by the hour for Wayfair (NYSE:W)', fontsize=20)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Event Count', fontsize=16)
pylab.legend(loc='upper right')
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda y, p: format(int(y), ',')))

# display
plt.show()

#%% BY THE Minute bid vs ask
sns.set_style("darkgrid")
fig = plt.figure(figsize=(20,8))
ax = fig.add_subplot(111)

df_plot = df1.groupby(by=['date_hour','side']).size().reset_index(name="counts")
df_plot = df_plot.sort_values(by=['date_hour'])
df_plot = df_plot.pivot(index='date_hour', columns='side', values='counts')
df_plot = df_plot.reset_index()
#df_plot.head()

# plot the trends
plt.plot('date_hour', 'A', data=df_plot, marker='o',color='tomato', label = 'Ask')
plt.plot('date_hour', 'B', data=df_plot, marker='o', color='deepskyblue', label = 'Bid')

# format plots
fig.suptitle('Ask and Bid events by the hour for Wayfair (NYSE:W)', fontsize=20)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Event Count', fontsize=16)
pylab.legend(loc='upper right')
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda y, p: format(int(y), ',')))

# display
plt.show()

# The bid represents demand and the ask represents supply for an asset
# low spread implies high liquidity 
# we can see below that demand is generally slightly higher for W stock

#%% write processed data to csv
df1.to_csv('data/data_processed.csv',index=None)

