# SYS6016 - Project Stage II - Charu Rawat (cr4zy) and Elena Gillis (emg3sc)

import pandas as pd
from scipy import stats

#%% Read in processed data
df1 = pd.read_csv('data/data_processed.csv')
print(df1.shape)

#%% Transfoming data to get midprice
# obtain min of ask and max of bid for every timestamp
a_min = df1[df1['side'] == 'A'].groupby(['date_second'], as_index=False).agg({'price': [min]})
b_max = df1[df1['side'] == 'B'].groupby(['date_second'], as_index=False).agg({'price': [max]})

a_min.columns = a_min.columns.droplevel(level=0)
a_min.columns = ['date_second', 'min_ask']

b_max.columns = b_max.columns.droplevel(level=0)
b_max.columns = ['date_second', 'max_bid']

df2 = pd.DataFrame({'date_second': df1['date_second'].unique()}) # get all unique timestamps
df2 = df2.merge(a_min, how = 'left')
df2 = df2.merge(b_max, how = 'left')

# impute missing values for missing bi and ask values
df2['max_bid'].interpolate(method='linear', inplace=True)
df2['min_ask'].interpolate(method='linear', inplace=True)

#calculate absolute raw midprice
df2['midprice'] = (df2['max_bid'] + df2['min_ask']) /2

# calculate prop change in midprice for bid-ask values
df2['min_diff'] = (df2['min_ask']/df2['midprice']) - 1
df2['max_diff'] = (df2['max_bid']/df2['midprice']) - 1

# mid price change
df2['midprice_lag'] = df2['midprice'].shift(1)
df2['midprice_change'] = df2['midprice_lag']/ df2['midprice'] - 1

#%% Normalization and moving averages

# Z- Score normalization  of the midprice
df2['midprice_norm'] = stats.zscore(df2['midprice'])

# moving average of normalized midprice
window_size = 60 # for 60 seconds
df2['movavg_midprice_norm'] = df2['midprice_norm'][::-1].rolling(window=window_size, min_periods=0).mean()[::-1]

#%% computing labels - Change in midprice using normalized values

# penalizing factor or tuning parameter - 
# will effect how sensitive do we want our mid price change to be to the data
alpha = 0.001

# function to define label for change in mid price movement
def get_label(moving_avg, norm_mid_price, alpha):
    if (moving_avg/ norm_mid_price) > (1 + alpha):
        return 1
    elif (moving_avg/ norm_mid_price) < (1 - alpha):
        return -1
    return 0

df2['label'] = df2.apply(lambda x: get_label(x['movavg_midprice_norm'], x['midprice_norm'], alpha), axis=1)
# moving avg is calculated over norm midprice
# only the midprice_change is over non-normalized values

# print count for labels
print(df2['label'].value_counts())

#%% Create 60 second windows in data

# check null values
print(df2.isna().sum())

# drop null values
df3 = df2.dropna()
df3.shape

# our x values are changes in midprice that are non-normalized since they are ratios already

def lag_cols(df,shift_level = 60):
    
    ### Creating lag values
    all_lagged_columns = pd.concat([df['midprice_change'].shift(each) for each in range(shift_level)],axis = 1)
    
    ## Colnames 
    cols = [ list(['midprice_change_lag'+str(each)]) for each in range(shift_level)]
    flat_list = [item for sublist in cols for item in sublist]
    
    all_lagged_columns.columns = flat_list
    
    return(all_lagged_columns)

# new dataframe with lag columns
df4 = lag_cols(df3)

df4.dropna(inplace = True)
df4.shape # should drop 60 rows


#%% Create dataframe with x input features and label column
ind = list(df4.index)
df5 = pd.concat([df4, df3.loc[ind]['label']], axis = 1)

#%% Save traspormed data
df5.to_csv('data/data_transformed.csv',index=None)