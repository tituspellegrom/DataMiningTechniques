import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from collections import ChainMap
pio.renderers.default = "browser"
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'
from keras.models import Sequential
from keras.layers import Dense
from livelossplot import PlotLossesKeras
import tensorflow as tf
import itertools

DATA_FOLDER = 'data/'
PLOTS_FOLDER = 'plots/'
FILE_NAME = 'dataset_mood_smartphone.csv'

df = pd.read_csv(DATA_FOLDER+FILE_NAME, index_col=0, parse_dates=['time'])

# TODO: remove negative time values
df.dropna(subset=['value'], inplace=True)
df['date'] = df['time'].dt.date

# TODO: Decide on duplicates removal => or let pivot_table handle?
# df = df.set_index(['id', 'time', 'variable']).sort_index()
# duplicates = df[df.index.duplicated(keep=False)]
# df = df.groupby(level=[df.index.names]).agg({'value': np.mean})    # Remove duplicates by taking mean
# duplicates2 = df[df.index.duplicated(keep=False)]


# Check which users we have, and how much
users = df['id'].unique().tolist()
n_users = len(users)

# Note: missing users
expected_users = {f"AS14.{i+1:02d}" for i in range(n_users)}
missing_users = expected_users.difference(users)

# Full feature aggregation per day
df_wide = df.copy()
df_wide = df_wide.pivot_table(index=['id', 'date'], columns='variable', values='value',
                              aggfunc=['min', 'max', 'median', 'mean', 'std', 'count', 'sum'])
df_wide = df_wide.swaplevel(0, 1, axis=1).sort_index(level='variable', axis=1)
dups = df_wide[df_wide.index.duplicated(keep=False)]    # suddenly duplicates??
df_wide.drop_duplicates(keep='last', inplace=True)

# Drop irrelevant aggregations
cols_score = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']
cols_count = ['call', 'sms']
cols_time = [col for col in df['variable'].unique().tolist() if col.startswith('appCat.') or col == 'screen']

agg_fn_score = {col: ['min', 'max', 'median', 'mean', 'std', 'count'] for col in cols_score}
agg_fn_count = {col: ['count', ] for col in cols_count}
agg_fn_time = {col: ['min', 'max', 'median', 'mean', 'std', 'count', 'sum'] for col in cols_time}
agg_dict = ChainMap(agg_fn_score, agg_fn_count, agg_fn_time)

relevant_cols = [(col, fn) for col, fn_lst in agg_dict.items() for fn in fn_lst]
df_wide = df_wide.loc[:, relevant_cols]

# Fill NaN's
# Time & Count features
df_filled = df_wide[cols_count+cols_time].fillna(0) # Workaround due to MultiIndex on columns fucking up
df_wide.fillna(df_filled, inplace=True)

# TODO: Forward Fill per user_id => Now polluted
df_filled = df_wide[cols_score].fillna(method='ffill')   # use last known value
df_wide.fillna(df_filled, inplace=True)

# Maybe day of week contains information (weekend=happy?)
df_wide['day_of_week'] = df_wide.index.get_level_values(level='date').dayofweek
df_wide[['mon', 'thue', 'wed', 'thu', 'fri', 'sat', 'sun']] = pd.get_dummies(df_wide['day_of_week']) # saturday happiest

# heatmap mood
df_mood = df_wide.loc[:, ('mood', 'mean')].unstack(level='date').sort_index(level='date', axis=1)
fig = go.Figure(data=go.Heatmap(z=df_mood.values,
                                x=df_mood.columns,
                                y=df_mood.index.values,
                                colorscale='rdylgn',
                                zmid=5.5
                                ))
fig.show()
fig.write_html(PLOTS_FOLDER+"mood_heatmap.html")


# Data Relationships
pd.plotting.scatter_matrix(df_wide.loc[:, [('mood', 'mean'), ('circumplex.arousal', 'mean')]])


df_corr = df_wide.groupby('date').mean().corr()
fig = go.Figure(data=go.Heatmap(z=df_corr.values,
                                x=list(map(lambda c: '-'.join(c), df_corr.axes[1])),
                                y=list(map(lambda c: '-'.join(c), df_corr.axes[0])),
                                zmid=0
                                ))
fig.show()
fig.write_html(PLOTS_FOLDER+"correlation.html")

# Basic ANN
# TODO: Encode user_id and add to features
df_collapse = df_wide.copy()
df_collapse.dropna(inplace=True)
df_collapse = df_collapse.groupby('date').mean()
lookback = 3
l_history = [df_collapse.shift(i+1) for i in range(lookback)]

df_X = pd.concat(l_history, axis=1).dropna()
df_Y = df_collapse[('mood', 'mean')].shift(-1).dropna().to_frame()

# Select rows where both are not NaN
idx = df_Y.index.intersection(df_X.index)
X = df_X.loc[idx, :].values
y = df_Y.loc[idx, :].values

# Split & Normalizing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# TODO: don't scale the category columns
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
y_train = sc.fit_transform(y_train)
y_test = sc.transform(y_test)
print(X_train)
print(X_test)

m_x, m_y = X_train.shape[1], y_train.shape[1]

model = Sequential()
model.add(Dense(400, activation='relu'))
model.add(Dense(m_y, activation='relu'))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(X_train, y_train, epochs=100,
          callbacks=[PlotLossesKeras()], batch_size=X_train.shape[0])

Y_pred = model.predict(X_test)













