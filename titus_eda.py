import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from collections import ChainMap
pio.renderers.default = "browser"
import matplotlib.pyplot as plt

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
group_score = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']
group_count = ['call', 'sms']
group_time = [col for col in df_wide.columns.get_level_values('variable') if col.startswith('appCat.') or col == 'screen']

agg_fn_score = {col: ['min', 'max', 'median', 'mean', 'std', 'count'] for col in group_score}
agg_fn_count = {col: ['count', ] for col in group_count}
agg_fn_time = {col: ['min', 'max', 'median', 'mean', 'std', 'count', 'sum'] for col in group_time}
agg_dict = ChainMap(agg_fn_score, agg_fn_count, agg_fn_time)

relevant_cols = [(col, fn) for col, fn_lst in agg_dict.items() for fn in fn_lst]
df_wide = df_wide.loc[:, relevant_cols]

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
                                ))
fig.show()
fig.write_html(PLOTS_FOLDER+"correlation.html")

# Basic ANN
# TODO: Encode user_id and add to features

df_collapse = df_wide.groupby('date').mean()
lookback = 3
l_history = [df_collapse.shift(i+1) for i in range(lookback)]

df_X = pd.concat(l_history, axis=1)
df_Y = df_collapse[('mood', 'mean')].shift(-1)

# Select rows where target is not NaN
data_idx = ~df_Y.isna()
df_X = df_X[data_idx]
df_Y = df_Y[data_idx]

# Split & Normalizing













