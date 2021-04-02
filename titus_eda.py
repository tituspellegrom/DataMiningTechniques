import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from collections import ChainMap
pio.renderers.default = "browser"

DATA_FOLDER = 'data/'
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
group_score = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']
group_count = ['call', 'sms']
group_time = [col for col in df.columns if col.startswith('appCat.') or col == 'screen']

agg_fn_score = {col: ['min', 'max', 'median', 'mean', 'std', 'count'] for col in group_score}
agg_fn_count = {col: ['count'] for col in group_score}
agg_fn_time = {col: ['min', 'max', 'median', 'mean', 'std', 'count', 'sum'] for col in group_time}
agg_dict = ChainMap(agg_fn_score, agg_fn_count, agg_fn_time)

df_wide = df.copy()
df_wide = df_wide.pivot_table(index=['id', 'date'], columns='variable', values='value',
                              aggfunc=['min', 'max', 'median', 'mean', 'std', 'count', 'sum'])
df_wide = df_wide.swaplevel(0, 1, axis=1).sort_index(level='variable', axis=1)
dups = df_wide[df_wide.index.duplicated(keep=False)]    # suddenly duplicates??
df_wide.drop_duplicates(keep='last', inplace=True)

# heatmap mood
df_mood = df_wide.loc[:, ('mood', 'mean')].unstack(level='date').sort_index(level='date', axis=1)
fig = go.Figure(data=go.Heatmap(z=df_mood.values,
                                x=df_mood.columns,
                                y=df_mood.index.values,
                                colorscale='rdylgn',
                                zmid=5.5
                                ))
fig.show()








