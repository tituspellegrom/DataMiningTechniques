import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

DATA_FOLDER = 'data/'
FILE_NAME = 'dataset_mood_smartphone.csv'

df = pd.read_csv(DATA_FOLDER+FILE_NAME, index_col=0, parse_dates=['time'])
print(df.dtypes)

# Check which users we have, and how much
users = df['id'].unique().tolist()
n_users = len(users)

# Note: missing users
expected_users = {f"AS14.{i+1:02d}" for i in range(n_users)}
missing_users = expected_users.difference(users)

# how often per day are measurements made?
df['date'] = df['time'].dt.date
df_timeline = df.groupby(['id', 'variable', 'date']).size().unstack(level=0)

# Plot nr. of mood logs per user
df_moods = df_timeline.query('variable == "mood"').reset_index(level=0, drop=True)
fig = go.Figure([go.Scatter(x=df_moods.index, y=df_moods[usr], mode='lines+markers', name=usr)
                 for usr in df_moods.columns])
fig.show() # between 0 and 6 mood logs per day => some gaps










