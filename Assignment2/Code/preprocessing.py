import pandas as pd
import torch.nn as nn
import torchfm.model.dfm as dfm
import dask.dataframe as dd
from tqdm import tqdm

def get_labels(row):
    if row['click_bool'] == 1:
        if row['booking_bool'] == 1:
            return 5
        else:
            return 1
    else:
        return 0


print('Loading data...')
df = pd.read_pickle('df_temporary.pkl')


#workaround to get categorical columns at end of dataframe
id_cols = ['srch_id', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id', 'srch_destination_id']
non_id_cols = list(set(df.columns)-set(id_cols))
cols_sorted = []
cols_sorted.extend(non_id_cols)
cols_sorted.extend(id_cols)
df = df.reindex(columns=cols_sorted)

print('Creating label...')
# create label column and drop bool_click and bool_book
df['label'] = df.apply(lambda row: get_labels(row), axis=1)
df.drop(['click_bool', 'booking_bool'], axis=1, inplace=True)

print('Encoding dummies...')
#create one hot for all id columns
embedding_dims = []
df_dummies = pd.DataFrame()
df_dummies = dd.from_pandas(df_dummies, npartitions=1)
for column in tqdm(id_cols):
    column_dummy = pd.get_dummies(df[column], sparse=True, dtype=int)
    column_dummy = dd.from_pandas(column_dummy, npartitions=1)
    embedding_dims.append(column_dummy.shape[1])
    df_dummies = df_dummies.append(column_dummy)
#
# df.drop(id_cols, axis=1, inplace=True)

# search_dummy = pd.get_dummies(df['srch_id'], sparse=True)
# search_dummy_first = search_dummy
# search_emb = nn.Embedding(search_dummy.shape[1], 3)
#
# print(search_dummy)
#
# df = df.append(search_dummy)
print(df.head())

# dfm.DeepFactorizationMachineModel()


