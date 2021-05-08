import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset
import numpy as np
import pickle
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

def create_label_column(df):
    df['label'] = 0
    book_indices = df['booking_bool'] == 1
    click_indices = df['click_bool'] == 1

    df['label'].loc[click_indices] = 1
    df['label'].loc[book_indices] = 5

    return df

def test_load():
    df = pd.read_pickle('df_temporary.pkl')
    # TODO Google drive path as argument
    df = create_label_column(df)
    df.drop(['click_bool', 'booking_bool', 'date_time'], axis=1, inplace=True)
    id_cols = pd.read_csv('df_temporary_idcols.csv')['id_columns'].values.tolist()
    non_id_cols = list(set(df.columns) - set(id_cols))
    # TODO scale non_id_cols
    scaled_features = StandardScaler().fit_transform(df[non_id_cols].values)
    df[non_id_cols] = pd.DataFrame(scaled_features, index=df[non_id_cols].index, columns=non_id_cols)
    df_cols = []
    df_cols.extend(non_id_cols)
    df_cols.extend(id_cols)
    df = df.reindex(columns=df_cols)
    return df

class SparseDataset(Dataset):
    def __init__(self, filename):
        # self.samples = sp.load_npz(filename).tocsr()
        self.samples = test_load()
    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        item = self.samples.iloc[idx].values

        categories = torch.LongTensor(item[15:])
        features = item[:15]
        x_normed = x / x.max(axis=0)
        label = item[13]

        return input, label
