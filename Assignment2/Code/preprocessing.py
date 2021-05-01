import pandas as pd
import torch.nn as nn
import torchfm.model.dfm as dfm
import dask.dataframe as dd
from tqdm import tqdm
import numpy as np
import gc

def get_labels(row):
    '''
    Creates label column based on variables 'click_bool' and 'booking_bool. 1 for only click
    5 for click and book and else 0.
    :param row: row of the DataFrame
    :return: either 0, 1 or 5
    '''
    if row['click_bool'] == 1:
        if row['booking_bool'] == 1:
            return 5
        else:
            return 1
    else:
        return 0


def convert_dtypes(df, id_cols):
    '''
    Sets dtype of column to minimal required dtype to speed up dummy encoding process
    :param df: DataFrame containing input data
    :param id_cols: Column names of columns that need to be converted to one-hot-encoded columns
    :return: df: DataFrame with adjusted dtypes
    '''

    for column in id_cols:
        if df[column].max() <= 255:
            df[column] = df[column].astype(np.uint8)
        elif df[column].max() <= 65535:
            df[column] = df[column].astype(np.uint16)
        elif df[column].max() <= 4294967295:
            df[column] = df[column].astype(np.uint32)

    return df


def one_hot_encode(df, id_cols):
    '''
    Creates one-hot encodings of columns containing ids in sparse format. Also stores the feature size of the id column
    in embedding_dims as this is required input for the DeepFM algorithm. Stores all intermediate one-hot-encoded
    dataframes in df_dummies_list
    :param df: input DataFrame
    :param id_cols: list of column names that should be encoded
    :return: DataFrame with all encodings
    '''

    print('Encoding dummies...')
    embedding_dims = []
    df_dummies_list = []

    for column in tqdm(id_cols):
        column_dummy = pd.get_dummies(df[column], sparse=True)
        embedding_dims.append(column_dummy.shape[1])
        df_dummies_list.append(column_dummy)

        # force garbage collector to clear memory
        del column_dummy
        gc.collect()

    df_dummy = pd.concat(df_dummies_list, axis=1)

    return df_dummy, embedding_dims

def preprocess():
    '''
    Preprocesses the input data to a dataframe with all id columns one-hot encoded
    :return: DataFrame with features and one-hot encoded categories. List of embedding dimensions needed for DeepFM
    '''
    print('Loading data...')
    df = pd.read_pickle('df_temporary.pkl')
    id_cols = ['srch_id', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id', 'srch_destination_id']
    non_id_cols = list(set(df.columns) - set(id_cols))

    # create label column and drop unneeded columns
    print('Creating label...')
    df['label'] = df.apply(lambda row: get_labels(row), axis=1)
    df.drop(['click_bool', 'booking_bool', 'date_time'], axis=1, inplace=True)

    df = convert_dtypes(df, id_cols)
    df_dummy, embedding_dims_id = one_hot_encode(df, id_cols)
    data_merged = pd.concat([df, df_dummy], axis=1)
    data_merged.drop(id_cols, axis=1, inplace=True) #drop id columns as these are now encoded

    data_merged.to_pickle('data_merged.pkl')

    # add feature dims of non_id_cols, this is alway one
    embedding_dims = [1]*len(non_id_cols)
    embedding_dims.extend(embedding_dims_id)

    return data_merged, embedding_dims