import pandas as pd
from tqdm import tqdm
import numpy as np
import gc
import pickle
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sp


def create_label_column(df):
    df['label'] = 0
    book_indices = df['booking_bool'] == 1
    click_indices = df['click_bool'] == 1

    df['label'].loc[click_indices] = 1
    df['label'].loc[book_indices] = 5


    return df


def create_label_column2(df):
    df['label'] = 0
    book_indices = df['booking_bool'] == 1
    click_indices = df['click_bool'] == 1

    df['label'].loc[click_indices] = 1
    df['label'].loc[book_indices] = 5


    return df['label'].to_numpy()


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


def one_hot_encode2(X):
    '''
    Creates one-hot encodings of numpy array containing categorical data, in sparse format.
    Also stores the feature size of the categories per column
    in ids_dims as this is required input for the DeepFM algorithm. Stores all intermediate one-hot-encoded
    array in a list
    :param X: input numpy array
    :return: X_encoded with all encodings, ids_dims as a list containing dimensions
    '''

    print('Encoding dummies...')
    X_enc_list = []

    enc = OneHotEncoder(sparse=True)

    n, m = X.shape
    for i in tqdm(range(m)):
        X_i = X[:, i].reshape(-1, 1)

        X_i_enc = enc.fit_transform(X_i)

        X_enc_list += [X_i_enc]

    X_enc = sp.hstack(X_enc_list)
    ids_dims = [X_enc.shape[1] for X_enc in X_enc_list]

    return X_enc, ids_dims


def preprocess2(data_name):
    '''
    Preprocesses the input data to a dataframe with all id columns one-hot encoded
    :return: DataFrame with features and one-hot encoded categories. List of embedding dimensions needed for DeepFM
    '''

    print('Loading data...')

    df = pd.read_pickle(f'../{data_name}.pkl')
    groups = df['srch_id'].to_numpy()

    # id_cols = ['srch_id', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id', 'srch_destination_id']
    # pd.DataFrame(id_cols, columns=['id_columns']).to_csv(f'{data_name}_idcols.csv', index=False)

    X_label = create_label_column2(df).reshape(-1, 1)
    df.drop(['click_bool', 'booking_bool', 'date_time'], axis=1, inplace=True)

    id_cols = pd.read_csv(f'{data_name}_idcols.csv')['id_columns'].values.tolist()
    non_id_cols = list(set(df.columns) - set(id_cols))

    X_non_ids = df[non_id_cols].to_numpy()
    X_ids = df[id_cols].to_numpy()

    X_ids_enc, ids_dims = one_hot_encode2(X_ids)
    X_enc = sp.hstack([X_non_ids.astype(float), X_ids_enc.astype(float), X_label.astype(float)])

    non_ids_dims = [1] * X_non_ids.shape[1]
    embedding_dims = np.array(non_ids_dims+ids_dims+[1]) # label also dimension 1 => is this needed?

    sp.save_npz(f'{data_name}_data_merged.npz', X_enc)
    embedding_dims.tofile(f'{data_name}_embedding_dims.txt')
    groups.tofile(f'{data_name}_groups.txt')

    return X_enc, embedding_dims, groups


def preprocess():
    '''
    Preprocesses the input data to a dataframe with all id columns one-hot encoded
    :return: DataFrame with features and one-hot encoded categories. List of embedding dimensions needed for DeepFM
    '''
    print('Loading data...')
    df = pd.read_pickle('df_temporary.pkl')
    groups = df['srch_id'].values
    id_cols = ['srch_id', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id', 'srch_destination_id']
    non_id_cols = list(set(df.columns) - set(id_cols))

    df = convert_dtypes(df, id_cols)
    df_dummy, embedding_dims_id = one_hot_encode(df, id_cols)
    data_merged = pd.concat([df, df_dummy], axis=1)
    data_merged.drop(id_cols, axis=1, inplace=True) #drop id columns as these are now encoded

    # create label column and drop unneeded columns
    print('Creating label...')
    # df['label'] = df.apply(lambda row: get_labels(row), axis=1)
    # df['label'] = df['label'].astype('category')
    data_merged = create_label_column(data_merged)
    data_merged.drop(['click_bool', 'booking_bool', 'date_time'], axis=1, inplace=True)

    new_columns = [str(i) for i in range(0, data_merged.shape[1])]
    data_merged.columns = new_columns
    # data_merged.reindex(columns=new_columns, copy=False)

    print('Saving data...')

    data_merged.to_pickle('data_merged.pkl')


    # add feature dims of non_id_cols, this is alway one
    embedding_dims = [1]*len(non_id_cols)
    embedding_dims.extend(embedding_dims_id)

    #save embedding_dims and groups

    with open('embedding_dims.txt', 'wb') as fp:
        pickle.dump(embedding_dims, fp)

    with open('groups.txt', 'wb') as fp:
        pickle.dump(groups, fp)

    return data_merged, embedding_dims, groups


if __name__ == '__main__':
    preprocess2('df_temporary')
    preprocess2('df_features')

